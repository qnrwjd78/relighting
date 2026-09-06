from __future__ import annotations

import argparse
import copy
from collections import OrderedDict
from contextlib import nullcontext
from datetime import datetime
import importlib
import inspect
import json
import math
import os
import random
import shutil
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import accelerate
import torch
from torch import nn
from tqdm import tqdm

from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import ImageCropAndResize, LoadAudio, LoadVideo, ToAbsolutePath
from diffsynth.diffusion import *  # noqa: F403 - DiffSynth exposes trainer utilities here.
try:
    from diffsynth.diffusion.runner import OffloadTrainingManager  # type: ignore
except ImportError:
    from diffsynth.core import OffloadTrainingManager  # type: ignore

try:
    from diffsynth.diffusion.runner import get_optimizer_class  # type: ignore
except ImportError:
    def get_optimizer_class(customized_optimizer=None):
        if customized_optimizer is None:
            return torch.optim.AdamW
        module_name, class_name = customized_optimizer.rsplit(".", 1)
        module = importlib.import_module(module_name)
        print(f"Customized optimizer `{customized_optimizer}` imported.")
        return getattr(module, class_name)

try:
    from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing  # type: ignore
except ImportError:
    def initialize_deepspeed_gradient_checkpointing(accelerator):
        if getattr(accelerator.state, "deepspeed_plugin", None) is None:
            return
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" not in ds_config:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
            return
        import deepspeed

        act_config = ds_config["activation_checkpointing"]
        deepspeed.checkpointing.configure(
            mpu_=None,
            partition_activations=act_config.get("partition_activations", False),
            checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
            contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False),
        )
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

from model.light_encoder import LightokenEncoder, attrs_from_batch
from model.wan import TokenLightTypeEmbedding, model_fn_wan_video_tokenlight


os.environ["TOKENIZERS_PARALLELISM"] = "false"

TOKENLIGHT_DEFAULT_PROMPT = "photorealistic object relighting, preserve geometry and materials"
DEFAULT_TRAIN_CONFIG_PATH = "configs/train_config.json"
DEFAULT_SINGLE_TRAIN_CONFIG_PATH = "configs/train_single.json"
DEFAULT_ZERO3_TRAIN_CONFIG_PATH = "configs/train_zero3.json"
TRAIN_CONFIG_BY_MODE = {
    "single": DEFAULT_SINGLE_TRAIN_CONFIG_PATH,
    "zero3": DEFAULT_ZERO3_TRAIN_CONFIG_PATH,
}


def install_zero3_loader_compat() -> None:
    """Provide the private HF ZeRO-3 loader expected by DiffSynth when missing."""

    try:
        import transformers.integrations.deepspeed as ds_integration
    except Exception:
        return
    if hasattr(ds_integration, "_load_state_dict_into_zero3_model"):
        return

    def _load_state_dict_into_zero3_model(model_to_load, state_dict, load_config=None):
        del load_config
        import deepspeed

        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata
        error_msgs = []
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        def load(module, prefix: str = "") -> None:
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            params = dict(module.named_parameters(prefix=prefix[:-1], recurse=False))
            params_to_gather = [param for name, param in params.items() if name in state_dict]
            context = (
                deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0)
                if params_to_gather
                else nullcontext()
            )
            with context:
                if rank == 0:
                    module._load_from_state_dict(
                        state_dict,
                        prefix,
                        local_metadata,
                        True,
                        [],
                        [],
                        error_msgs,
                    )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(model_to_load)
        for name, buffer in model_to_load.named_buffers():
            value = state_dict.get(name)
            if isinstance(value, torch.Tensor):
                buffer.data.copy_(value.to(device=buffer.device, dtype=buffer.dtype))
        return error_msgs

    ds_integration._load_state_dict_into_zero3_model = _load_state_dict_into_zero3_model


install_zero3_loader_compat()


def _as_frames(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _normalize_wan_lora_target_modules(value):
    """Map common Diffusers attention names to DiffSynth Wan module names."""
    if not isinstance(value, str) or not value:
        return value
    aliases = {
        "to_q": "q",
        "to_k": "k",
        "to_v": "v",
        "to_out.0": "o",
        "to_out": "o",
    }
    parts = [part.strip() for part in value.split(",") if part.strip()]
    mapped = [aliases.get(part, part) for part in parts]
    if mapped != parts:
        warnings.warn(
            "Mapped Diffusers LoRA target names to DiffSynth Wan names: "
            f"{','.join(parts)} -> {','.join(mapped)}",
            stacklevel=2,
        )
    return ",".join(mapped)


def _load_checkpoint_state_dict(checkpoint_path: str | None) -> dict[str, torch.Tensor]:
    if checkpoint_path in (None, "", "None", "null"):
        return {}
    path = Path(str(checkpoint_path))
    if not path.exists() or path.is_dir():
        return {}
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, dict):
        for key in ("state_dict", "module", "model"):
            value = loaded.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(value, torch.Tensor) for value in loaded.values()):
            return loaded
    return {}


def _extract_module_state(state_dict: dict[str, torch.Tensor], module_prefix: str) -> dict[str, torch.Tensor]:
    result = {}
    prefixes = (f"{module_prefix}.", f"module.{module_prefix}.")
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                result[key[len(prefix) :]] = value
                break
    return result


def _load_module_state_from_checkpoint(module: nn.Module | None, state_dict: dict[str, torch.Tensor], prefix: str) -> None:
    if module is None:
        return
    module_state = _extract_module_state(state_dict, prefix)
    if not module_state:
        return
    missing, unexpected = module.load_state_dict(module_state, strict=False)
    print(
        f"Loaded {prefix} from checkpoint: "
        f"tensors={len(module_state)}, missing={len(missing)}, unexpected={len(unexpected)}"
    )


def _split_csv_paths(value) -> list[str]:
    if value in (None, "", "None", "none", "null"):
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _resolve_path(value: str | Path, *, base_path: str | Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    base = Path(base_path) if base_path not in (None, "") else Path.cwd()
    if not base.is_absolute():
        base = Path.cwd() / base
    return base / path


def _relative_to_or_none(path: Path, root: Path) -> str | None:
    try:
        return path.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()
    except ValueError:
        return None


class VaeLatentCache:
    def __init__(self, cache_dirs: list[str], *, max_open_shards: int = 4) -> None:
        if not cache_dirs:
            raise ValueError("At least one VAE latent cache directory is required.")
        self.caches: list[dict] = []
        self.max_open_shards = max(1, int(max_open_shards))
        self._open_shards: OrderedDict[tuple[int, str], dict[str, torch.Tensor]] = OrderedDict()
        for cache_dir in cache_dirs:
            self._add_cache(Path(cache_dir))
        if not self.caches:
            raise ValueError(f"No VAE latent cache indexes found in: {cache_dirs}")

    @staticmethod
    def _read_json(path: Path) -> dict:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _index_files(root: Path) -> list[Path]:
        merged = root / "index.jsonl"
        if merged.exists():
            return [merged]
        return sorted(root.glob("index_part_*.jsonl"))

    def _add_cache(self, cache_dir: Path) -> None:
        root = _resolve_path(cache_dir)
        if not root.exists():
            raise FileNotFoundError(f"Missing VAE latent cache directory: {root}")
        config = self._read_json(root / "cache_config.json") or self._read_json(root / "cache_summary.json")
        data_root_value = config.get("data_root")
        data_root = _resolve_path(data_root_value) if data_root_value else None
        entries: dict[str, dict] = {}
        for index_path in self._index_files(root):
            with index_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    path = row.get("path")
                    if isinstance(path, str) and path:
                        entries[path] = row
        if not entries:
            raise FileNotFoundError(f"No VAE latent cache index rows found under {root}")
        self.caches.append({"root": root, "data_root": data_root, "entries": entries})
        print(
            "Loaded VAE latent cache index: "
            f"root={root} data_root={data_root} assets={len(entries)}"
        )

    def _candidate_paths(self, value: str, *, dataset_base_path: str | Path | None) -> list[tuple[int, str]]:
        raw = Path(value)
        abs_path = _resolve_path(raw, base_path=dataset_base_path)
        candidates: list[tuple[int, str]] = []
        for cache_index, cache in enumerate(self.caches):
            data_root = cache.get("data_root")
            if data_root is not None:
                rel = _relative_to_or_none(abs_path, data_root)
                if rel is not None:
                    candidates.append((cache_index, rel))
            if not raw.is_absolute():
                candidates.append((cache_index, raw.as_posix()))
        return candidates

    def _load_shard(self, cache_index: int, shard: str) -> dict[str, torch.Tensor]:
        key = (cache_index, shard)
        cached = self._open_shards.get(key)
        if cached is not None:
            self._open_shards.move_to_end(key)
            return cached
        from safetensors.torch import load_file

        shard_path = self.caches[cache_index]["root"] / shard
        tensors = load_file(str(shard_path), device="cpu")
        self._open_shards[key] = tensors
        self._open_shards.move_to_end(key)
        while len(self._open_shards) > self.max_open_shards:
            self._open_shards.popitem(last=False)
        return tensors

    def load(self, value: str, *, dataset_base_path: str | Path | None) -> torch.Tensor:
        for cache_index, rel_path in self._candidate_paths(value, dataset_base_path=dataset_base_path):
            entry = self.caches[cache_index]["entries"].get(rel_path)
            if entry is None:
                continue
            shard = entry.get("shard")
            tensor = entry.get("tensor")
            if not isinstance(shard, str) or not isinstance(tensor, str):
                raise KeyError(f"Malformed cache index entry for {value!r}: {entry}")
            tensors = self._load_shard(cache_index, shard)
            if tensor not in tensors:
                raise KeyError(f"Tensor {tensor!r} not found in cache shard {shard!r}")
            return tensors[tensor]
        raise KeyError(f"Could not find {value!r} in any VAE latent cache.")


class VaeLatentCachedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        cache: VaeLatentCache,
        *,
        dataset_base_path: str | Path | None,
        target_key: str = "video",
        source_key: str = "input_image",
    ) -> None:
        self.dataset = dataset
        self.cache = cache
        self.dataset_base_path = dataset_base_path
        self.target_key = target_key
        self.source_key = source_key
        self.data = dataset.data
        self.repeat = getattr(dataset, "repeat", 1)
        self.load_from_cache = False

    def __len__(self) -> int:
        return len(self.data) * int(self.repeat)

    def __getitem__(self, index: int) -> dict:
        row = self.data[index % len(self.data)].copy()
        target_path = row.get(self.target_key)
        if not isinstance(target_path, str) or not target_path:
            raise KeyError(f"Missing target image key {self.target_key!r} in metadata row")
        row["input_latents"] = self.cache.load(target_path, dataset_base_path=self.dataset_base_path)
        source_path = row.get(self.source_key)
        if isinstance(source_path, str) and source_path:
            row["tokenlight_source_latents"] = self.cache.load(source_path, dataset_base_path=self.dataset_base_path)
        return row


def _require_nonempty_light_attrs(attrs: list[dict], *, key: str) -> list[dict]:
    missing = [index for index, item in enumerate(attrs) if not item]
    if missing:
        raise ValueError(
            f"TokenLight light tokens are enabled, but {key} is missing or empty for "
            f"batch item(s) {missing}. This would train with null light tokens only. "
            f"Check that metadata.csv contains {key} and that the dataloader preserves it."
        )
    return attrs


def _float_or_none(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if torch.isfinite(torch.tensor(number)).item() else None


def _triple_or_none(value):
    values = list(value) if isinstance(value, (list, tuple)) else []
    values = (values + [None, None, None])[:3]
    return [_float_or_none(item) for item in values]


def _attrs_from_blender_relight_row(row: dict) -> dict[str, float]:
    x, y, z = _triple_or_none(row.get("canonical_position"))
    r, g, b = _triple_or_none(row.get("rgb_color"))
    attrs = {
        "a": _float_or_none(row.get("ambient_scale")),
        "x": x,
        "y": y,
        "z": z,
        "r": r,
        "g": g,
        "b": b,
        "lambda": _float_or_none(row.get("lambda_intensity")),
        "d": _float_or_none(row.get("radius")),
    }
    return {key: value for key, value in attrs.items() if value is not None}


def _scene_id_from_row(row: dict) -> str | None:
    scene_id = row.get("scene_id")
    if isinstance(scene_id, str) and scene_id:
        return scene_id
    for key in ("video", "image", "input_image", "ambient_image"):
        value = row.get(key)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if not isinstance(item, str):
                continue
            for part in Path(item).parts:
                if part.startswith("scene_"):
                    return part
    return None


def _dataset_file_exists(base_path: str | None, value) -> bool:
    if value in (None, ""):
        return False
    if isinstance(value, (list, tuple)):
        return all(_dataset_file_exists(base_path, item) for item in value)
    if not isinstance(value, str):
        return True
    path = Path(value)
    if not path.is_absolute():
        path = Path(base_path or ".") / path
    return path.exists()


def _row_values(value):
    return value if isinstance(value, list) else [value]


def _candidate_scene_meta_paths(base_path: str | None, row: dict) -> list[Path]:
    candidates: list[Path] = []
    for key in ("video", "image", "input_image", "ambient_image"):
        for item in _row_values(row.get(key)):
            if not isinstance(item, str) or not item:
                continue
            path = Path(item)
            if not path.is_absolute():
                path = Path(base_path or ".") / path
            for parent in path.parents:
                if parent.name.startswith("scene_"):
                    candidates.append(parent / "meta.json")
                    break
    scene_id = row.get("scene_id")
    if isinstance(scene_id, str) and scene_id.startswith("scene_"):
        candidates.append(Path(base_path or ".") / "scenes" / scene_id / "meta.json")

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = path.as_posix()
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _load_row_light_validity(base_path: str | None, row: dict, cache: dict[str, tuple[bool, dict[int, bool]]]):
    meta_path = next((path for path in _candidate_scene_meta_paths(base_path, row) if path.exists()), None)
    if meta_path is None:
        return None
    cache_key = meta_path.resolve().as_posix()
    if cache_key in cache:
        return cache[cache_key]

    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        cache[cache_key] = (False, {})
        return cache[cache_key]

    scene_valid = meta.get("valid") is not False
    light_valid: dict[int, bool] = {}
    point_lights = meta.get("spatial", {}).get("point_lights", [])
    if isinstance(point_lights, list):
        for light in point_lights:
            if not isinstance(light, dict) or "id" not in light:
                continue
            try:
                light_valid[int(light["id"])] = light.get("valid") is not False
            except (TypeError, ValueError):
                continue

    cache[cache_key] = (scene_valid, light_valid)
    return cache[cache_key]


def _filter_tokenlight_metadata_rows(rows: list[dict], base_path: str | None) -> tuple[list[dict], dict[str, int]]:
    filtered = []
    stats = {
        "row_invalid": 0,
        "missing_file": 0,
        "missing_or_bad_scene_meta": 0,
        "invalid_scene": 0,
        "invalid_light": 0,
    }
    scene_cache: dict[str, tuple[bool, dict[int, bool]]] = {}
    for row in rows:
        if row.get("valid") is False:
            stats["row_invalid"] += 1
            continue

        if not _dataset_file_exists(base_path, row.get("video")) or not _dataset_file_exists(base_path, row.get("input_image")):
            stats["missing_file"] += 1
            continue

        validity = _load_row_light_validity(base_path, row, scene_cache)
        if validity is None:
            stats["missing_or_bad_scene_meta"] += 1
            continue
        scene_valid, light_valid = validity
        if scene_valid is False:
            stats["invalid_scene"] += 1
            continue
        light_ids = row.get("light_ids")
        if light_ids is None:
            light_ids = [row.get("light_id")] if row.get("light_id") is not None else []
        if not isinstance(light_ids, list):
            light_ids = [light_ids]
        invalid_light = False
        for light_id in light_ids:
            try:
                light_id_int = int(light_id)
            except (TypeError, ValueError):
                light_id_int = None
            if light_id_int is not None and light_valid.get(light_id_int) is False:
                stats["invalid_light"] += 1
                invalid_light = True
                break
        if invalid_light:
            continue

        filtered.append(row)
    return filtered, stats


def _normalize_tokenlight_metadata_rows(rows: list[dict]) -> list[dict]:
    normalized = []
    for row in rows:
        if "video" in row and "input_image" in row:
            item = dict(row)
            item["prompt"] = item.get("prompt") or TOKENLIGHT_DEFAULT_PROMPT
            normalized.append(item)
            continue
        if "image" not in row or "ambient_image" not in row:
            normalized.append(row)
            continue
        attrs = _attrs_from_blender_relight_row(row)
        if not attrs:
            continue
        normalized.append(
            {
                "video": row["image"],
                "input_image": row["ambient_image"],
                "prompt": TOKENLIGHT_DEFAULT_PROMPT,
                "attrs_json": json.dumps(attrs, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
                "scene_id": row.get("scene_id"),
                "light_id": row.get("light_id"),
                "valid": row.get("valid"),
            }
        )
    return normalized


def _load_json_config(path: str | None) -> dict:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_train_config(config: dict) -> dict:
    ignored = {"launch"}
    flat = {}

    def visit(value):
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            if key in ignored:
                continue
            if isinstance(item, dict):
                visit(item)
            else:
                flat[key] = item

    visit(config)
    return flat


def _parser_destinations(parser: argparse.ArgumentParser) -> set[str]:
    return {action.dest for action in parser._actions if action.dest != argparse.SUPPRESS}


def _apply_config_defaults(parser: argparse.ArgumentParser, config: dict) -> None:
    destinations = _parser_destinations(parser)
    defaults = {key: value for key, value in _flatten_train_config(config).items() if key in destinations}
    parser.set_defaults(**defaults)
    for action in parser._actions:
        if action.dest in defaults:
            action.required = False


def _csv_value(value) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return value


def _resolve_weight_paths(args) -> None:
    args.data_file_keys = _csv_value(args.data_file_keys)
    args.lora_target_modules = _csv_value(args.lora_target_modules)
    weights_dir = getattr(args, "weights_dir", None)
    if args.model_paths is not None and not isinstance(args.model_paths, str):
        args.model_paths = json.dumps(args.model_paths, separators=(",", ":"))
    if args.model_paths is None and weights_dir:
        shards = [
            str(Path(weights_dir) / "diffusion_pytorch_model-00001-of-00003.safetensors"),
            str(Path(weights_dir) / "diffusion_pytorch_model-00002-of-00003.safetensors"),
            str(Path(weights_dir) / "diffusion_pytorch_model-00003-of-00003.safetensors"),
        ]
        args.model_paths = json.dumps(
            [shards, str(Path(weights_dir) / "models_t5_umt5-xxl-enc-bf16.pth"), str(Path(weights_dir) / "Wan2.2_VAE.pth")],
            separators=(",", ":"),
        )
    if args.tokenizer_path is None and weights_dir:
        args.tokenizer_path = str(Path(weights_dir) / "google" / "umt5-xxl")
    if args.dataset_metadata_path is None and args.dataset_base_path:
        args.dataset_metadata_path = str(Path(args.dataset_base_path) / "metadata.jsonl")


def _append_timestamp_to_output_path(args) -> None:
    if not getattr(args, "append_timestamp", False):
        return
    timestamp = os.environ.get("TOKENLIGHT_RUN_TIMESTAMP") or datetime.now().strftime(args.timestamp_format)
    args.output_path = f"{str(args.output_path).rstrip('/')}_{timestamp}"


def _default_config_for_mode(train_mode: str) -> str:
    return TRAIN_CONFIG_BY_MODE.get(train_mode, DEFAULT_TRAIN_CONFIG_PATH)


def parse_tokenlight_args(train_mode: str = "single"):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        default=os.environ.get("TOKENLIGHT_TRAIN_CONFIG", _default_config_for_mode(train_mode)),
    )
    pre_args, _ = pre_parser.parse_known_args()

    raw_config = _load_json_config(pre_args.config)

    parser = wan_parser(train_mode)
    _apply_config_defaults(parser, raw_config)
    parser.set_defaults(config=pre_args.config)
    args = parser.parse_args()
    args.train_mode = train_mode
    raw_config = _load_json_config(args.config)

    _resolve_weight_paths(args)
    _append_timestamp_to_output_path(args)
    return args, raw_config, raw_config


def _parse_vae_latent_cache_specs(value: str | None) -> list[tuple[Path, Path]]:
    if value in (None, "", "None", "none", "null"):
        return []
    specs: list[tuple[Path, Path]] = []
    for item in str(value).split(";"):
        item = item.strip()
        if not item:
            continue
        if "::" in item:
            metadata_path, cache_dir = item.split("::", 1)
        elif "=" in item:
            metadata_path, cache_dir = item.split("=", 1)
        else:
            raise ValueError(
                "Each --vae_latent_cache_specs item must be metadata_path=cache_dir "
                f"or metadata_path::cache_dir, got {item!r}"
            )
        specs.append((_resolve_path(metadata_path.strip(), base_path=REPO_ROOT), _resolve_path(cache_dir.strip(), base_path=REPO_ROOT)))
    return specs


def _read_jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class VaeLatentCacheStore:
    def __init__(self, cache_dir: Path, *, shard_lru: int = 64) -> None:
        self.cache_dir = cache_dir
        self.shard_lru = max(0, int(shard_lru))
        self.index = self._load_index(cache_dir)
        self._shards: OrderedDict[str, object] = OrderedDict()

    @staticmethod
    def _load_index(cache_dir: Path) -> dict[str, dict]:
        index_path = cache_dir / "index.jsonl"
        if not index_path.exists():
            part_paths = sorted(cache_dir.glob("index_part_*.jsonl"))
            if not part_paths:
                raise FileNotFoundError(f"Missing VAE cache index: {index_path}")
            rows = []
            for part_path in part_paths:
                rows.extend(_read_jsonl_rows(part_path))
        else:
            rows = _read_jsonl_rows(index_path)
        index: dict[str, dict] = {}
        for row in rows:
            path = row.get("path")
            if isinstance(path, str) and path:
                index[path] = row
        if not index:
            raise ValueError(f"No VAE cache entries found in {cache_dir}")
        return index

    def _open_shard(self, shard_rel: str):
        from safetensors import safe_open

        shard = self._shards.get(shard_rel)
        if shard is not None:
            self._shards.move_to_end(shard_rel)
            return shard
        shard_path = self.cache_dir / shard_rel
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing VAE cache shard: {shard_path}")
        shard = safe_open(str(shard_path), framework="pt", device="cpu")
        self._shards[shard_rel] = shard
        if self.shard_lru and len(self._shards) > self.shard_lru:
            self._shards.popitem(last=False)
        return shard

    def get(self, path: str) -> torch.Tensor:
        entry = self.index.get(path)
        if entry is None:
            raise KeyError(f"Missing VAE latent cache entry for {path!r} in {self.cache_dir}")
        shard = self._open_shard(str(entry["shard"]))
        return shard.get_tensor(str(entry["tensor"]))

    def get_optional(self, path: str) -> torch.Tensor | None:
        entry = self.index.get(path)
        if entry is None:
            return None
        shard = self._open_shard(str(entry["shard"]))
        return shard.get_tensor(str(entry["tensor"]))


class TokenLightVaeLatentCacheDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        specs: list[tuple[Path, Path]],
        *,
        data_file_keys: list[str],
        height: int,
        width: int,
        repeat: int = 1,
        shard_lru: int = 64,
    ) -> None:
        if not specs:
            raise ValueError("TokenLightVaeLatentCacheDataset requires at least one cache spec")
        self.data_file_keys = [key for key in data_file_keys if key]
        self.data_file_key_set = set(self.data_file_keys)
        self.height = int(height)
        self.width = int(width)
        self.repeat = int(repeat)
        self.load_from_cache = False
        self.skip_tokenlight_file_filter = True
        self.is_vae_latent_cache_dataset = True
        self.stores: list[VaeLatentCacheStore] = []
        self.data: list[dict] = []
        for source_index, (metadata_path, cache_dir) in enumerate(specs):
            if not metadata_path.exists():
                raise FileNotFoundError(f"Missing metadata for VAE cache dataset: {metadata_path}")
            if not cache_dir.exists():
                raise FileNotFoundError(f"Missing VAE latent cache dir: {cache_dir}")
            store = VaeLatentCacheStore(cache_dir, shard_lru=shard_lru)
            self.stores.append(store)
            for row in _read_jsonl_rows(metadata_path):
                item = dict(row)
                item["_vae_latent_cache_source"] = source_index
                self.data.append(item)
        if not self.data:
            raise ValueError("No metadata rows found for VAE latent cache dataset")

    def __len__(self) -> int:
        return len(self.data) * self.repeat

    def _make_item(self, index: int) -> dict:
        row = dict(self.data[index % len(self.data)])
        store = self.stores[int(row["_vae_latent_cache_source"])]
        row["_tokenlight_cached_latents"] = True
        row["_tokenlight_height"] = self.height
        row["_tokenlight_width"] = self.width
        if "video" not in row:
            raise KeyError("Cached TokenLight metadata row is missing required 'video' key")
        row["_tokenlight_input_latents"] = store.get(str(row["video"]))
        if "input_image" in row and "input_image" in self.data_file_key_set:
            row["_tokenlight_source_latents"] = store.get(str(row["input_image"]))
        if "mask" in row and "mask" in self.data_file_keys:
            mask_latents = store.get_optional(str(row["mask"]))
            if mask_latents is not None:
                row["_tokenlight_mask_latents"] = mask_latents
        return row

    def __getitem__(self, index: int) -> dict:
        return self._make_item(index)

    def __getitems__(self, indices: list[int]) -> list[dict]:
        return [self._make_item(int(index)) for index in indices]


def _json_safe_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    return str(value)


def _accelerator_runtime_state(accelerator) -> dict:
    state = getattr(accelerator, "state", None)
    distributed_type = getattr(state, "distributed_type", None)
    ds_plugin = getattr(state, "deepspeed_plugin", None)
    ds_config = getattr(ds_plugin, "deepspeed_config", None) if ds_plugin is not None else None
    zero_config = ds_config.get("zero_optimization", {}) if isinstance(ds_config, dict) else {}
    return {
        "distributed_type": str(distributed_type) if distributed_type is not None else None,
        "num_processes": int(getattr(accelerator, "num_processes", 1)),
        "mixed_precision": getattr(accelerator, "mixed_precision", None),
        "deepspeed_config": _json_safe_value(ds_config),
        "zero_stage": zero_config.get("stage") if isinstance(zero_config, dict) else None,
        "offload_optimizer_device": (zero_config.get("offload_optimizer") or {}).get("device")
        if isinstance(zero_config, dict)
        else None,
        "offload_param_device": (zero_config.get("offload_param") or {}).get("device")
        if isinstance(zero_config, dict)
        else None,
    }


def _parameter_runtime_summary(module: nn.Module) -> dict:
    summary = {
        "parameters": 0,
        "logical_parameters": 0,
        "trainable_parameters": 0,
        "trainable_logical_parameters": 0,
        "deepspeed_parameters": 0,
        "deepspeed_logical_parameters": 0,
        "trainable_deepspeed_parameters": 0,
        "trainable_deepspeed_logical_parameters": 0,
        "deepspeed_partition_parameters": 0,
        "deepspeed_partition_bytes": 0,
        "device_numel": {},
        "deepspeed_partition_device_numel": {},
        "deepspeed_status_counts": {},
    }
    for param in module.parameters():
        numel = int(param.numel())
        logical_numel = numel
        ds_numel = getattr(param, "ds_numel", None)
        if ds_numel is not None:
            try:
                logical_numel = int(ds_numel)
            except (TypeError, ValueError):
                logical_numel = numel
        elif numel == 0 and hasattr(param, "ds_shape"):
            logical_numel = 1
            for dim in getattr(param, "ds_shape"):
                logical_numel *= int(dim)
        summary["parameters"] += numel
        summary["logical_parameters"] += logical_numel
        if param.requires_grad:
            summary["trainable_parameters"] += numel
            summary["trainable_logical_parameters"] += logical_numel
        if hasattr(param, "ds_id"):
            summary["deepspeed_parameters"] += numel
            summary["deepspeed_logical_parameters"] += logical_numel
            if param.requires_grad:
                summary["trainable_deepspeed_parameters"] += numel
                summary["trainable_deepspeed_logical_parameters"] += logical_numel
            ds_status = str(getattr(param, "ds_status", "unknown"))
            summary["deepspeed_status_counts"][ds_status] = summary["deepspeed_status_counts"].get(ds_status, 0) + 1
            ds_tensor = getattr(param, "ds_tensor", None)
            if isinstance(ds_tensor, torch.Tensor):
                ds_tensor_numel = int(ds_tensor.numel())
                summary["deepspeed_partition_parameters"] += ds_tensor_numel
                summary["deepspeed_partition_bytes"] += ds_tensor_numel * int(ds_tensor.element_size())
                ds_device_key = str(ds_tensor.device)
                summary["deepspeed_partition_device_numel"][ds_device_key] = (
                    summary["deepspeed_partition_device_numel"].get(ds_device_key, 0) + ds_tensor_numel
                )
        device_key = str(param.device)
        summary["device_numel"][device_key] = summary["device_numel"].get(device_key, 0) + numel
    return summary


def _cuda_runtime_memory(accelerator) -> dict:
    if not torch.cuda.is_available():
        return {}
    device = accelerator.device
    try:
        index = int(device.index if device.index is not None else torch.cuda.current_device())
    except Exception:
        index = torch.cuda.current_device()
    return {
        "device": str(device),
        "allocated_bytes": int(torch.cuda.memory_allocated(index)),
        "reserved_bytes": int(torch.cuda.memory_reserved(index)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
    }


def _component_parameter_summaries(model) -> dict:
    pipe = getattr(model, "pipe", None)
    components = {
        "training_module": model,
        "pipe.dit": getattr(pipe, "dit", None),
        "pipe.vae": getattr(pipe, "vae", None),
        "pipe.text_encoder": getattr(pipe, "text_encoder", None),
        "light_encoder": getattr(model, "light_encoder", None),
        "tokenlight_type_embedding": getattr(model, "tokenlight_type_embedding", None),
        "joint_mask_predictor": getattr(model, "joint_mask_predictor", None),
    }
    return {
        name: _parameter_runtime_summary(module)
        for name, module in components.items()
        if isinstance(module, nn.Module)
    }


def save_training_runtime_snapshot(args, accelerator, model) -> None:
    if not accelerator.is_main_process:
        return
    unwrapped_model = accelerator.unwrap_model(model)
    runtime = {
        "accelerator": _accelerator_runtime_state(accelerator),
        "tokenlight": {
            "use_gradient_checkpointing": bool(getattr(unwrapped_model, "use_gradient_checkpointing", False)),
            "use_gradient_checkpointing_offload": bool(
                getattr(unwrapped_model, "use_gradient_checkpointing_offload", False)
            ),
        },
        "parameters": _parameter_runtime_summary(unwrapped_model),
        "component_parameters": _component_parameter_summaries(unwrapped_model),
        "cuda_memory": _cuda_runtime_memory(accelerator),
    }
    output_path = Path(args.output_path)
    with (output_path / "train_runtime_resolved.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe_value(runtime), f, indent=2, sort_keys=True)
    ds = runtime["accelerator"]
    print(
        "Runtime setup: "
        f"distributed_type={ds['distributed_type']}, "
        f"zero_stage={ds['zero_stage']}, "
        f"offload_param={ds['offload_param_device']}, "
        f"offload_optimizer={ds['offload_optimizer_device']}, "
        f"use_gradient_checkpointing={runtime['tokenlight']['use_gradient_checkpointing']}"
    )


def save_training_config_snapshot(args, raw_config: dict, merged_config: dict, accelerator) -> None:
    if not accelerator.is_main_process:
        return
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config) if args.config else None
    if config_path is not None and config_path.exists():
        shutil.copy2(config_path, output_path / "train_config.json")
    else:
        with (output_path / "train_config.json").open("w", encoding="utf-8") as f:
            json.dump(raw_config, f, indent=2, sort_keys=True)

    resolved = {key: _json_safe_value(value) for key, value in vars(args).items() if not key.startswith("_")}
    resolved["num_processes"] = int(getattr(accelerator, "num_processes", 1))
    resolved["per_gpu_batch_size"] = int(args.batch_size)
    resolved["effective_global_batch_size"] = (
        int(args.batch_size) * int(args.gradient_accumulation_steps) * int(getattr(accelerator, "num_processes", 1))
    )
    with (output_path / "train_config_resolved.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "raw_config": _json_safe_value(raw_config),
                "merged_config": _json_safe_value(merged_config),
                "resolved_args": resolved,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"Training config copied to {output_path / 'train_config.json'}")
    print(
        "Batch setup: "
        f"num_processes={resolved['num_processes']}, "
        f"per_gpu_batch_size={resolved['per_gpu_batch_size']}, "
        f"gradient_accumulation_steps={args.gradient_accumulation_steps}, "
        f"effective_global_batch_size={resolved['effective_global_batch_size']}"
    )


def _finite_float(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean_across_processes(accelerator, value):
    value = _finite_float(value)
    if value is None:
        return None
    tensor = torch.tensor([value], device=accelerator.device, dtype=torch.float32)
    if getattr(accelerator, "num_processes", 1) > 1:
        try:
            tensor = accelerator.gather(tensor).mean()
        except Exception:
            tensor = tensor.mean()
    return _finite_float(tensor.item())


def _loss_to_float(accelerator, loss):
    if loss is None:
        return None
    value = loss.detach().float().mean()
    if getattr(accelerator, "num_processes", 1) > 1:
        try:
            value = accelerator.gather(value.reshape(1)).mean()
        except Exception:
            value = value.mean()
    return _finite_float(value.item())


def _module_gradient_stats(module):
    if module is None:
        return {}
    total_sq = 0.0
    param_total_sq = 0.0
    trainable_params = 0
    params_with_grad = 0
    for param in module.parameters():
        if not param.requires_grad:
            continue
        trainable_params += param.numel()
        try:
            param_norm = param.detach().float().norm(2).item()
        except Exception:
            param_norm = None
        if param_norm is not None and math.isfinite(param_norm):
            param_total_sq += param_norm * param_norm
        grad = param.grad
        if grad is None:
            continue
        try:
            grad = grad.detach()
            if grad.is_sparse:
                grad = grad.coalesce().values()
            norm = grad.float().norm(2).item()
        except Exception:
            continue
        if not math.isfinite(norm):
            continue
        params_with_grad += param.numel()
        total_sq += norm * norm
    if trainable_params == 0:
        return {}
    stats = {
        "grad_param_fraction": params_with_grad / trainable_params,
        "param_norm": math.sqrt(param_total_sq),
    }
    if params_with_grad > 0:
        stats["grad_norm"] = math.sqrt(total_sq)
    return stats


def _collect_train_metrics(accelerator, model, optimizer, loss):
    unwrapped_model = accelerator.unwrap_model(model)
    metrics = {
        "train/loss": _loss_to_float(accelerator, loss),
    }
    if optimizer.param_groups:
        metrics["train/learning_rate"] = _finite_float(optimizer.param_groups[0].get("lr"))

    for prefix, module in (
        ("train/light_encoder", getattr(unwrapped_model, "light_encoder", None)),
        ("train/type_embedding", getattr(unwrapped_model, "tokenlight_type_embedding", None)),
        ("train/joint_mask_predictor", getattr(unwrapped_model, "joint_mask_predictor", None)),
    ):
        for key, value in _module_gradient_stats(module).items():
            metrics[f"{prefix}_{key}"] = _mean_across_processes(accelerator, value)
    pipe = getattr(unwrapped_model, "pipe", None)
    extra_loss_metrics = getattr(pipe, "_tokenlight_loss_metrics", None)
    if isinstance(extra_loss_metrics, dict):
        for key, value in extra_loss_metrics.items():
            if isinstance(value, torch.Tensor):
                metrics[str(key)] = _loss_to_float(accelerator, value)
            else:
                metrics[str(key)] = _mean_across_processes(accelerator, value)
    return {key: value for key, value in metrics.items() if value is not None}


def _trainable_parameters(model):
    params = []
    seen = set()
    trainable_modules = getattr(model, "trainable_modules", None)
    if callable(trainable_modules):
        for parameter in trainable_modules():
            if parameter.requires_grad and id(parameter) not in seen:
                params.append(parameter)
                seen.add(id(parameter))
    for parameter in model.parameters():
        if parameter.requires_grad and id(parameter) not in seen:
            params.append(parameter)
            seen.add(id(parameter))
    return params


def _preferred_trainable_dtype(model) -> torch.dtype:
    pipe = getattr(model, "pipe", None)
    dtype = getattr(pipe, "torch_dtype", None)
    if isinstance(dtype, torch.dtype) and dtype.is_floating_point:
        return dtype
    dit = getattr(pipe, "dit", None)
    if isinstance(dit, nn.Module):
        for parameter in dit.parameters():
            if parameter.is_floating_point():
                return parameter.dtype
    return torch.bfloat16


def _coerce_trainable_parameter_dtype(model, dtype: torch.dtype) -> dict[str, int]:
    counts: dict[str, int] = {}
    for parameter in _trainable_parameters(model):
        if not parameter.is_floating_point():
            continue
        counts[str(parameter.dtype)] = counts.get(str(parameter.dtype), 0) + int(parameter.numel())
        if parameter.dtype != dtype:
            parameter.data = parameter.data.to(dtype=dtype)
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.data.to(dtype=dtype)
    return counts


def _trainable_dtype_counts(model) -> dict[str, int]:
    counts: dict[str, int] = {}
    for parameter in _trainable_parameters(model):
        if parameter.is_floating_point():
            counts[str(parameter.dtype)] = counts.get(str(parameter.dtype), 0) + int(parameter.numel())
    return counts


def _call_compatible_method(obj, method_name: str, *args, **kwargs):
    method = getattr(obj, method_name, None)
    if method is None:
        return None
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method(*args, **kwargs)

    parameters = list(signature.parameters.values())
    accepts_args = any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in parameters)
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters)
    if accepts_args:
        compatible_args = args
    else:
        positional_slots = [
            param
            for param in parameters
            if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        compatible_args = args[: len(positional_slots)]
    if accepts_kwargs:
        compatible_kwargs = kwargs
    else:
        accepted = {
            param.name
            for param in parameters
            if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        compatible_kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    return method(*compatible_args, **compatible_kwargs)


class TokenLightTensorBoardMetrics:
    def __init__(self, output_path: str, enabled: bool = False):
        self.output_path = output_path
        self.enabled = bool(enabled)
        self.writer = None
        self.initialized = False

    def _init_writer(self):
        if self.initialized:
            return
        self.initialized = True
        if not self.enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:
            warnings.warn(f"TensorBoard metrics disabled because SummaryWriter could not be imported: {exc}")
            return
        log_dir = os.path.join(self.output_path, "tensorboard_log")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TokenLight TensorBoard metrics enabled. Run `tensorboard --logdir={log_dir}`.")

    def log(self, accelerator, step: int, metrics: dict[str, float]):
        if not accelerator.is_main_process:
            return
        self._init_writer()
        if self.writer is None:
            return
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        self.writer.flush()

    def close(self, accelerator):
        if not accelerator.is_main_process:
            return
        if self.writer is not None:
            self.writer.close()
            self.writer = None


class TokenLightWanTrainingModule(DiffusionTrainingModule):  # noqa: F405
    """Wan2.2 TI2V trainer with TokenLight numeric light tokens."""

    def resume_from_checkpoint(self, resume_from_checkpoint=None, remove_prefix_in_ckpt=None):
        if resume_from_checkpoint in (None, "", "None", "null"):
            return
        parent_resume = getattr(super(TokenLightWanTrainingModule, self), "resume_from_checkpoint", None)
        if parent_resume is None:
            raise AttributeError(
                "This installed DiffSynth version does not provide resume_from_checkpoint(). "
                "Set runtime.resume_from_checkpoint to null, or use lora_checkpoint to load trainable weights."
            )
        return parent_resume(resume_from_checkpoint, remove_prefix_in_ckpt)

    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        resume_from_checkpoint=None,
        remove_prefix_in_ckpt=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        tokenlight_light_tokens=True,
        tokenlight_attrs_key="attrs_json",
        tokenlight_token_dim=0,
        tokenlight_fourier_features=512,
        tokenlight_fourier_sigma=5.0,
        tokenlight_max_lights=1,
        tokenlight_light_dropout=0.0,
        tokenlight_cfg_drop_prob=0.0,
        tokenlight_source_tokens=True,
        tokenlight_mask_tokens=True,
        prompt_context_cache_size=4,
    ):
        super().__init__()
        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            device=device,
        )
        tokenizer_config = (
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")
            if tokenizer_path is None
            else ModelConfig(tokenizer_path)
        )
        audio_processor_config = self.parse_path_or_model_id(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            audio_processor_config=audio_processor_config,
        )
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        self.resume_from_checkpoint(resume_from_checkpoint, remove_prefix_in_ckpt)
        lora_target_modules = _normalize_wan_lora_target_modules(lora_target_modules)
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint,
            preset_lora_path,
            preset_lora_model,
            task=task,
        )

        self.tokenlight_attrs_key = tokenlight_attrs_key
        self.tokenlight_light_tokens = bool(tokenlight_light_tokens)
        self.tokenlight_source_tokens = bool(tokenlight_source_tokens)
        self.tokenlight_mask_tokens = bool(tokenlight_mask_tokens)
        self.tokenlight_cfg_drop_prob = float(tokenlight_cfg_drop_prob)
        token_dim = int(tokenlight_token_dim) if int(tokenlight_token_dim) > 0 else int(self.pipe.dit.dim)
        self.light_encoder = (
            LightokenEncoder(
                token_dim=token_dim,
                fourier_features=int(tokenlight_fourier_features),
                fourier_sigma=float(tokenlight_fourier_sigma),
                max_lights=int(tokenlight_max_lights),
                dropout=float(tokenlight_light_dropout),
            )
            if self.tokenlight_light_tokens
            else None
        )
        self.tokenlight_type_embedding = (
            TokenLightTypeEmbedding(token_dim)
            if self.tokenlight_light_tokens or self.tokenlight_source_tokens or self.tokenlight_mask_tokens
            else None
        )
        aux_state = _load_checkpoint_state_dict(lora_checkpoint or resume_from_checkpoint)
        _load_module_state_from_checkpoint(self.light_encoder, aux_state, "light_encoder")
        _load_module_state_from_checkpoint(self.tokenlight_type_embedding, aux_state, "tokenlight_type_embedding")
        if self.tokenlight_light_tokens or self.tokenlight_source_tokens or self.tokenlight_mask_tokens:
            self.pipe.model_fn = self._tokenlight_model_fn

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = [item for item in extra_inputs.split(",") if item] if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.prompt_context_cache_size = max(0, int(prompt_context_cache_size or 0))
        self._prompt_context_cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()
        self._text_encoder_trainable_cache: bool | None = None
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(  # noqa: F405
                pipe, **inputs_shared, **inputs_posi
            ),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(  # noqa: F405
                pipe, **inputs_shared, **inputs_posi
            ),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(  # noqa: F405
                pipe, **inputs_shared, **inputs_posi
            ),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(  # noqa: F405
                pipe, **inputs_shared, **inputs_posi
            ),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                image = _as_frames(data.get("input_image", data["video"]))[0]
                inputs_shared["input_image"] = image
                inputs_shared["tokenlight_source_image"] = image
            elif extra_input == "end_image":
                inputs_shared["end_image"] = _as_frames(data.get("end_image", data["video"]))[-1]
            elif extra_input in {"reference_image", "vace_reference_image"}:
                inputs_shared[extra_input] = _as_frames(data[extra_input])[0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        if inputs_shared.get("framewise_decoding", False):
            inputs_shared["num_frames"] = 4 * (len(_as_frames(data["video"])) - 1) + 1
        return inputs_shared

    def get_pipeline_inputs(self, data):
        video = _as_frames(data["video"])
        first_frame = video[0]
        tokenlight_source_image = _as_frames(data.get("input_image", video))[0]
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            "input_video": video,
            "height": first_frame.size[1],
            "width": first_frame.size[0],
            "num_frames": len(video),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "tokenlight_source_image": tokenlight_source_image,
        }
        if "mask" in data:
            inputs_shared["tokenlight_mask_image"] = _as_frames(data["mask"])[0]
        return self.parse_extra_inputs(data, self.extra_inputs, inputs_shared), inputs_posi, inputs_nega

    @staticmethod
    def _is_collated_batch(data):
        if not isinstance(data, dict):
            return False
        prompt = data.get("prompt")
        if isinstance(prompt, list):
            return True
        video = data.get("video")
        return isinstance(video, list) and bool(video) and isinstance(video[0], list)

    @staticmethod
    def _batch_size_from_data(data):
        for key in ("prompt", "attrs_json", "attrs", "task"):
            value = data.get(key)
            if isinstance(value, list):
                return len(value)
        video = data.get("video")
        if isinstance(video, list) and video and isinstance(video[0], list):
            return len(video)
        return 1

    @staticmethod
    def _has_cached_latents(data):
        return isinstance(data, dict) and "_tokenlight_input_latents" in data

    @staticmethod
    def _first_scalar(value, default):
        if isinstance(value, list) and value:
            return value[0]
        return default if value in (None, "") else value

    def _stack_cached_latents(self, value, *, batch: int, name: str) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.unsqueeze(0) if value.ndim == 4 else value
        elif isinstance(value, list):
            tensors = []
            for item in value:
                if not isinstance(item, torch.Tensor):
                    raise TypeError(f"Expected tensor for cached {name}, got {type(item).__name__}")
                tensors.append(item)
            tensor = torch.stack(tensors, dim=0)
        else:
            raise TypeError(f"Missing cached {name} latents")
        if tensor.ndim != 5:
            raise ValueError(f"Expected cached {name} latents with shape [B,C,T,H,W], got {tuple(tensor.shape)}")
        if int(tensor.shape[0]) != int(batch):
            raise ValueError(f"Cached {name} batch size mismatch: {tensor.shape[0]} != {batch}")
        try:
            non_blocking = bool(tensor.is_pinned())
        except RuntimeError:
            non_blocking = False
        return tensor.to(dtype=self.pipe.torch_dtype, device=self.pipe.device, non_blocking=non_blocking)

    def _cached_batched_inputs(self, data):
        batch = self._batch_size_from_data(data)
        prompts = data.get("prompt")
        if isinstance(prompts, str):
            prompts = [prompts]
        if not isinstance(prompts, list) or len(prompts) != batch:
            raise ValueError(f"Expected {batch} prompts for cached VAE training")

        input_latents = self._stack_cached_latents(
            data.get("_tokenlight_input_latents"),
            batch=batch,
            name="input",
        )
        height = int(self._first_scalar(data.get("_tokenlight_height"), input_latents.shape[-2] * 16))
        width = int(self._first_scalar(data.get("_tokenlight_width"), input_latents.shape[-1] * 16))
        inputs_posi = {"prompt": prompts, "context": self._encode_prompts(prompts)}
        inputs_nega = {}
        inputs_shared = {
            "input_video": None,
            "input_latents": input_latents,
            "height": height,
            "width": width,
            "num_frames": int(input_latents.shape[2]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }

        if self.tokenlight_source_tokens:
            source_latents = data.get("_tokenlight_source_latents")
            if source_latents is None:
                source_latents = data.get("_tokenlight_input_latents")
            inputs_shared["tokenlight_source_latents"] = self._stack_cached_latents(
                source_latents,
                batch=batch,
                name="source",
            )

        if self.tokenlight_mask_tokens:
            mask_latents = data.get("_tokenlight_mask_latents")
            if mask_latents is None:
                inputs_shared["tokenlight_mask_latents"] = torch.zeros_like(input_latents)
            else:
                inputs_shared["tokenlight_mask_latents"] = self._stack_cached_latents(
                    mask_latents,
                    batch=batch,
                    name="mask",
                )

        attrs = attrs_from_batch(data, key=self.tokenlight_attrs_key)
        if self.light_encoder is not None:
            attrs = _require_nonempty_light_attrs(attrs, key=self.tokenlight_attrs_key)
            drop_light = False
            if self.training and self.tokenlight_cfg_drop_prob > 0:
                drop_light = torch.rand(batch, device=inputs_posi["context"].device) < self.tokenlight_cfg_drop_prob
            inputs_posi["tokenlight_drop_light"] = drop_light
            inputs_nega["tokenlight_drop_light"] = True
        inputs_shared["tokenlight_attrs"] = attrs
        return inputs_shared, inputs_posi, inputs_nega

    def _text_encoder_has_trainable_parameters(self) -> bool:
        if self._text_encoder_trainable_cache is not None:
            return self._text_encoder_trainable_cache
        text_encoder = getattr(self.pipe, "text_encoder", None)
        if not isinstance(text_encoder, nn.Module):
            self._text_encoder_trainable_cache = False
        else:
            self._text_encoder_trainable_cache = any(param.requires_grad for param in text_encoder.parameters())
        return self._text_encoder_trainable_cache

    def _encode_prompts(self, prompts):
        prompt_key = tuple(str(prompt) for prompt in prompts)
        can_cache = self.prompt_context_cache_size > 0 and not self._text_encoder_has_trainable_parameters()
        cache_key = None
        if can_cache:
            cache_key = (prompt_key, str(self.pipe.device), str(self.pipe.torch_dtype))
            cached = self._prompt_context_cache.get(cache_key)
            if cached is not None:
                self._prompt_context_cache.move_to_end(cache_key)
                return cached

        self.pipe.load_models_to_device(["text_encoder"])
        ids, mask = self.pipe.tokenizer(prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.pipe.device)
        mask = mask.to(self.pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = torch.no_grad() if can_cache else nullcontext()
        with context:
            prompt_emb = self.pipe.text_encoder(ids, mask)
        for index, seq_len in enumerate(seq_lens):
            prompt_emb[index, seq_len:] = 0
        if cache_key is not None:
            prompt_emb = prompt_emb.detach()
            self._prompt_context_cache[cache_key] = prompt_emb
            self._prompt_context_cache.move_to_end(cache_key)
            while len(self._prompt_context_cache) > self.prompt_context_cache_size:
                self._prompt_context_cache.popitem(last=False)
        return prompt_emb

    def _preprocess_videos_batched(self, videos):
        tensors = []
        for video in videos:
            frames = _as_frames(video)
            tensors.append(self.pipe.preprocess_video(frames, device=self.pipe.device))
        return torch.cat(tensors, dim=0).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

    def _encode_video_latents_batched(self, videos, inputs_shared):
        tiled = bool(inputs_shared.get("tiled", False))
        tile_size = inputs_shared.get("tile_size", (30, 52))
        tile_stride = inputs_shared.get("tile_stride", (15, 26))
        self.pipe.load_models_to_device(["vae"])
        pixel_values = self._preprocess_videos_batched(videos)
        latents = self.pipe.vae.encode(
            pixel_values,
            device=self.pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return latents.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

    def _batched_inputs(self, data):
        batch = self._batch_size_from_data(data)
        videos = data["video"]
        if not isinstance(videos, list) or len(videos) != batch:
            raise ValueError(f"Expected {batch} batched videos, got {type(videos).__name__}")
        first_frame = _as_frames(videos[0])[0]
        prompts = data.get("prompt")
        if not isinstance(prompts, list) or len(prompts) != batch:
            raise ValueError(f"Expected {batch} prompts for batched training")
        inputs_posi = {"prompt": prompts, "context": self._encode_prompts(prompts)}
        inputs_nega = {}
        inputs_shared = {
            "input_video": videos,
            "input_latents": None,
            "height": first_frame.size[1],
            "width": first_frame.size[0],
            "num_frames": len(_as_frames(videos[0])),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared["input_latents"] = self._encode_video_latents_batched(videos, inputs_shared)

        if self.tokenlight_source_tokens:
            source_videos = data.get("input_image", videos)
            if isinstance(source_videos, list) and len(source_videos) == batch:
                source_videos = [_as_frames(item) for item in source_videos]
            else:
                source_videos = [[_as_frames(video)[0]] for video in videos]
            inputs_shared["tokenlight_source_latents"] = self._encode_video_latents_batched(source_videos, inputs_shared)

        if self.tokenlight_mask_tokens:
            masks = data.get("mask")
            if isinstance(masks, list) and len(masks) == batch:
                mask_videos = [_as_frames(item) for item in masks]
                inputs_shared["tokenlight_mask_latents"] = self._encode_video_latents_batched(mask_videos, inputs_shared)
            else:
                inputs_shared["tokenlight_mask_latents"] = torch.zeros_like(inputs_shared["input_latents"])

        attrs = attrs_from_batch(data, key=self.tokenlight_attrs_key)
        if self.light_encoder is not None:
            attrs = _require_nonempty_light_attrs(attrs, key=self.tokenlight_attrs_key)
            drop_light = False
            if self.training and self.tokenlight_cfg_drop_prob > 0:
                drop_light = torch.rand(batch, device=inputs_posi["context"].device) < self.tokenlight_cfg_drop_prob
            inputs_posi["tokenlight_drop_light"] = drop_light
            inputs_nega["tokenlight_drop_light"] = True
        inputs_shared["tokenlight_attrs"] = attrs
        return inputs_shared, inputs_posi, inputs_nega

    def _tokenlight_model_fn(self, **kwargs):
        return model_fn_wan_video_tokenlight(
            tokenlight_light_encoder=self.light_encoder,
            tokenlight_type_embedding=self.tokenlight_type_embedding,
            **kwargs,
        )

    def _encode_image_latents(self, image, inputs_shared):
        width = int(inputs_shared["width"])
        height = int(inputs_shared["height"])
        tiled = bool(inputs_shared.get("tiled", False))
        tile_size = inputs_shared.get("tile_size", (30, 52))
        tile_stride = inputs_shared.get("tile_stride", (15, 26))
        if hasattr(image, "resize"):
            image = image.resize((width, height))
        self.pipe.load_models_to_device(["vae"])
        pixel_values = self.pipe.preprocess_image(image).transpose(0, 1)
        latents = self.pipe.vae.encode(
            [pixel_values.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)],
            device=self.pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return latents.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

    def _prepare_tokenlight_inputs(self, inputs, data):
        inputs_shared, inputs_posi, inputs_nega = inputs
        if not (self.tokenlight_light_tokens or self.tokenlight_source_tokens or self.tokenlight_mask_tokens):
            return inputs

        attrs = attrs_from_batch(data, key=self.tokenlight_attrs_key)
        if self.light_encoder is not None:
            attrs = _require_nonempty_light_attrs(attrs, key=self.tokenlight_attrs_key)
        inputs_shared["tokenlight_attrs"] = attrs
        if self.tokenlight_source_tokens:
            if "first_frame_latents" in inputs_shared:
                inputs_shared["tokenlight_source_latents"] = inputs_shared["first_frame_latents"]
            elif "tokenlight_source_image" in inputs_shared:
                inputs_shared["tokenlight_source_latents"] = self._encode_image_latents(
                    inputs_shared["tokenlight_source_image"],
                    inputs_shared,
                )
        if self.tokenlight_mask_tokens:
            if "tokenlight_mask_image" in inputs_shared:
                inputs_shared["tokenlight_mask_latents"] = self._encode_image_latents(
                    inputs_shared["tokenlight_mask_image"],
                    inputs_shared,
                )
            elif "input_latents" in inputs_shared:
                inputs_shared["tokenlight_mask_latents"] = torch.zeros_like(inputs_shared["input_latents"])
        if self.light_encoder is not None:
            drop_light = False
            if self.training and self.tokenlight_cfg_drop_prob > 0 and "context" in inputs_posi:
                batch = inputs_posi["context"].shape[0]
                drop_light = torch.rand(batch, device=inputs_posi["context"].device) < self.tokenlight_cfg_drop_prob
            inputs_posi["tokenlight_drop_light"] = drop_light
            inputs_nega["tokenlight_drop_light"] = True
        inputs_shared.pop("tokenlight_source_image", None)
        inputs_shared.pop("tokenlight_mask_image", None)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None and self._has_cached_latents(data):
            inputs = self._cached_batched_inputs(data)
            return self.task_to_loss[self.task](self.pipe, *inputs)
        if inputs is None and self._is_collated_batch(data):
            inputs = self._batched_inputs(data)
            return self.task_to_loss[self.task](self.pipe, *inputs)
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        inputs = self._prepare_tokenlight_inputs(inputs, data)
        return self.task_to_loss[self.task](self.pipe, *inputs)

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        exported = super().export_trainable_state_dict(state_dict, remove_prefix=remove_prefix)
        if self.light_encoder is not None:
            for key, value in state_dict.items():
                if key.startswith("light_encoder."):
                    exported[key] = value
        if self.tokenlight_type_embedding is not None:
            for key, value in state_dict.items():
                if key.startswith("tokenlight_type_embedding."):
                    exported[key] = value
        return exported


def wan_parser(train_mode: str = "single") -> argparse.ArgumentParser:
    if train_mode not in {"single", "zero3"}:
        raise ValueError(f"Unsupported TokenLight train mode: {train_mode}")

    parser = argparse.ArgumentParser(description=f"TokenLight Wan2.2-TI2V-5B {train_mode} trainer.")
    parser.add_argument("--config", type=str, default=None, help="Path to TokenLight train config JSON.")

    parser.add_argument("--weights_dir", type=str, default="weights/Wan2.2-TI2V-5B")
    parser.add_argument("--model_paths", type=str, default=None)
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--audio_processor_path", type=str, default=None)

    parser.add_argument("--output_path", type=str, default="model/train/tokenlight_wan22_lora")
    parser.add_argument("--append_timestamp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--timestamp_format", type=str, default="%Y%m%d_%H%M%S")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.")

    parser.add_argument("--dataset_base_path", type=str, default=None)
    parser.add_argument("--dataset_metadata_path", type=str, default=None)
    parser.add_argument("--data_file_keys", type=str, default="video,input_image")
    parser.add_argument(
        "--vae_latent_cache_specs",
        type=str,
        default=None,
        help=(
            "Semicolon-separated metadata/cache pairs for precomputed VAE latents, "
            "for example metadata.jsonl=vae_latent_cache;other.jsonl=other_cache."
        ),
    )
    parser.add_argument(
        "--vae_latent_cache_shard_lru",
        type=int,
        default=64,
        help="Number of safetensors shard handles to keep open per dataset worker.",
    )
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--dataset_num_workers", type=int, default=0)
    parser.add_argument("--dataset_auto_workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataset_pin_memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dataset_persistent_workers", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dataset_prefetch_factor", type=int, default=2)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--max_pixels", type=int, default=1280 * 704)
    parser.add_argument("--batch_size", type=int, default=1, help="Per-GPU DataLoader batch size.")
    parser.add_argument("--balanced_task_batch", default=None)
    parser.add_argument("--balanced_batch_seed", type=int, default=0)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--customized_optimizer", type=str, default=None)

    parser.add_argument("--trainable_models", type=str, default=None)
    parser.add_argument("--lora_base_model", type=str, default="dit")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_checkpoint", type=str, default=None)
    parser.add_argument("--preset_lora_path", type=str, default=None)
    parser.add_argument("--preset_lora_model", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument(
        "--task",
        type=str,
        choices=(
            "sft",
            "sft:train",
            "sft:data_process",
            "direct_distill",
            "direct_distill:train",
            "direct_distill:data_process",
        ),
        default="sft",
    )
    parser.add_argument("--use_gradient_checkpointing", action=argparse.BooleanOptionalAction, default=train_mode == "zero3")
    parser.add_argument("--find_unused_parameters", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--initialize_model_on_cpu", action=argparse.BooleanOptionalAction, default=train_mode == "zero3")
    parser.add_argument("--framewise_decoding", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0)
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0)
    parser.add_argument("--extra_inputs", type=str, default=None)
    parser.add_argument("--fp8_models", type=str, default=None)
    parser.add_argument("--offload_models", type=str, default=None)
    parser.add_argument("--prompt_context_cache_size", type=int, default=4)

    parser.add_argument("--tokenlight_light_tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tokenlight_source_tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tokenlight_mask_tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tokenlight_attrs_key", default="attrs_json")
    parser.add_argument("--tokenlight_token_dim", type=int, default=0)
    parser.add_argument("--tokenlight_fourier_features", type=int, default=512)
    parser.add_argument("--tokenlight_fourier_sigma", type=float, default=5.0)
    parser.add_argument("--tokenlight_max_lights", type=int, default=1)
    parser.add_argument("--tokenlight_light_dropout", type=float, default=0.0)
    parser.add_argument("--tokenlight_cfg_drop_prob", type=float, default=0.1)

    parser.add_argument("--enable_tensorboard_log", action=argparse.BooleanOptionalAction, default=False)

    if train_mode == "single":
        parser.add_argument("--enable_model_cpu_offload", action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument("--enable_optimizer_cpu_offload", action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument("--cpu_offload_split_threshold", type=float, default=None)
    return parser


def build_dataset(args):
    data_file_keys = [key for key in args.data_file_keys.split(",") if key]
    cache_specs = _parse_vae_latent_cache_specs(getattr(args, "vae_latent_cache_specs", None))
    if cache_specs:
        dataset = TokenLightVaeLatentCacheDataset(
            cache_specs,
            data_file_keys=data_file_keys,
            height=args.height,
            width=args.width,
            repeat=args.dataset_repeat,
            shard_lru=int(getattr(args, "vae_latent_cache_shard_lru", 64)),
        )
    else:
        dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=data_file_keys,
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=args.dataset_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=16,
                width_division_factor=16,
                num_frames=args.num_frames,
                time_division_factor=4 if not args.framewise_decoding else 1,
                time_division_remainder=1 if not args.framewise_decoding else 0,
            ),
            special_operator_map={
                "animate_face_video": ToAbsolutePath(args.dataset_base_path)
                >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
                "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
                "wantodance_music_path": ToAbsolutePath(args.dataset_base_path),
            },
        )
    if dataset.data:
        original_count = len(dataset.data)
        normalized = _normalize_tokenlight_metadata_rows(dataset.data)
        if getattr(dataset, "skip_tokenlight_file_filter", False):
            filtered = [row for row in normalized if row.get("valid") is not False]
            filter_stats = {"row_invalid": len(normalized) - len(filtered)}
        else:
            filtered, filter_stats = _filter_tokenlight_metadata_rows(normalized, args.dataset_base_path)
        removed = original_count - len(filtered)
        if len(normalized) != original_count or removed:
            nonzero_stats = {key: value for key, value in filter_stats.items() if value}
            print(
                "TokenLight metadata normalized/filtered: "
                f"kept {len(filtered)} of {original_count} rows; "
                f"removed={removed}; reasons={nonzero_stats}"
            )
        dataset.data = filtered
    return dataset


_CACHED_LATENT_BATCH_KEYS = {
    "_tokenlight_input_latents",
    "_tokenlight_source_latents",
    "_tokenlight_mask_latents",
}


def _stack_tensor_values(values: list):
    if not values or not all(isinstance(value, torch.Tensor) for value in values):
        return None
    try:
        return torch.stack(values, dim=0)
    except RuntimeError:
        return None


def _collate_tokenlight_batch(batch):
    if len(batch) == 1:
        return batch[0]
    keys = sorted({key for item in batch for key in item.keys()})
    result = {}
    for key in keys:
        values = [item.get(key) for item in batch]
        if key in _CACHED_LATENT_BATCH_KEYS:
            stacked = _stack_tensor_values(values)
            if stacked is not None:
                result[key] = stacked
            elif key in {"_tokenlight_source_latents", "_tokenlight_mask_latents"}:
                continue
            else:
                result[key] = values
        else:
            result[key] = values
    return result


def _optional_bool(value, *, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return bool(default)
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(value)


def _dataloader_runtime_kwargs(args, dataset, *, num_workers: int, accelerator=None) -> dict:
    is_cached = bool(getattr(dataset, "is_vae_latent_cache_dataset", False))
    cuda_available = bool(torch.cuda.is_available())
    requested_workers = int(num_workers)
    effective_workers = requested_workers
    if (
        is_cached
        and requested_workers <= 0
        and _optional_bool(getattr(args, "dataset_auto_workers", True), default=True)
    ):
        num_processes = max(1, int(getattr(accelerator, "num_processes", 1) or 1))
        cpu_count = max(1, int(os.cpu_count() or 1))
        effective_workers = max(1, min(4, cpu_count // num_processes))
    pin_memory = _optional_bool(
        getattr(args, "dataset_pin_memory", None),
        default=is_cached and cuda_available,
    )
    kwargs = {
        "num_workers": effective_workers,
        "pin_memory": pin_memory,
    }
    if effective_workers > 0:
        kwargs["persistent_workers"] = _optional_bool(
            getattr(args, "dataset_persistent_workers", None),
            default=is_cached,
        )
        prefetch_factor = int(getattr(args, "dataset_prefetch_factor", 2) or 0)
        if prefetch_factor > 0:
            kwargs["prefetch_factor"] = prefetch_factor
    if accelerator is None or getattr(accelerator, "is_main_process", True):
        print(
            "DataLoader setup: "
            f"cached={is_cached}, "
            f"num_workers={kwargs['num_workers']}"
            f"{f' (auto from {requested_workers})' if kwargs['num_workers'] != requested_workers else ''}, "
            f"pin_memory={kwargs['pin_memory']}, "
            f"persistent_workers={kwargs.get('persistent_workers', False)}, "
            f"prefetch_factor={kwargs.get('prefetch_factor', None)}"
        )
    return kwargs


def _parse_balanced_task_batch(value) -> dict[str, int] | None:
    if value in (None, "", "none", "None", "null"):
        return None
    if isinstance(value, dict):
        result = {str(key): int(item) for key, item in value.items() if int(item) > 0}
        return result or None
    result = {}
    for item in str(value).split(","):
        if not item.strip():
            continue
        key, count = item.split(":", 1)
        result[key.strip()] = int(count)
    return {key: count for key, count in result.items() if count > 0} or None


def _configure_deepspeed_batch_size(accelerator, args, batch_size: int) -> None:
    plugin = getattr(getattr(accelerator, "state", None), "deepspeed_plugin", None)
    ds_config = getattr(plugin, "deepspeed_config", None)
    if not isinstance(ds_config, dict):
        return
    grad_accum = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
    num_processes = int(getattr(accelerator, "num_processes", 1) or 1)
    ds_config["train_micro_batch_size_per_gpu"] = int(batch_size)
    ds_config["gradient_accumulation_steps"] = grad_accum
    ds_config["train_batch_size"] = int(batch_size) * grad_accum * num_processes
    print(
        "DeepSpeed batch setup: "
        f"train_micro_batch_size_per_gpu={ds_config['train_micro_batch_size_per_gpu']}, "
        f"gradient_accumulation_steps={ds_config['gradient_accumulation_steps']}, "
        f"train_batch_size={ds_config['train_batch_size']}"
    )


class BalancedTaskBatchSampler:
    def __init__(self, rows, task_batch: dict[str, int], *, seed: int = 0) -> None:
        self.rows = rows
        self.task_batch = task_batch
        self.seed = int(seed)
        self.batch_size = int(sum(task_batch.values()))
        self.pools: dict[str, list[int]] = {task: [] for task in task_batch}
        for index, row in enumerate(rows):
            task = str(row.get("task", ""))
            if task in self.pools:
                self.pools[task].append(index)
        missing = [task for task, pool in self.pools.items() if not pool]
        if missing:
            raise ValueError(f"Cannot build balanced batches; missing task rows: {missing}")
        self._length = max(math.ceil(len(self.pools[task]) / count) for task, count in self.task_batch.items())

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        rng = random.Random(self.seed)
        pools = {task: list(indices) for task, indices in self.pools.items()}
        cursors = {task: 0 for task in self.task_batch}
        for indices in pools.values():
            rng.shuffle(indices)
        for _ in range(len(self)):
            batch = []
            for task, count in self.task_batch.items():
                pool = pools[task]
                for _ in range(count):
                    if cursors[task] >= len(pool):
                        rng.shuffle(pool)
                        cursors[task] = 0
                    batch.append(pool[cursors[task]])
                    cursors[task] += 1
            rng.shuffle(batch)
            yield batch


def launch_tokenlight_training_task(
    accelerator,
    dataset,
    model,
    model_logger,
    args=None,
    **kwargs,
):
    del kwargs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    num_workers = args.dataset_num_workers
    save_steps = args.save_steps
    num_epochs = int(args.num_epochs)
    start_epoch = int(getattr(args, "start_epoch", 0) or 0)
    enable_model_cpu_offload = getattr(args, "enable_model_cpu_offload", False)
    enable_optimizer_cpu_offload = getattr(args, "enable_optimizer_cpu_offload", False)
    cpu_offload_split_threshold = getattr(args, "cpu_offload_split_threshold", None)
    customized_optimizer = args.customized_optimizer
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if batch_size > 1:
        warnings.warn(
            "TokenLight Wan batch_size > 1 uses true batched target/source VAE latents. "
            "Increase gradually because attention activations scale with batch size.",
            stacklevel=2,
        )

    trainable_dtype = _preferred_trainable_dtype(model)
    before_dtype_counts = _coerce_trainable_parameter_dtype(model, trainable_dtype)
    after_dtype_counts = _trainable_dtype_counts(model)
    if accelerator.is_main_process:
        print(
            "Trainable parameter dtype setup: "
            f"target={trainable_dtype}, before={before_dtype_counts}, after={after_dtype_counts}"
        )

    optimizer_class = get_optimizer_class(customized_optimizer)  # noqa: F405
    optimizer = optimizer_class(_trainable_parameters(model), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    task_batch = _parse_balanced_task_batch(getattr(args, "balanced_task_batch", None))
    dataloader_kwargs = _dataloader_runtime_kwargs(args, dataset, num_workers=num_workers, accelerator=accelerator)
    if task_batch is None:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate_tokenlight_batch,
            drop_last=batch_size > 1,
            **dataloader_kwargs,
        )
    else:
        if sum(task_batch.values()) != batch_size:
            raise ValueError(f"balanced_task_batch sums to {sum(task_batch.values())}, but batch_size={batch_size}")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=BalancedTaskBatchSampler(
                dataset.data,
                task_batch,
                seed=int(getattr(args, "balanced_batch_seed", 0)),
            ),
            collate_fn=_collate_tokenlight_batch,
            **dataloader_kwargs,
        )

    _configure_deepspeed_batch_size(accelerator, args, batch_size)
    if enable_model_cpu_offload:
        optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
        model.pipe.device = accelerator.device
        offload_manager = OffloadTrainingManager(  # noqa: F405
            model,
            accelerator.device,
            enable_optimizer_cpu_offload,
            cpu_offload_split_threshold,
        )
    else:
        model.to(device=accelerator.device)
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
        offload_manager = None

    save_training_runtime_snapshot(args, accelerator, model)
    tb_metrics = TokenLightTensorBoardMetrics(args.output_path, enabled=args.enable_tensorboard_log)
    initialize_deepspeed_gradient_checkpointing(accelerator)  # noqa: F405
    local_step = int(getattr(model_logger, "num_steps", 0))
    try:
        for epoch_id in range(start_epoch, start_epoch + num_epochs):
            for data in tqdm(dataloader):
                with accelerator.accumulate(model):
                    if dataset.load_from_cache:
                        loss = model({}, inputs=data)
                    else:
                        loss = model(data)
                    accelerator.backward(loss)
                    if enable_model_cpu_offload:
                        offload_manager.after_backward()
                    optimizer.step()
                    scheduler.step()
                    metrics = _collect_train_metrics(accelerator, model, optimizer, loss)
                    optimizer.zero_grad()
                    _call_compatible_method(model_logger, "on_step_end", accelerator, model, save_steps, loss=loss)
                    logger_step = getattr(model_logger, "num_steps", None)
                    if logger_step is None or int(logger_step) <= local_step:
                        local_step += 1
                    else:
                        local_step = int(logger_step)
                    tb_metrics.log(accelerator, local_step, metrics)
            if save_steps is None:
                _call_compatible_method(model_logger, "on_epoch_end", accelerator, model, epoch_id)

        _call_compatible_method(model_logger, "on_training_end", accelerator, model, save_steps)
    finally:
        tb_metrics.close(accelerator)


def _make_model_logger(args):
    logger_kwargs = {
        "remove_prefix_in_ckpt": args.remove_prefix_in_ckpt,
        "enable_tensorboard_log": getattr(args, "enable_tensorboard_log", False),
        "enable_swanlab_log": getattr(args, "enable_swanlab_log", False),
        "swanlab_project": getattr(args, "swanlab_project", "DiffSynth-Studio"),
        "enable_wandb_log": getattr(args, "enable_wandb_log", False),
        "wandb_project": getattr(args, "wandb_project", "DiffSynth-Studio"),
    }
    try:
        signature = inspect.signature(ModelLogger)  # noqa: F405
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        parameters = signature.parameters
        accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())
        if not accepts_kwargs:
            logger_kwargs = {key: value for key, value in logger_kwargs.items() if key in parameters}

    try:
        return ModelLogger(args.output_path, **logger_kwargs)  # noqa: F405
    except TypeError as exc:
        fallback_kwargs = {
            key: value for key, value in logger_kwargs.items() if key == "remove_prefix_in_ckpt"
        }
        warnings.warn(
            f"ModelLogger did not accept all logging arguments ({exc}); retrying with checkpoint-only logging args.",
            stacklevel=2,
        )
        try:
            return ModelLogger(args.output_path, **fallback_kwargs)  # noqa: F405
        except TypeError:
            warnings.warn("ModelLogger did not accept remove_prefix_in_ckpt; retrying with output path only.", stacklevel=2)
            return ModelLogger(args.output_path)  # noqa: F405


def main(train_mode: str = "single") -> None:
    args, raw_config, merged_config = parse_tokenlight_args(train_mode)
    accelerator_kwargs = {
        "kwargs_handlers": [
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)
        ],
    }
    if train_mode == "single":
        accelerator_kwargs["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    accelerator = accelerate.Accelerator(**accelerator_kwargs)
    save_training_config_snapshot(args, raw_config, merged_config, accelerator)
    dataset = build_dataset(args)
    model = TokenLightWanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=getattr(args, "use_gradient_checkpointing_offload", False),
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        resume_from_checkpoint=getattr(args, "resume_from_checkpoint", None),
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device="cpu" if (args.initialize_model_on_cpu or getattr(args, "enable_model_cpu_offload", False)) else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        tokenlight_light_tokens=args.tokenlight_light_tokens,
        tokenlight_attrs_key=args.tokenlight_attrs_key,
        tokenlight_token_dim=args.tokenlight_token_dim,
        tokenlight_fourier_features=args.tokenlight_fourier_features,
        tokenlight_fourier_sigma=args.tokenlight_fourier_sigma,
        tokenlight_max_lights=args.tokenlight_max_lights,
        tokenlight_light_dropout=args.tokenlight_light_dropout,
        tokenlight_cfg_drop_prob=args.tokenlight_cfg_drop_prob,
        tokenlight_source_tokens=args.tokenlight_source_tokens,
        tokenlight_mask_tokens=args.tokenlight_mask_tokens,
        prompt_context_cache_size=args.prompt_context_cache_size,
    )
    model_logger = _make_model_logger(args)
    launcher_map = {
        "sft:data_process": launch_data_process_task,  # noqa: F405
        "direct_distill:data_process": launch_data_process_task,  # noqa: F405
        "sft": launch_tokenlight_training_task,
        "sft:train": launch_tokenlight_training_task,
        "direct_distill": launch_tokenlight_training_task,
        "direct_distill:train": launch_tokenlight_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)


if __name__ == "__main__":
    main()
