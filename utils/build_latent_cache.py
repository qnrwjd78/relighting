#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CACHE_VERSION = 1
DEFAULT_DATA_ROOT = "data/objaverse_ratio3p5_cube1p6_direct_scene0000_1999_640_png"
MODE_PRESETS = {
    "rgb": ("video,input_image", "none"),
    "pbr": ("pbr_depth_image,pbr_normal_image", "none"),
    "rgb-pbr": ("video,input_image,pbr_depth_image,pbr_normal_image", "none"),
    "luminance": ("video", "luminance"),
    "log-luminance": ("video", "log_luminance"),
}

torch = None
Image = None
DataLoader = None
tqdm = None
ModelConfig = None
WanVideoPipeline = None
make_illumination_image_tensor = None
unit_to_vae_range = None


@dataclass(frozen=True)
class CacheAsset:
    asset_index: int
    path: str
    keys: tuple[str, ...]


def runtime_imports(*, include_model: bool, include_illumination: bool = False) -> None:
    global torch, Image, DataLoader, tqdm, ModelConfig, WanVideoPipeline
    global make_illumination_image_tensor, unit_to_vae_range
    if torch is None:
        import torch as _torch
        from PIL import Image as _Image
        from torch.utils.data import DataLoader as _DataLoader
        from tqdm import tqdm as _tqdm

        torch, Image, DataLoader, tqdm = _torch, _Image, _DataLoader, _tqdm
    if include_model and WanVideoPipeline is None:
        from diffsynth.pipelines.wan_video import ModelConfig as _ModelConfig
        from diffsynth.pipelines.wan_video import WanVideoPipeline as _WanVideoPipeline

        ModelConfig, WanVideoPipeline = _ModelConfig, _WanVideoPipeline
    if include_illumination and make_illumination_image_tensor is None:
        from model.illumination_latent_head import make_illumination_image_tensor as _make
        from model.illumination_latent_head import unit_to_vae_range as _to_range

        make_illumination_image_tensor, unit_to_vae_range = _make, _to_range


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RGB, PBR, luminance, or log-luminance sharded VAE latent caches."
    )
    parser.add_argument("--mode", choices=(*MODE_PRESETS, "custom", "all"), default="rgb-pbr")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--metadata-path", default="")
    parser.add_argument("--output-dir", "--output-path", "--output", default="")
    parser.add_argument("--image-keys", default="", help="Overrides the selected mode's image keys.")
    parser.add_argument("--transform", choices=("none", "luminance", "log_luminance"), default="none", help="Used by --mode custom.")
    parser.add_argument("--log-luminance-eps", type=float, default=1e-3)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--weights-dir", default="weights/Wan2.2-TI2V-5B")
    parser.add_argument("--vae-path", default="")
    parser.add_argument("--gpu-devices", "--gpu_devices", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--vae-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--save-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--vae-tiled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tile-size", type=int, nargs=2, default=(30, 52))
    parser.add_argument("--tile-stride", type=int, nargs=2, default=(15, 26))
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--partition-count", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--partition-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--shard-prefix", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-output-prepare", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def csv_items(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def mode_settings(args: argparse.Namespace) -> tuple[list[str], str]:
    if args.mode == "custom":
        if not args.image_keys:
            raise ValueError("--mode custom requires --image-keys")
        return csv_items(args.image_keys), str(args.transform)
    keys, transform = MODE_PRESETS[args.mode]
    return csv_items(args.image_keys or keys), transform


def default_metadata(data_root: Path) -> Path:
    return REPO_ROOT / "data_train" / data_root.name / "metadata.jsonl"


def default_output(data_root: Path, mode: str) -> Path:
    return REPO_ROOT / "data_train" / data_root.name / "latent_cache" / mode.replace("-", "_")


def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def canonical_path(value: str, data_root: Path) -> str:
    raw = Path(value)
    if not raw.is_absolute():
        return raw.as_posix()
    try:
        return raw.relative_to(data_root).as_posix()
    except ValueError:
        return raw.as_posix()


def discover_assets(rows: list[dict[str, Any]], data_root: Path, keys: list[str], maximum: int | None) -> list[CacheAsset]:
    paths: OrderedDict[str, set[str]] = OrderedDict()
    for row in rows:
        if row.get("valid") is False:
            continue
        for key in keys:
            value = row.get(key)
            if value in (None, ""):
                continue
            path = canonical_path(str(value), data_root)
            paths.setdefault(path, set()).add(key)
            if maximum is not None and len(paths) >= maximum:
                break
        if maximum is not None and len(paths) >= maximum:
            break
    return [CacheAsset(index, path, tuple(sorted(asset_keys))) for index, (path, asset_keys) in enumerate(paths.items())]


def cache_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:20]


class AssetDataset:
    def __init__(self, assets: list[CacheAsset], data_root: Path, size: tuple[int, int]) -> None:
        self.assets, self.data_root, self.size = assets, data_root, size

    def __len__(self) -> int:
        return len(self.assets)

    def __getitem__(self, index: int) -> tuple[int, str, Any]:
        runtime_imports(include_model=False)
        asset = self.assets[index]
        path = Path(asset.path)
        path = path if path.is_absolute() else self.data_root / path
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.size != self.size:
                image = image.resize(self.size, Image.Resampling.BILINEAR)
            image = image.copy()
        return index, asset.path, image


def collate(items: list[tuple[int, str, Any]]) -> tuple[list[int], list[str], list[Any]]:
    indices, paths, images = zip(*items)
    return list(indices), list(paths), list(images)


def load_pipe(args: argparse.Namespace) -> Any:
    runtime_imports(include_model=True)
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    vae_path = repo_path(args.vae_path or Path(args.weights_dir) / "Wan2.2_VAE.pth")
    if not vae_path.is_file():
        raise FileNotFoundError(vae_path)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16 if args.vae_dtype == "bf16" else torch.float32,
        device=device,
        model_configs=[ModelConfig(str(vae_path))],
        tokenizer_config=None,
    )
    pipe.load_models_to_device(["vae"])
    pipe.vae.eval()
    for parameter in pipe.vae.parameters():
        parameter.requires_grad_(False)
    return pipe


def output_dtype(name: str) -> Any:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def preprocess(pipe: Any, images: list[Any], transform: str, eps: float) -> Any:
    if transform == "none":
        videos = [pipe.preprocess_video([image], torch_dtype=torch.float32, device=pipe.device) for image in images]
        return torch.cat(videos, dim=0)
    runtime_imports(include_model=False, include_illumination=True)
    unit = torch.cat(
        [
            pipe.preprocess_video([image], torch_dtype=torch.float32, device=pipe.device, min_value=0, max_value=1)
            for image in images
        ],
        dim=0,
    )
    target = "luminance" if transform == "luminance" else "log_luminance"
    return unit_to_vae_range(make_illumination_image_tensor(unit, target=target, eps=eps))


def encode(pipe: Any, images: list[Any], transform: str, args: argparse.Namespace) -> Any:
    with torch.no_grad():
        pixels = preprocess(pipe, images, transform, float(args.log_luminance_eps))
        latents = pipe.vae.encode(
            pixels.to(dtype=pipe.torch_dtype, device=pipe.device),
            device=pipe.device,
            tiled=bool(args.vae_tiled),
            tile_size=tuple(args.tile_size),
            tile_stride=tuple(args.tile_stride),
        )
    return latents.to(dtype=output_dtype(args.save_dtype), device="cpu").contiguous()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_index(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} is not empty; pass --overwrite")
    (path / "shards").mkdir(parents=True, exist_ok=True)
    if overwrite:
        for item in path.glob("*.json*"):
            if item.is_file():
                item.unlink()
        for item in (path / "shards").glob("*"):
            if item.is_file():
                item.unlink()


def atomic_shard(tensors: dict[str, Any], path: Path, metadata: dict[str, str]) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    save_file(tensors, str(temporary), metadata=metadata)
    os.replace(temporary, path)


def cache_config(
    args: argparse.Namespace,
    data_root: Path,
    metadata: Path,
    output: Path,
    keys: list[str],
    transform: str,
    rows: list[dict[str, Any]],
    assets: list[CacheAsset],
) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    for asset in assets:
        for key in asset.keys:
            counts[key] += 1
    return {
        "cache_version": CACHE_VERSION,
        "cache_kind": "vae_latent_cache" if transform == "none" else "illumination_latent_cache",
        "mode": args.mode,
        "transform": transform,
        "log_luminance_eps": float(args.log_luminance_eps),
        "data_root": str(data_root),
        "metadata_path": str(metadata),
        "output_dir": str(output),
        "image_keys": keys,
        "height": int(args.height),
        "width": int(args.width),
        "vae_path": str(repo_path(args.vae_path or Path(args.weights_dir) / "Wan2.2_VAE.pth")),
        "vae_dtype": args.vae_dtype,
        "save_dtype": args.save_dtype,
        "vae_tiled": bool(args.vae_tiled),
        "tile_size": list(args.tile_size),
        "tile_stride": list(args.tile_stride),
        "batch_size": int(args.batch_size),
        "shard_size": int(args.shard_size),
        "row_count": len(rows),
        "asset_count": len(assets),
        "asset_counts_by_key": dict(sorted(counts.items())),
        "partition_count": int(args.partition_count),
        "partition_index": int(args.partition_index),
    }


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    if args.gpu_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_devices)
    keys, transform = mode_settings(args)
    data_root = repo_path(args.data_root)
    metadata = repo_path(args.metadata_path) if args.metadata_path else default_metadata(data_root)
    output = repo_path(args.output_dir) if args.output_dir else default_output(data_root, args.mode)
    rows = load_rows(metadata, args.limit_rows)
    all_assets = discover_assets(rows, data_root, keys, args.max_items)
    if not all_assets:
        raise ValueError(f"No assets found for {keys} in {metadata}")
    if args.partition_index < 0 or args.partition_index >= args.partition_count:
        raise ValueError("partition-index is outside partition-count")
    assets = all_assets[args.partition_index::args.partition_count]
    config = cache_config(args, data_root, metadata, output, keys, transform, rows, all_assets)
    config["partition_asset_count"] = len(assets)

    if args.dry_run:
        print(json.dumps(config, indent=2))
        return config
    if not args.skip_output_prepare:
        prepare_output(output, bool(args.overwrite))
        write_json(output / "cache_config.json", config)

    runtime_imports(include_model=False)
    pipe = load_pipe(args)
    dataset = AssetDataset(assets, data_root, (int(args.width), int(args.height)))
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate,
        pin_memory=False,
        drop_last=False,
    )
    index_rows: list[dict[str, Any]] = []
    shard_tensors: dict[str, Any] = {}
    shard_rows: list[dict[str, Any]] = []
    shard_index = 0

    def flush() -> None:
        nonlocal shard_index, shard_tensors, shard_rows
        if not shard_tensors:
            return
        relative = Path("shards") / f"{args.shard_prefix}shard_{shard_index:06d}.safetensors"
        atomic_shard(
            shard_tensors,
            output / relative,
            {
                "cache_version": str(CACHE_VERSION),
                "mode": args.mode,
                "transform": transform,
                "save_dtype": args.save_dtype,
                "height": str(args.height),
                "width": str(args.width),
            },
        )
        for row in shard_rows:
            row["shard"] = relative.as_posix()
        index_rows.extend(shard_rows)
        shard_tensors, shard_rows = {}, []
        shard_index += 1

    progress = tqdm(loader, desc=f"encode {args.mode} latents")
    for local_indices, paths, images in progress:
        latents = encode(pipe, images, transform, args)
        for offset, (local_index, path) in enumerate(zip(local_indices, paths)):
            asset = assets[int(local_index)]
            identity = f"{transform}:{asset.path}" if transform != "none" else asset.path
            prefix = f"{transform}_" if transform != "none" else ""
            tensor_name = f"latent_{prefix}{cache_id(identity)}"
            latent = latents[offset].contiguous()
            shard_tensors[tensor_name] = latent
            row = {
                "asset_index": int(asset.asset_index),
                "partition_index": int(args.partition_index),
                "path": path,
                "keys": list(asset.keys),
                "tensor": tensor_name,
                "shape": list(latent.shape),
                "dtype": str(latent.dtype).replace("torch.", ""),
            }
            if transform != "none":
                row.update({"target": transform, "eps": float(args.log_luminance_eps)})
            shard_rows.append(row)
            if len(shard_tensors) >= int(args.shard_size):
                flush()
        progress.set_postfix(assets=len(index_rows) + len(shard_rows))
    flush()

    index_name = f"index_part_{args.partition_index:02d}.jsonl" if args.partition_count > 1 else "index.jsonl"
    write_index(output / index_name, index_rows)
    summary = {**config, "partition_asset_count": len(assets), "shard_count": shard_index, "index_path": str(output / index_name)}
    summary_name = f"cache_summary_part_{args.partition_index:02d}.json" if args.partition_count > 1 else "cache_summary.json"
    write_json(output / summary_name, summary)
    print(json.dumps(summary, indent=2))
    return summary


def devices(value: str) -> list[str]:
    return csv_items(value)


def worker_command(args: argparse.Namespace, gpu: str, count: int, index: int) -> list[str]:
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--mode", args.mode,
        "--data-root", args.data_root,
        "--metadata-path", args.metadata_path,
        "--output-dir", args.output_dir,
        "--image-keys", args.image_keys,
        "--transform", args.transform,
        "--log-luminance-eps", str(args.log_luminance_eps),
        "--height", str(args.height), "--width", str(args.width),
        "--weights-dir", args.weights_dir, "--vae-path", args.vae_path,
        "--gpu-devices", gpu, "--device", args.device,
        "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers),
        "--shard-size", str(args.shard_size), "--vae-dtype", args.vae_dtype, "--save-dtype", args.save_dtype,
        "--tile-size", *(str(value) for value in args.tile_size),
        "--tile-stride", *(str(value) for value in args.tile_stride),
        "--partition-count", str(count), "--partition-index", str(index),
        "--shard-prefix", f"part_{index:02d}_", "--worker", "--skip-output-prepare",
    ]
    command.append("--vae-tiled" if args.vae_tiled else "--no-vae-tiled")
    if args.limit_rows is not None:
        command.extend(("--limit-rows", str(args.limit_rows)))
    if args.max_items is not None:
        command.extend(("--max-items", str(args.max_items)))
    return command


def merge_indexes(output: Path, count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(count):
        path = output / f"index_part_{index:02d}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    rows.sort(key=lambda row: int(row["asset_index"]))
    write_index(output / "index.jsonl", rows)
    return rows


def launch_multi_gpu(args: argparse.Namespace) -> dict[str, Any]:
    gpu_list = devices(args.gpu_devices)
    data_root = repo_path(args.data_root)
    output = repo_path(args.output_dir) if args.output_dir else default_output(data_root, args.mode)
    args.output_dir = str(output)
    prepare_output(output, bool(args.overwrite))
    processes = [
        (gpu, subprocess.Popen(worker_command(args, gpu, len(gpu_list), index), cwd=REPO_ROOT))
        for index, gpu in enumerate(gpu_list)
    ]
    failures = [(gpu, process.wait()) for gpu, process in processes]
    failures = [(gpu, code) for gpu, code in failures if code]
    if failures:
        raise RuntimeError(f"cache workers failed: {failures}")
    rows = merge_indexes(output, len(gpu_list))
    first = json.loads((output / "cache_summary_part_00.json").read_text(encoding="utf-8"))
    summary = {
        **first,
        "partition_count": len(gpu_list),
        "partition_index": None,
        "gpu_devices": gpu_list,
        "merged_index_rows": len(rows),
        "shard_count": len(list((output / "shards").glob("*.safetensors"))),
        "index_path": str(output / "index.jsonl"),
    }
    write_json(output / "cache_config.json", summary)
    write_json(output / "cache_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def run_one(args: argparse.Namespace) -> dict[str, Any]:
    if not args.worker and not args.dry_run and args.partition_count == 1 and len(devices(args.gpu_devices)) > 1:
        return launch_multi_gpu(args)
    return build_cache(args)


def main() -> int:
    args = parse_args()
    if args.mode != "all":
        run_one(args)
        return 0

    root = repo_path(args.output_dir) if args.output_dir else default_output(repo_path(args.data_root), "all")
    summaries = {}
    for mode in ("rgb-pbr", "luminance", "log-luminance"):
        child = argparse.Namespace(**vars(args))
        child.mode = mode
        child.output_dir = str(root / mode.replace("-", "_"))
        child.image_keys = ""
        summaries[mode] = run_one(child)
    write_json(root / "all_cache_summary.json", {"cache_version": CACHE_VERSION, "modes": summaries})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
