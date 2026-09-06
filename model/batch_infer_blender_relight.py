from __future__ import annotations

import argparse
import concurrent.futures
import copy
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import re
import shutil
import sys
from typing import Any

from PIL import Image, ImageDraw

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        del kwargs
        return iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_INFER_CONFIG_PATH = None


@dataclass(frozen=True)
class InferJob:
    index: int
    scene_id: str
    light_id: int
    source: str
    target: str
    pred: str
    attrs: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch TokenLight inference and metrics for Blender Relight scenes.")
    parser.add_argument("--config", default=os.environ.get("TOKENLIGHT_INFER_CONFIG", DEFAULT_INFER_CONFIG_PATH))
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--scene_num", type=int, default=None)
    parser.add_argument("--gpu_devices", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--lora_checkpoint", default=None)
    parser.add_argument("--light_checkpoint", default=None)
    parser.add_argument("--metadata_jsonl", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--metric_device", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--withgt", action="store_true", help="Also save output+GT comparison images.")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Expected JSON object in {config_path}")
    return config


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    merged = copy.deepcopy(config)
    if args.data_root is not None:
        merged.setdefault("data", {})["data_root"] = args.data_root
    if args.scene_num is not None:
        merged.setdefault("data", {})["scene_num"] = int(args.scene_num)
    if args.metadata_jsonl is not None:
        merged.setdefault("data", {})["metadata_jsonl"] = args.metadata_jsonl
    if args.seed is not None:
        merged.setdefault("data", {})["seed"] = int(args.seed)
    if args.gpu_devices is not None:
        merged.setdefault("launch", {})["gpu_devices"] = args.gpu_devices
    if args.output is not None:
        merged.setdefault("output", {})["output_path"] = args.output
    if args.checkpoint is not None:
        merged.setdefault("model", {})["checkpoint"] = args.checkpoint
    if args.lora_checkpoint is not None:
        merged.setdefault("model", {})["lora_checkpoint"] = args.lora_checkpoint
    if args.light_checkpoint is not None:
        merged.setdefault("model", {})["light_checkpoint"] = args.light_checkpoint
    if args.metric_device is not None:
        merged.setdefault("metrics", {})["metric_device"] = args.metric_device
    if args.skip_existing:
        merged.setdefault("output", {})["skip_existing"] = True
    if args.withgt:
        merged.setdefault("output", {})["with_gt"] = True
    return merged


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def timestamped_path(path: Path, enabled: bool, timestamp_format: str) -> Path:
    if not enabled:
        return path
    timestamp = os.environ.get("TOKENLIGHT_RUN_TIMESTAMP") or datetime.now().strftime(timestamp_format)
    if path.suffix.lower() == ".json":
        return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
    return path.with_name(f"{path.name.rstrip('/')}_{timestamp}")


def output_layout(config: dict[str, Any]) -> tuple[Path, Path]:
    output_config = config.get("output", {})
    output_path = timestamped_path(
        resolve_path(output_config.get("output_path", "model/infer/tokenlight_blender_relight")),
        bool(output_config.get("append_timestamp", False)),
        str(output_config.get("timestamp_format", "%Y%m%d_%H%M%S")),
    )
    if output_path.suffix.lower() == ".json":
        return output_path.with_name(f"{output_path.stem}_images"), output_path
    return output_path, output_path / "metrics.json"


def parse_gpu_devices(value: str | list[Any] | None) -> list[str]:
    if isinstance(value, list):
        devices = [str(item).strip() for item in value if str(item).strip()]
    else:
        devices = [item.strip() for item in str(value or "0").split(",") if item.strip()]
    if not devices:
        raise ValueError("gpu_devices must contain at least one GPU id or cpu")
    normalized = []
    for device in devices:
        lowered = device.lower()
        if lowered == "cpu":
            normalized.append("cpu")
            continue
        if lowered.startswith("cuda:"):
            device = device.split(":", 1)[1]
        if not device.isdigit():
            raise ValueError(f"Unsupported GPU device value: {device!r}")
        normalized.append(device)
    return normalized


def scene_dirs(data_root: Path) -> list[Path]:
    root = data_root / "scenes" if (data_root / "scenes").exists() else data_root
    scenes = sorted(path for path in root.glob("scene_*") if path.is_dir())
    if not scenes:
        raise FileNotFoundError(f"No scene_* directories found under {data_root}")
    return scenes


def light_id_from_path(path: Path) -> int | None:
    match = re.fullmatch(r"light_(\d{3})\.png", path.name)
    return int(match.group(1)) if match else None


def available_light_ids(scene_dir: Path) -> list[int]:
    random_dir = light_image_dir(scene_dir)
    ids = []
    for path in random_dir.glob("light_*.png"):
        light_id = light_id_from_path(path)
        if light_id is not None:
            ids.append(light_id)
    return sorted(set(ids))


def source_image_path(scene_dir: Path) -> Path:
    for path in (
        scene_dir / "spatial_random" / "ambient.png",
        scene_dir / "spatial" / "ambient.png",
    ):
        if path.exists():
            return path
    return scene_dir / "spatial_random" / "ambient.png"


def light_image_dir(scene_dir: Path) -> Path:
    for path in (
        scene_dir / "spatial_random",
        scene_dir / "spatial" / "point_lights",
    ):
        if path.exists():
            return path
    return scene_dir / "spatial_random"


def maybe_float(value: Any, fallback: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def triple(value: Any, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    if value is None:
        return fallback
    values = list(value) if isinstance(value, (list, tuple)) else []
    values = (values + list(fallback))[:3]
    return maybe_float(values[0]), maybe_float(values[1]), maybe_float(values[2])


def attrs_from_record(record: dict[str, Any]) -> dict[str, float]:
    x, y, z = triple(record.get("canonical_position"), (float("nan"), float("nan"), float("nan")))
    r, g, b = triple(record.get("rgb_color"), (1.0, 1.0, 1.0))
    attrs = {
        "a": maybe_float(record.get("ambient_scale")),
        "x": x,
        "y": y,
        "z": z,
        "r": r,
        "g": g,
        "b": b,
        "lambda": maybe_float(record.get("lambda_intensity"), 1.0),
        "d": maybe_float(record.get("radius"), 0.06),
    }
    return {key: value for key, value in attrs.items() if math.isfinite(float(value))}


def attrs_from_meta(scene_dir: Path, light_id: int) -> dict[str, float]:
    meta_path = scene_dir / "meta.json"
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as exc:
        raise ValueError(f"Could not read valid JSON from {meta_path}: {exc}") from exc
    lights = meta.get("spatial", {}).get("point_lights", [])
    light = next((item for item in lights if int(item.get("id", -1)) == int(light_id)), None)
    if light is None:
        raise ValueError(f"Could not find light {light_id:03d} in {meta_path}")
    if light.get("valid") is False:
        raise ValueError(f"Light {light_id:03d} is marked invalid in {meta_path}")
    x, y, z = triple(light.get("canonical_position"), (float("nan"), float("nan"), float("nan")))
    r, g, b = triple(light.get("component_color"), (1.0, 1.0, 1.0))
    ambient = meta.get("spatial", {}).get("ambient_source", {})
    lambda_intensity = maybe_float(light.get("lambda_intensity"), float("nan"))
    if not math.isfinite(lambda_intensity):
        lambda_intensity = maybe_float(light.get("canonical_energy"), 500.0) / 500.0
    return {
        "a": maybe_float(ambient.get("strength"), 1.0),
        "x": x,
        "y": y,
        "z": z,
        "r": r,
        "g": g,
        "b": b,
        "lambda": lambda_intensity,
        "d": maybe_float(light.get("canonical_radius"), 0.06),
    }


def metadata_candidates(data_root: Path, explicit: str = "") -> list[Path]:
    if explicit:
        return [resolve_path(explicit)]
    candidates = [
        data_root / "metadata.jsonl",
        data_root / "metadata_nomask.jsonl",
        data_root.parent / "metadata.jsonl",
        data_root.parent / "metadata_nomask.jsonl",
        data_root.parent / "synthetic" / "tokenlight_synthetic_1280_random_spatial_color" / "metadata.jsonl",
    ]
    for parent in (data_root, data_root.parent):
        if parent.exists():
            try:
                for path in parent.glob("**/metadata.jsonl"):
                    if path not in candidates:
                        candidates.append(path)
            except OSError:
                continue
    return candidates


def load_metadata_index(
    data_root: Path,
    explicit: str,
    wanted_scenes: set[str],
) -> tuple[dict[tuple[str, int], dict[str, Any]], Path | None]:
    for path in metadata_candidates(data_root, explicit):
        if not path.exists():
            continue
        index: dict[tuple[str, int], dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                scene_id = str(record.get("scene_id", ""))
                if scene_id not in wanted_scenes:
                    continue
                light_id = int(record.get("light_id", -1))
                if light_id >= 0:
                    index[(scene_id, light_id)] = record
        if index:
            return index, path
    return {}, None


def select_jobs(
    data_root: Path,
    scene_num: int,
    output_dir: Path,
    seed: int,
    metadata_index: dict[tuple[str, int], dict[str, Any]],
) -> tuple[list[InferJob], list[str]]:
    if scene_num <= 0:
        raise ValueError("scene_num must be positive")
    jobs: list[InferJob] = []
    warnings: list[str] = []
    valid_metadata_ids: dict[str, set[int]] = {}
    for (scene_id, light_id), record in metadata_index.items():
        if record.get("valid") is not False:
            valid_metadata_ids.setdefault(scene_id, set()).add(light_id)
    for scene_dir in scene_dirs(data_root):
        scene_id = scene_dir.name
        source = source_image_path(scene_dir)
        if not source.exists():
            warnings.append(f"{scene_id}: missing source {source}")
            continue
        light_ids = available_light_ids(scene_dir)
        if scene_id in valid_metadata_ids:
            light_ids = sorted(set(light_ids) & valid_metadata_ids[scene_id])
        if not light_ids:
            warnings.append(f"{scene_id}: no light_*.png files under {light_image_dir(scene_dir)}")
            continue
        if len(light_ids) < scene_num:
            warnings.append(f"{scene_id}: requested {scene_num} lights but only {len(light_ids)} exist")
        rng = random.Random(f"{seed}:{scene_id}")
        chosen = sorted(rng.sample(light_ids, min(scene_num, len(light_ids))))
        for light_id in chosen:
            target = light_image_dir(scene_dir) / f"light_{light_id:03d}.png"
            record = metadata_index.get((scene_id, light_id))
            if record is not None:
                if record.get("valid") is False:
                    warnings.append(f"{scene_id}/light_{light_id:03d}: skipped invalid metadata record")
                    continue
                attrs = attrs_from_record(record)
            else:
                try:
                    attrs = attrs_from_meta(scene_dir, light_id)
                except ValueError as exc:
                    warnings.append(f"{scene_id}/light_{light_id:03d}: skipped metadata fallback ({exc})")
                    continue
                warnings.append(f"{scene_id}/light_{light_id:03d}: attrs from meta.json fallback")
            pred = output_dir / f"{scene_id}_light_{light_id:03d}.png"
            jobs.append(
                InferJob(
                    index=len(jobs),
                    scene_id=scene_id,
                    light_id=light_id,
                    source=source.as_posix(),
                    target=target.as_posix(),
                    pred=pred.as_posix(),
                    attrs=attrs,
                )
            )
    if not jobs:
        raise FileNotFoundError(f"No inference jobs could be built from {data_root}")
    return jobs, warnings


def infer_args_from_config(config: dict[str, Any], device: str):
    from types import SimpleNamespace

    values: dict[str, Any] = {}
    values.update(config.get("model", {}))
    values.update(config.get("inference", {}))
    for key in ("weights_dir", "checkpoint", "lora_checkpoint", "light_checkpoint"):
        value = values.get(key)
        if value:
            values[key] = resolve_path(value).as_posix()
    values["device"] = device
    return SimpleNamespace(**values)


def setup_pipeline(args):
    from model.infer_tokenlight import (
        extract_light_state,
        extract_lora_state,
        extract_type_state,
        infer_light_encoder_shape,
        infer_type_embedding_num_types,
        load_pipe,
        load_state,
    )
    from model.lightoken_encoder import LightokenEncoder
    from model.tokenlight_wan import TokenLightTypeEmbedding

    pipe = load_pipe(args)
    combined = load_state(getattr(args, "checkpoint", ""))
    lora_state = load_state(args.lora_checkpoint) if getattr(args, "lora_checkpoint", "") else combined
    lora = extract_lora_state(lora_state)
    if lora:
        pipe.load_lora(pipe.dit, state_dict=lora, alpha=1.0)

    token_dim = int(args.token_dim) if int(getattr(args, "token_dim", 0)) > 0 else int(pipe.dit.dim)
    light_checkpoint_state = load_state(args.light_checkpoint) if getattr(args, "light_checkpoint", "") else combined
    light_state = extract_light_state(light_checkpoint_state)
    max_lights, fourier_features = infer_light_encoder_shape(
        light_state,
        requested_max_lights=getattr(args, "tokenlight_max_lights", getattr(args, "max_lights", 1)),
        requested_fourier_features=int(args.fourier_features),
    )
    light_encoder = LightokenEncoder(
        token_dim,
        fourier_features=fourier_features,
        fourier_sigma=float(args.fourier_sigma),
        max_lights=max_lights,
    ).to(device=pipe.device, dtype=pipe.torch_dtype)
    if light_state:
        light_encoder.load_state_dict(light_state, strict=False)
    light_encoder.eval()

    type_embedding = None
    type_state = extract_type_state(light_checkpoint_state)
    if type_state:
        num_types = infer_type_embedding_num_types(type_state, requested_num_types=4)
        type_embedding = TokenLightTypeEmbedding(token_dim, num_types=num_types).to(
            device=pipe.device,
            dtype=pipe.torch_dtype,
        )
        type_embedding.load_state_dict(type_state, strict=False)
        type_embedding.eval()
    return pipe, light_encoder, type_embedding


def with_gt_path(pred: Path) -> Path:
    suffix = pred.suffix if pred.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} else ".png"
    return pred.with_name(f"{pred.stem}_withgt{suffix}")


def save_with_gt_image(pred: str | Path, target: str | Path) -> None:
    pred_path = Path(pred)
    if pred_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        return
    output = Image.open(pred_path).convert("RGB")
    gt = Image.open(target).convert("RGB").resize(output.size)
    label_height = max(32, output.height // 24)
    canvas = Image.new("RGB", (output.width + gt.width, output.height + label_height), "white")
    canvas.paste(output, (0, 0))
    canvas.paste(gt, (output.width, 0))
    draw = ImageDraw.Draw(canvas)
    y = output.height + label_height // 2
    for label, center_x in (("output", output.width // 2), ("gt", output.width + gt.width // 2)):
        bbox = draw.textbbox((0, 0), label)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text((center_x - text_width // 2, y - text_height // 2), label, fill="black")
    canvas.save(with_gt_path(pred_path))


def run_worker(device_id: str, jobs: list[InferJob], config: dict[str, Any], skip_existing: bool) -> list[str]:
    if device_id == "cpu":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        device = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        device = "cuda:0"

    from diffsynth.utils.data import save_video
    from model.infer_tokenlight import generate

    args = infer_args_from_config(config, device)
    pipe, light_encoder, type_embedding = setup_pipeline(args)
    with_gt = bool(config.get("output", {}).get("with_gt", False))
    completed = []
    for job in jobs:
        pred = Path(job.pred)
        if skip_existing and pred.exists():
            if with_gt and not with_gt_path(pred).exists():
                save_with_gt_image(pred, job.target)
            completed.append(job.pred)
            print(f"[skip:{device_id}] {job.scene_id} light_{job.light_id:03d}", flush=True)
            continue
        pred.parent.mkdir(parents=True, exist_ok=True)
        source = Image.open(job.source).convert("RGB")
        print(f"[infer:{device_id}] {job.scene_id} light_{job.light_id:03d}", flush=True)
        video = generate(pipe, light_encoder, type_embedding, job.attrs, source, None, args)
        if pred.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            video[0].save(pred)
        else:
            save_video(video, str(pred), fps=int(args.fps), quality=5)
        if with_gt:
            save_with_gt_image(pred, job.target)
        completed.append(job.pred)
    return completed


def run_jobs(config: dict[str, Any], jobs: list[InferJob], gpu_devices: list[str], skip_existing: bool) -> None:
    by_device = {device: [] for device in gpu_devices}
    for index, job in enumerate(jobs):
        by_device[gpu_devices[index % len(gpu_devices)]].append(job)
    assigned = [(device, items) for device, items in by_device.items() if items]
    if len(assigned) == 1:
        run_worker(assigned[0][0], assigned[0][1], config, skip_existing)
        return
    context = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(assigned), mp_context=context) as executor:
        futures = [executor.submit(run_worker, device, items, config, skip_existing) for device, items in assigned]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def load_tensor_image(path: str | Path):
    import numpy as np
    import torch

    image = Image.open(path).convert("RGB")
    array = np.asarray(image).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
    return tensor


def psnr(pred, target, max_val: float = 1.0, eps: float = 1e-8):
    import torch
    from torch.nn import functional as F

    mse = F.mse_loss(pred.float(), target.float())
    return 20.0 * torch.log10(torch.tensor(max_val, device=pred.device)) - 10.0 * torch.log10(mse + eps)


def ssim(pred, target, max_val: float = 1.0, window_size: int = 11):
    import torch
    from torch.nn import functional as F

    pred = pred.float()
    target = target.float()
    padding = window_size // 2
    channels = pred.shape[1]
    weight = torch.ones(channels, 1, window_size, window_size, device=pred.device) / (window_size * window_size)
    mu_x = F.conv2d(pred, weight, padding=padding, groups=channels)
    mu_y = F.conv2d(target, weight, padding=padding, groups=channels)
    sigma_x = F.conv2d(pred * pred, weight, padding=padding, groups=channels) - mu_x.square()
    sigma_y = F.conv2d(target * target, weight, padding=padding, groups=channels) - mu_y.square()
    sigma_xy = F.conv2d(pred * target, weight, padding=padding, groups=channels) - mu_x * mu_y
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )
    return score.mean()


class LpipsMetric:
    def __init__(self, device: Any, net: str = "alex") -> None:
        import lpips

        self.device = device
        self.model = lpips.LPIPS(net=net).to(device).eval()

    def __call__(self, pred, target) -> float:
        import torch

        with torch.no_grad():
            return float(self.model(pred * 2.0 - 1.0, target * 2.0 - 1.0).mean().detach().cpu())


def average_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("psnr", "ssim", "lpips")
    if not records:
        return {key: float("nan") for key in keys}
    return {key: float(sum(float(record["metrics"][key]) for record in records) / len(records)) for key in keys}


def resolve_metric_device(value: str, gpu_devices: list[str]):
    import torch

    if value != "auto":
        return torch.device(value)
    first = next((device for device in gpu_devices if device != "cpu"), None)
    if first is not None and torch.cuda.is_available():
        index = int(first)
        if index < torch.cuda.device_count():
            return torch.device(f"cuda:{index}")
        return torch.device("cuda:0")
    return torch.device("cpu")


def evaluate_jobs(jobs: list[InferJob], device: Any, lpips_net: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    lpips_metric = LpipsMetric(device, net=lpips_net)
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for job in tqdm(jobs, desc="metrics"):
            pred = load_tensor_image(job.pred).unsqueeze(0).to(device)
            target = load_tensor_image(job.target).unsqueeze(0).to(device)
            if pred.shape != target.shape:
                raise ValueError(f"Shape mismatch for {job.pred}: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
            metrics = {
                "psnr": float(psnr(pred, target).detach().cpu()),
                "ssim": float(ssim(pred, target).detach().cpu()),
                "lpips": lpips_metric(pred, target),
            }
            records.append({**asdict(job), "metrics": metrics})

    per_scene: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        per_scene.setdefault(str(record["scene_id"]), []).append(record)
    scene_averages = {scene_id: {"count": len(items), **average_metrics(items)} for scene_id, items in sorted(per_scene.items())}
    scene_mean_records = [
        {"metrics": {key: value for key, value in metrics.items() if key != "count"}}
        for metrics in scene_averages.values()
    ]
    return records, {
        "summary": {
            "count": len(records),
            "overall": average_metrics(records),
            "scene_mean_average": average_metrics(scene_mean_records),
        },
        "scene_averages": scene_averages,
    }


def save_config_snapshot(config_path: str, raw_config: dict[str, Any], resolved_config: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = resolve_path(config_path)
    if source.exists():
        shutil.copy2(source, output_dir / "infer_config.json")
    else:
        (output_dir / "infer_config.json").write_text(json.dumps(raw_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "infer_config_resolved.json").write_text(
        json.dumps(resolved_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.config is None:
        raise ValueError("Batch inference now requires an explicit --config path.")
    raw_config = load_json(args.config)
    config = apply_cli_overrides(raw_config, args)
    data_root = resolve_path(config.get("data", {}).get("data_root", "data/blender_relight/test"))
    scene_num = int(config.get("data", {}).get("scene_num", 1))
    seed = int(config.get("data", {}).get("seed", 1234))
    gpu_devices = parse_gpu_devices(config.get("launch", {}).get("gpu_devices", "0"))
    output_dir, metrics_path = output_layout(config)
    skip_existing = bool(config.get("output", {}).get("skip_existing", False))
    metadata_jsonl = str(config.get("data", {}).get("metadata_jsonl", "") or "")

    scenes = scene_dirs(data_root)
    metadata_index, metadata_path = load_metadata_index(data_root, metadata_jsonl, {scene.name for scene in scenes})
    jobs, warnings = select_jobs(data_root, scene_num, output_dir, seed, metadata_index)
    if metadata_path is None:
        warnings.append("No matching metadata.jsonl found; using meta.json fallback attrs.")

    resolved_config = copy.deepcopy(config)
    resolved_config.setdefault("resolved", {})
    resolved_config["resolved"].update(
        {
            "data_root": data_root.as_posix(),
            "output_dir": output_dir.as_posix(),
            "metrics_path": metrics_path.as_posix(),
            "gpu_devices": gpu_devices,
            "scene_count": len(scenes),
            "job_count": len(jobs),
            "metadata_jsonl": "" if metadata_path is None else metadata_path.as_posix(),
        }
    )
    save_config_snapshot(args.config, raw_config, resolved_config, output_dir)

    print(f"[batch] scenes={len(scenes)} jobs={len(jobs)} output={output_dir}", flush=True)
    print(f"[batch] gpu_devices={','.join(gpu_devices)} scene_num={scene_num}", flush=True)
    if metadata_path is not None:
        print(f"[batch] attrs metadata={metadata_path}", flush=True)
    if args.dry_run:
        for job in jobs:
            print(f"[dry-run] {job.scene_id} light_{job.light_id:03d} -> {job.pred}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    run_jobs(config, jobs, gpu_devices, skip_existing)
    metric_config = config.get("metrics", {})
    records, metrics_payload = evaluate_jobs(
        jobs,
        resolve_metric_device(str(metric_config.get("metric_device", "auto")), gpu_devices),
        str(metric_config.get("lpips_net", "alex")),
    )
    payload = {
        "config": args.config,
        "data_root": data_root.as_posix(),
        "output_dir": output_dir.as_posix(),
        "metrics_path": metrics_path.as_posix(),
        "scene_num": scene_num,
        "seed": seed,
        "gpu_devices": gpu_devices,
        "metadata_jsonl": "" if metadata_path is None else metadata_path.as_posix(),
        "warnings": warnings,
        **metrics_payload,
        "records": records,
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[batch] wrote metrics: {metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
