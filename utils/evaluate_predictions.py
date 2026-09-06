#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import infer_manifest as inference  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RGB or PBR-TokenLight manifest predictions with PSNR, SSIM, and LPIPS."
    )
    parser.add_argument("--infer-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--output", default="", help="Default: <infer-dir>/metrics.json")
    parser.add_argument("--prediction-suffix", default="")
    parser.add_argument("--target-key", default="target_image")
    parser.add_argument("--target-fallback-key", default="video")
    parser.add_argument("--target-transform", choices=("none", "luminance", "log_luminance"), default="none")
    parser.add_argument("--metric-device", default="auto")
    parser.add_argument("--lpips-net", default="alex")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-missing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_repo(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("valid") is False:
                continue
            row["_manifest_index"] = index
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def row_value(row: dict[str, Any], key: str, fallback_key: str = "") -> Any:
    value = row.get(key)
    return value if value not in (None, "") else row.get(fallback_key)


def resolve_data(value: Any, base_path: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base_path / path


def prediction_name(row: dict[str, Any], suffix: str) -> str:
    scene_id = str(row.get("scene_id") or f"item_{int(row.get('_manifest_index', 0)):06d}")
    light_id = row.get("light_id")
    stem = scene_id if light_id is None else f"{scene_id}_light_{int(light_id):03d}"
    return f"{stem}{suffix}.png"


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def transform_target(tensor: torch.Tensor, transform: str) -> torch.Tensor:
    if transform == "none":
        return tensor
    weights = tensor.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 3, 1, 1)
    luminance = (tensor[:, :3] * weights).sum(dim=1, keepdim=True).clamp(0.0, 1.0)
    if transform == "log_luminance":
        eps = 1e-3
        log_eps = math.log(eps)
        luminance = ((torch.log(luminance.clamp_min(eps)) - log_eps) / -log_eps).clamp(0.0, 1.0)
    return luminance.expand(-1, 3, -1, -1).contiguous()


def finite_average(values: list[float]) -> float | None:
    valid = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(valid) / len(valid)) if valid else None


def average_metrics(records: list[dict[str, Any]]) -> dict[str, float | None]:
    return {
        metric: finite_average([record["metrics"][metric] for record in records])
        for metric in ("psnr", "ssim", "lpips")
    }


def main() -> int:
    args = parse_args()
    infer_dir = resolve_repo(args.infer_dir)
    manifest = resolve_repo(args.manifest)
    base_path = resolve_repo(args.base_path)
    output = resolve_repo(args.output) if args.output else infer_dir / "metrics.json"
    rows = load_rows(manifest, int(args.limit))

    inference.ensure_runtime_imports(include_model=False)
    device = inference.metric_device(args.metric_device)
    lpips_metric = inference.LpipsMetric(device, args.lpips_net)
    records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    with torch.no_grad():
        for row in tqdm(rows, desc="evaluate predictions"):
            prediction_path = infer_dir / prediction_name(row, args.prediction_suffix)
            target_value = row_value(row, args.target_key, args.target_fallback_key)
            target_path = resolve_data(target_value, base_path) if target_value else None
            if not prediction_path.is_file() or target_path is None or not target_path.is_file():
                item = {
                    "manifest_index": row.get("_manifest_index"),
                    "prediction": str(prediction_path),
                    "target": None if target_path is None else str(target_path),
                }
                missing.append(item)
                if not args.allow_missing:
                    raise FileNotFoundError(item)
                continue

            with Image.open(prediction_path) as image:
                size = image.size
            prediction = load_rgb(prediction_path).to(device)
            target = transform_target(load_rgb(target_path, size=size), args.target_transform).to(device)
            records.append(
                {
                    "manifest_index": row.get("_manifest_index"),
                    "scene_id": row.get("scene_id"),
                    "light_id": row.get("light_id"),
                    "prediction": str(prediction_path),
                    "target": str(target_path),
                    "metrics": {
                        "psnr": inference.psnr(prediction, target),
                        "ssim": inference.ssim(prediction, target),
                        "lpips": lpips_metric(prediction, target),
                    },
                }
            )

    by_scene: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_scene.setdefault(str(record.get("scene_id") or ""), []).append(record)
    payload = {
        "schema_version": 1,
        "manifest": str(manifest),
        "base_path": str(base_path),
        "infer_dir": str(infer_dir),
        "prediction_suffix": args.prediction_suffix,
        "target_transform": args.target_transform,
        "metric_device": str(device),
        "lpips_net": args.lpips_net,
        "summary": {
            "expected_count": len(rows),
            "evaluated_count": len(records),
            "missing_count": len(missing),
            "metrics": average_metrics(records),
        },
        "scene_averages": {
            scene_id: {"count": len(items), "metrics": average_metrics(items)}
            for scene_id, items in sorted(by_scene.items())
        },
        "missing": missing,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)
    print(output)
    return 1 if missing and not args.allow_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
