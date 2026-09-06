#!/usr/bin/env python3
"""Compare RGB baseline/oracle/raw/refined Wan outputs by shadow-aware region."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.shadow_pipeline_io import (  # noqa: E402
    baseline_prediction_name,
    light_position,
    read_jsonl,
    sample_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="NAME=DIRECTORY; filenames follow infer_manifest.py.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--lit-margin", type=int, default=3)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--lpips-net", default="alex")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _mapping(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=DIRECTORY, got {value!r}")
        name, directory = value.split("=", 1)
        if not name or not directory or name in result:
            raise ValueError(f"Malformed or duplicate prediction mapping: {value!r}")
        result[name] = Path(directory).resolve()
    return result


def _rgb(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def _mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (shape[1], shape[0]):
        image = image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8) >= 128


def _ssim_map(pred: torch.Tensor, target: torch.Tensor, window: int = 11) -> torch.Tensor:
    channels = pred.shape[1]
    weight = torch.ones(channels, 1, window, window, device=pred.device) / float(window * window)
    padding = window // 2
    mu_x = F.conv2d(pred, weight, padding=padding, groups=channels)
    mu_y = F.conv2d(target, weight, padding=padding, groups=channels)
    sigma_x = F.conv2d(pred.square(), weight, padding=padding, groups=channels) - mu_x.square()
    sigma_y = F.conv2d(target.square(), weight, padding=padding, groups=channels) - mu_y.square()
    sigma_xy = F.conv2d(pred * target, weight, padding=padding, groups=channels) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    return ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )


class Lpips:
    def __init__(self, enabled: bool, device: torch.device, net: str) -> None:
        self.enabled = bool(enabled)
        self.model = None
        if enabled:
            import lpips

            self.model = lpips.LPIPS(net=net).to(device).eval()

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
        if self.model is None or not bool(mask.any()):
            return float("nan")
        neutral = torch.full_like(pred, 0.5)
        pred_masked = torch.where(mask, pred, neutral)
        target_masked = torch.where(mask, target, neutral)
        with torch.inference_mode():
            return float(
                self.model(pred_masked * 2.0 - 1.0, target_masked * 2.0 - 1.0)
                .mean()
                .cpu()
            )


def _region_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    region: np.ndarray,
    lpips_metric: Lpips,
) -> dict[str, float]:
    mask = torch.from_numpy(region.copy()).to(pred.device)[None, None].expand_as(pred)
    count = int(region.sum())
    if count == 0:
        result = {key: float("nan") for key in ("psnr", "ssim", "mae")}
        if lpips_metric.enabled:
            result["lpips"] = float("nan")
        result["pixels"] = 0.0
        return result
    difference = pred - target
    mse = difference.square()[mask].mean()
    mae = difference.abs()[mask].mean()
    ssim_values = _ssim_map(pred, target)[mask]
    result = {
        "psnr": float((-10.0 * torch.log10(mse.clamp_min(1e-10))).cpu()),
        "ssim": float(ssim_values.mean().cpu()),
        "mae": float(mae.cpu()),
        "pixels": float(count),
    }
    if lpips_metric.enabled:
        result["lpips"] = lpips_metric(pred, target, mask)
    return result


def _regions(row: dict[str, Any], shape: tuple[int, int], boundary: int, lit_margin: int):
    object_mask = _mask(Path(row["object_mask"]), shape)
    receiver = _mask(Path(row["receiver_mask"]), shape) & ~object_mask
    shadow = _mask(Path(row["gt_shadow_mask"]), shape) & receiver
    structure = ndimage.generate_binary_structure(2, 2)
    core = ndimage.binary_erosion(shadow, structure=structure, iterations=max(0, boundary))
    outer = ndimage.binary_dilation(shadow, structure=structure, iterations=max(0, boundary))
    edge = outer & ~core & receiver
    shadow_margin = ndimage.binary_dilation(
        shadow, structure=structure, iterations=max(0, lit_margin)
    )
    lit_receiver = receiver & ~shadow_margin
    return {
        "full": np.ones(shape, dtype=bool),
        "object": object_mask,
        "receiver": receiver,
        "shadow": shadow,
        "shadow_core": core,
        "shadow_boundary": edge,
        "lit_receiver": lit_receiver,
    }


def _finite_mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def main() -> int:
    args = parse_args()
    predictions = _mapping(args.prediction)
    rows = read_jsonl(args.manifest.resolve())
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("No rows to evaluate")
    resolved_device = (
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    device = torch.device(resolved_device)
    lpips_metric = Lpips(args.lpips, device, args.lpips_net)
    records: list[dict[str, Any]] = []

    for row in rows:
        target = _rgb(Path(row["target_image"])).to(device)
        shape = tuple(int(value) for value in target.shape[-2:])
        regions = _regions(row, shape, args.boundary_radius, args.lit_margin)
        height = round(light_position(row)[2], 6)
        filename = baseline_prediction_name(row)
        for method, directory in predictions.items():
            pred = _rgb(directory / filename).to(device)
            if pred.shape != target.shape:
                raise ValueError(f"RGB shape mismatch: {directory / filename}")
            for region_name, region in regions.items():
                records.append(
                    {
                        "sample_id": sample_key(row),
                        "scene_id": row.get("scene_id"),
                        "sample_name": row.get("sample_name"),
                        "light_height": height,
                        "method": method,
                        "region": region_name,
                        **_region_metrics(pred, target, region, lpips_metric),
                    }
                )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    height_grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["method"], record["region"])].append(record)
        height_grouped[(record["method"], record["region"], record["light_height"])].append(record)
    metric_names = ("psnr", "ssim", "mae", "lpips") if args.lpips else ("psnr", "ssim", "mae")
    summary_methods: dict[str, Any] = defaultdict(dict)
    for (method, region), values in sorted(grouped.items()):
        summary_methods[method][region] = {
            "count": len(values),
            **{name: _finite_mean([float(item[name]) for item in values]) for name in metric_names},
            "by_height": {
                str(height): {
                    "count": len(items),
                    **{
                        name: _finite_mean([float(item[name]) for item in items])
                        for name in metric_names
                    },
                }
                for (group_method, group_region, height), items in sorted(height_grouped.items())
                if group_method == method and group_region == region
            },
        }

    summary = {
        "schema": "tokenlight_shadow_conditioned_rgb_metrics_v1",
        "manifest": args.manifest.resolve().as_posix(),
        "rows": len(rows),
        "prediction_dirs": {name: path.as_posix() for name, path in predictions.items()},
        "regions": ["full", "object", "receiver", "shadow", "shadow_core", "shadow_boundary", "lit_receiver"],
        "boundary_radius": args.boundary_radius,
        "lit_margin": args.lit_margin,
        "lpips_enabled": bool(args.lpips),
        "methods": dict(summary_methods),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
