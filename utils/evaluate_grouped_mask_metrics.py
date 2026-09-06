#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Iterable
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
from utils.evaluate_predictions import transform_target  # noqa: E402


DEFAULT_GROUPS = "position:0-47,color:48-79,power:80-87"
MASK_FIELDS = {
    "object_direct_lit_clean": "inf_mask",
    "object_shadow_clean": "shadow_mask",
    "object_shadow_clean_pad16": "shadow_mask_pad16",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate predictions by light-id group on the full image, object mask, and three per-sample mask regions."
        )
    )
    parser.add_argument("--infer-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--output-dir", default="", help="Default: <infer-dir>/grouped_mask_metrics")
    parser.add_argument("--prediction-suffix", default="")
    parser.add_argument("--target-key", default="target_image")
    parser.add_argument("--target-fallback-key", default="video")
    parser.add_argument("--object-mask-key", default="mask")
    parser.add_argument("--groups", default=DEFAULT_GROUPS)
    parser.add_argument("--target-transform", choices=("none", "luminance", "log_luminance"), default="none")
    parser.add_argument("--metric-device", default="auto")
    parser.add_argument("--lpips-net", default="alex")
    parser.add_argument("--no-lpips", action="store_true", help="Skip LPIPS for a faster MSE/PSNR/SSIM-only run.")
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


def parse_groups(value: str) -> list[tuple[str, int, int]]:
    groups: list[tuple[str, int, int]] = []
    names: set[str] = set()
    occupied: dict[int, str] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        try:
            name, bounds = item.split(":", 1)
            start_text, end_text = bounds.split("-", 1)
            start, end = int(start_text), int(end_text)
        except ValueError as exc:
            raise ValueError(f"Invalid group {item!r}; expected name:start-end") from exc
        name = name.strip()
        if not name or name in names or start < 0 or end < start:
            raise ValueError(f"Invalid or duplicate group {item!r}")
        for light_id in range(start, end + 1):
            if light_id in occupied:
                raise ValueError(f"Group {name!r} overlaps {occupied[light_id]!r} at light id {light_id}")
            occupied[light_id] = name
        names.add(name)
        groups.append((name, start, end))
    if not groups:
        raise ValueError("--groups must define at least one group")
    return groups


def group_name(row: dict[str, Any], groups: list[tuple[str, int, int]]) -> str:
    light_id = row.get("light_id")
    if light_id is None:
        return "ungrouped"
    value = int(light_id)
    for name, start, end in groups:
        if start <= value <= end:
            return name
    return "ungrouped"


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


def load_mask(path: Path, size: tuple[int, int]) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != size:
            image = image.resize(size, Image.Resampling.NEAREST)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy((array > 0.5).astype(np.float32)).unsqueeze(0).unsqueeze(0)


def finite_mean(values: Iterable[Any]) -> float | None:
    valid: list[float] = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            valid.append(number)
    return float(sum(valid) / len(valid)) if valid else None


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def loss_region(pixel_mse: torch.Tensor, rgb_squared_error: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    binary = mask > 0.5
    pixel_count = int(binary.sum().item())
    total_pixel_count = int(binary.numel())
    if pixel_count == 0:
        return {
            "pixel_count": 0,
            "pixel_fraction": 0.0,
            "mse": None,
            "pixel_mse_loss_sum": 0.0,
            "rgb_squared_error_sum": 0.0,
            "rgb_value_count": 0,
        }
    pixel_weights = binary.to(dtype=pixel_mse.dtype)
    rgb_weights = pixel_weights.expand_as(rgb_squared_error)
    pixel_loss_sum = float((pixel_mse * pixel_weights).sum().item())
    rgb_loss_sum = float((rgb_squared_error * rgb_weights).sum().item())
    return {
        "pixel_count": pixel_count,
        "pixel_fraction": float(pixel_count / total_pixel_count),
        "mse": float(pixel_loss_sum / pixel_count),
        "pixel_mse_loss_sum": pixel_loss_sum,
        "rgb_squared_error_sum": rgb_loss_sum,
        "rgb_value_count": pixel_count * int(rgb_squared_error.shape[1]),
    }


def add_full_loss_ratio(region: dict[str, Any], full_pixel_loss_sum: float) -> dict[str, Any]:
    output = dict(region)
    output["full_image_loss_ratio"] = safe_ratio(float(region["pixel_mse_loss_sum"]), full_pixel_loss_sum)
    return output


def image_metric_block(
    prediction: torch.Tensor,
    target: torch.Tensor,
    region: dict[str, Any],
    *,
    lpips_metric: Any | None,
) -> dict[str, Any]:
    return {
        **region,
        "psnr": inference.psnr(prediction, target),
        "ssim": inference.ssim(prediction, target),
        "lpips": None if lpips_metric is None else lpips_metric(prediction, target),
    }


def object_metric_block(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    region: dict[str, Any],
    *,
    lpips_metric: Any | None,
) -> dict[str, Any]:
    if int(region["pixel_count"]) == 0:
        return {**region, "psnr": None, "ssim": None, "lpips": None}
    object_prediction, object_target = inference.object_crop(prediction, target, mask)
    return {
        **region,
        "psnr": inference.masked_psnr(prediction, target, mask),
        "ssim": inference.ssim(object_prediction, object_target),
        "lpips": None if lpips_metric is None else lpips_metric(object_prediction, object_target),
    }


def aggregate_region(records: list[dict[str, Any]], key_path: tuple[str, ...], full_loss_sum: float) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    for record in records:
        value: Any = record
        for key in key_path:
            value = value[key]
        blocks.append(value)
    pixel_count = sum(int(block["pixel_count"]) for block in blocks)
    rgb_value_count = sum(int(block["rgb_value_count"]) for block in blocks)
    pixel_loss_sum = sum(float(block["pixel_mse_loss_sum"]) for block in blocks)
    rgb_loss_sum = sum(float(block["rgb_squared_error_sum"]) for block in blocks)
    total_pixels = sum(int(record["full_image"]["pixel_count"]) for record in records)
    return {
        "sample_count": len(blocks),
        "nonempty_sample_count": sum(int(block["pixel_count"]) > 0 for block in blocks),
        "pixel_count": pixel_count,
        "pixel_fraction": safe_ratio(pixel_count, total_pixels),
        "mse": safe_ratio(pixel_loss_sum, pixel_count),
        "pixel_mse_loss_sum": pixel_loss_sum,
        "rgb_squared_error_sum": rgb_loss_sum,
        "rgb_value_count": rgb_value_count,
        "full_image_loss_ratio": safe_ratio(pixel_loss_sum, full_loss_sum),
    }


def aggregate_image_metrics(records: list[dict[str, Any]], key: str, full_loss_sum: float) -> dict[str, Any]:
    region = aggregate_region(records, (key,), full_loss_sum)
    blocks = [record[key] for record in records]
    return {
        **region,
        "psnr_mean": finite_mean(block["psnr"] for block in blocks),
        "ssim_mean": finite_mean(block["ssim"] for block in blocks),
        "lpips_mean": finite_mean(block["lpips"] for block in blocks),
    }


def summarize(records: list[dict[str, Any]], expected_count: int, missing_count: int) -> dict[str, Any]:
    full_loss_sum = sum(float(record["full_image"]["pixel_mse_loss_sum"]) for record in records)
    if not records:
        return {
            "expected_count": expected_count,
            "evaluated_count": 0,
            "missing_count": missing_count,
            "full_image": None,
            "object_only": None,
            "mask_regions": {name: None for name in MASK_FIELDS},
        }
    return {
        "expected_count": expected_count,
        "evaluated_count": len(records),
        "missing_count": missing_count,
        "full_image": aggregate_image_metrics(records, "full_image", full_loss_sum),
        "object_only": aggregate_image_metrics(records, "object_only", full_loss_sum),
        "mask_regions": {
            name: aggregate_region(records, ("mask_regions", name), full_loss_sum) for name in MASK_FIELDS
        },
    }


def flatten_group_summary(group: str, summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "group": group,
        "expected_count": summary["expected_count"],
        "evaluated_count": summary["evaluated_count"],
        "missing_count": summary["missing_count"],
    }
    for scope in ("full_image", "object_only"):
        block = summary.get(scope) or {}
        for metric in (
            "psnr_mean",
            "ssim_mean",
            "lpips_mean",
            "mse",
            "pixel_mse_loss_sum",
            "full_image_loss_ratio",
            "pixel_count",
            "pixel_fraction",
        ):
            row[f"{scope}_{metric}"] = block.get(metric)
    for name in MASK_FIELDS:
        block = summary.get("mask_regions", {}).get(name) or {}
        for metric in ("mse", "pixel_mse_loss_sum", "full_image_loss_ratio", "pixel_count", "pixel_fraction"):
            row[f"{name}_{metric}"] = block.get(metric)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    infer_dir = resolve_repo(args.infer_dir)
    manifest = resolve_repo(args.manifest)
    base_path = resolve_repo(args.base_path)
    output_dir = resolve_repo(args.output_dir) if args.output_dir else infer_dir / "grouped_mask_metrics"
    groups = parse_groups(args.groups)
    rows = load_rows(manifest, int(args.limit))

    inference.ensure_runtime_imports(include_model=False)
    device = inference.metric_device(args.metric_device)
    lpips_metric = None if args.no_lpips else inference.LpipsMetric(device, args.lpips_net)
    records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    with torch.no_grad():
        for row in tqdm(rows, desc="grouped mask metrics"):
            prediction_path = infer_dir / prediction_name(row, args.prediction_suffix)
            target_value = row_value(row, args.target_key, args.target_fallback_key)
            object_mask_value = row.get(args.object_mask_key)
            required_values = {
                "target": target_value,
                "object_mask": object_mask_value,
                **{name: row.get(field) for name, field in MASK_FIELDS.items()},
            }
            paths = {
                name: resolve_data(value, base_path) if value else None for name, value in required_values.items()
            }
            absent = [name for name, path in paths.items() if path is None or not path.is_file()]
            if not prediction_path.is_file() or absent:
                item = {
                    "manifest_index": row.get("_manifest_index"),
                    "scene_id": row.get("scene_id"),
                    "light_id": row.get("light_id"),
                    "prediction": str(prediction_path),
                    "missing_assets": (["prediction"] if not prediction_path.is_file() else []) + absent,
                }
                missing.append(item)
                if not args.allow_missing:
                    raise FileNotFoundError(item)
                continue

            with Image.open(prediction_path) as image:
                size = image.size
            prediction = load_rgb(prediction_path).to(device)
            target = transform_target(load_rgb(paths["target"], size=size), args.target_transform).to(device)
            object_mask = load_mask(paths["object_mask"], size).to(device)
            region_masks = {name: load_mask(paths[name], size).to(device) for name in MASK_FIELDS}

            rgb_squared_error = (prediction.float() - target.float()).square()
            pixel_mse = rgb_squared_error.mean(dim=1, keepdim=True)
            full_mask = torch.ones_like(object_mask)
            full_region = loss_region(pixel_mse, rgb_squared_error, full_mask)
            full_loss_sum = float(full_region["pixel_mse_loss_sum"])
            full_region = add_full_loss_ratio(full_region, full_loss_sum)
            object_region = add_full_loss_ratio(
                loss_region(pixel_mse, rgb_squared_error, object_mask), full_loss_sum
            )
            mask_regions = {
                name: add_full_loss_ratio(loss_region(pixel_mse, rgb_squared_error, mask), full_loss_sum)
                for name, mask in region_masks.items()
            }
            records.append(
                {
                    "manifest_index": row.get("_manifest_index"),
                    "scene_id": row.get("scene_id"),
                    "light_id": row.get("light_id"),
                    "sample_name": row.get("sample_name"),
                    "task": row.get("task"),
                    "group": group_name(row, groups),
                    "prediction": str(prediction_path),
                    "target": str(paths["target"]),
                    "object_mask": str(paths["object_mask"]),
                    "full_image": image_metric_block(
                        prediction, target, full_region, lpips_metric=lpips_metric
                    ),
                    "object_only": object_metric_block(
                        prediction, target, object_mask, object_region, lpips_metric=lpips_metric
                    ),
                    "mask_regions": mask_regions,
                    "mask_paths": {name: str(paths[name]) for name in MASK_FIELDS},
                }
            )

    expected_by_group: dict[str, int] = {name: 0 for name, _, _ in groups}
    expected_by_group["ungrouped"] = 0
    for row in rows:
        expected_by_group[group_name(row, groups)] += 1
    missing_by_group: dict[str, int] = {name: 0 for name in expected_by_group}
    for item in missing:
        missing_by_group[group_name(item, groups)] += 1
    records_by_group: dict[str, list[dict[str, Any]]] = {name: [] for name in expected_by_group}
    for record in records:
        records_by_group[record["group"]].append(record)

    summaries = {
        "all": summarize(records, len(rows), len(missing)),
        **{
            name: summarize(records_by_group[name], expected_by_group[name], missing_by_group[name])
            for name in expected_by_group
            if expected_by_group[name] > 0
        },
    }
    payload = {
        "schema_version": 1,
        "definitions": {
            "pixel_mse": "mean squared RGB error per pixel in [0,1] image space",
            "pixel_mse_loss_sum": "sum of pixel_mse over the selected region",
            "full_image_loss_ratio": "region pixel_mse_loss_sum divided by full-image pixel_mse_loss_sum",
            "object_psnr": "PSNR computed strictly over object-mask RGB values",
            "object_ssim_lpips": "computed on the object bounding box with non-object pixels set to neutral gray",
            "mask_overlap": "the three mask regions are evaluated independently and may overlap; ratios need not sum to one",
        },
        "manifest": str(manifest),
        "base_path": str(base_path),
        "infer_dir": str(infer_dir),
        "prediction_suffix": args.prediction_suffix,
        "target_transform": args.target_transform,
        "metric_device": str(device),
        "lpips_net": None if args.no_lpips else args.lpips_net,
        "groups": [{"name": name, "light_id_start": start, "light_id_end": end} for name, start, end in groups],
        "summaries": summaries,
        "missing": missing,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    csv_rows = [flatten_group_summary(name, summary) for name, summary in summaries.items()]
    write_csv(output_dir / "group_summary.csv", csv_rows)

    print(json.dumps(summaries, ensure_ascii=False, indent=2, allow_nan=False))
    print(output_dir)
    return 1 if missing and not args.allow_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
