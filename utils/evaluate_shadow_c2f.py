#!/usr/bin/env python3
"""Evaluate raw, physics and refined cast-shadow masks on one manifest."""

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
from PIL import Image
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.shadow_pipeline_io import light_position, read_jsonl, sample_key  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help="NAME=ROW_KEY or NAME=ROW_KEY#NPZ_FIELD; repeat for each method.",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        help="NAME=VALUE. Unspecified methods use --default-threshold.",
    )
    parser.add_argument("--default-threshold", type=float, default=0.5)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        help="Optional held-out validation manifest used only to select thresholds.",
    )
    parser.add_argument("--calibration-limit", type=int, default=0)
    parser.add_argument("--calibration-min", type=float, default=0.05)
    parser.add_argument("--calibration-max", type=float, default=0.95)
    parser.add_argument("--calibration-steps", type=int, default=37)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--mask-to-receiver", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude-object", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _pairs(values: list[str], *, cast=str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=VALUE, got {value!r}")
        name, item = value.split("=", 1)
        if not name or not item or name in result:
            raise ValueError(f"Malformed or duplicate mapping: {value!r}")
        result[name] = cast(item)
    return result


def _gray(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    else:
        array = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    array = np.asarray(array, dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError(f"Expected 2D mask at {path}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite mask values at {path}")
    return array


def _prediction(row: dict[str, Any], spec: str) -> np.ndarray:
    key, separator, field = spec.partition("#")
    value = row.get(key)
    if not value:
        raise KeyError(f"{sample_key(row)} has no prediction path key {key!r}")
    path = Path(str(value))
    if separator:
        with np.load(path) as archive:
            if field not in archive:
                raise KeyError(f"{field!r} is absent from {path}")
            array = np.asarray(archive[field], dtype=np.float32).squeeze()
        if array.ndim != 2:
            raise ValueError(f"Expected 2D {field} in {path}, got {array.shape}")
        return array
    return _gray(path)


def _resize_probability(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if array.shape == shape:
        return array
    resized = ndimage.zoom(
        array,
        zoom=(shape[0] / array.shape[0], shape[1] / array.shape[1]),
        order=1,
        mode="nearest",
        prefilter=False,
    )
    if resized.shape != shape:
        # Rounding in scipy's zoom can differ by one pixel for unusual sizes.
        image = Image.fromarray(array.astype(np.float32), mode="F")
        resized = np.asarray(
            image.resize((shape[1], shape[0]), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
    return np.asarray(resized, dtype=np.float32)


def _support(row: dict[str, Any], shape: tuple[int, int], receiver: bool, object_: bool) -> tuple[np.ndarray, np.ndarray]:
    receiver_mask = np.ones(shape, dtype=bool)
    object_mask = np.zeros(shape, dtype=bool)
    if receiver:
        receiver_mask = _gray(Path(row["receiver_mask"])) >= 0.5
    if object_:
        object_mask = _gray(Path(row["object_mask"])) >= 0.5
    if receiver_mask.shape != shape or object_mask.shape != shape:
        raise ValueError(f"Support-mask shape mismatch for {sample_key(row)}")
    return receiver_mask, object_mask


def _confusion(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    return tp, fp, fn, tn


def _safe_ratio(numerator: float, denominator: float, *, empty_value: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float(empty_value)


def _boundary(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask)
    return np.logical_xor(mask, ndimage.binary_erosion(mask, structure=np.ones((3, 3))))


def _boundary_f(pred: np.ndarray, gt: np.ndarray, radius: int) -> float:
    pred_edge, gt_edge = _boundary(pred), _boundary(gt)
    if not pred_edge.any() and not gt_edge.any():
        return 1.0
    structure = ndimage.generate_binary_structure(2, 2)
    pred_near = ndimage.binary_dilation(pred_edge, structure=structure, iterations=max(0, radius))
    gt_near = ndimage.binary_dilation(gt_edge, structure=structure, iterations=max(0, radius))
    precision = _safe_ratio(np.logical_and(pred_edge, gt_near).sum(), pred_edge.sum(), empty_value=0.0)
    recall = _safe_ratio(np.logical_and(gt_edge, pred_near).sum(), gt_edge.sum(), empty_value=0.0)
    return _safe_ratio(2.0 * precision * recall, precision + recall, empty_value=0.0)


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return float(xs.mean()), float(ys.mean())


def _axis_angle(mask: np.ndarray) -> float | None:
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return None
    centered = np.stack((xs - xs.mean(), ys - ys.mean()), axis=1)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    values, vectors = np.linalg.eigh(covariance)
    vector = vectors[:, int(np.argmax(values))]
    return float(math.atan2(float(vector[1]), float(vector[0])))


def _angle_error(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_angle, gt_angle = _axis_angle(pred), _axis_angle(gt)
    if pred_angle is None and gt_angle is None:
        return 0.0
    if pred_angle is None or gt_angle is None:
        return 90.0
    delta = abs(math.degrees(pred_angle - gt_angle)) % 180.0
    return float(min(delta, 180.0 - delta))


def _metrics(pred: np.ndarray, gt: np.ndarray, boundary_radius: int) -> dict[str, float]:
    tp, fp, fn, tn = _confusion(pred, gt)
    both_empty = not pred.any() and not gt.any()
    iou = _safe_ratio(tp, tp + fp + fn, empty_value=1.0 if both_empty else 0.0)
    dice = _safe_ratio(2 * tp, 2 * tp + fp + fn, empty_value=1.0 if both_empty else 0.0)
    precision = _safe_ratio(tp, tp + fp, empty_value=1.0 if both_empty else 0.0)
    recall = _safe_ratio(tp, tp + fn, empty_value=1.0 if both_empty else 0.0)
    specificity = _safe_ratio(tn, tn + fp, empty_value=1.0)
    pred_center, gt_center = _centroid(pred), _centroid(gt)
    if pred_center is None and gt_center is None:
        centroid = 0.0
    elif pred_center is None or gt_center is None:
        centroid = 1.0
    else:
        diagonal = math.hypot(*pred.shape)
        centroid = math.hypot(pred_center[0] - gt_center[0], pred_center[1] - gt_center[1]) / diagonal
    gt_area = int(gt.sum())
    pred_area = int(pred.sum())
    area_ratio = float(pred_area / gt_area) if gt_area else (1.0 if pred_area == 0 else float("inf"))
    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "ber": 100.0 * (1.0 - 0.5 * (recall + specificity)),
        "boundary_f": _boundary_f(pred, gt, boundary_radius),
        "centroid_offset": centroid,
        "axis_angle_error_degrees": _angle_error(pred, gt),
        "area_ratio": area_ratio,
        "gt_area": float(gt_area),
        "pred_area": float(pred_area),
        "empty_gt": float(gt_area == 0),
    }


def _mean(records: list[dict[str, float]]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key in records[0]:
        values = [row[key] for row in records if math.isfinite(row[key])]
        result[key] = float(np.mean(values)) if values else None
    return result


def _calibrate(
    rows: list[dict[str, Any]], methods: dict[str, str], args: argparse.Namespace
) -> dict[str, float]:
    candidates = np.linspace(args.calibration_min, args.calibration_max, args.calibration_steps)
    scores = {name: np.zeros(len(candidates), dtype=np.float64) for name in methods}
    for row in rows:
        gt = _gray(Path(row["gt_shadow_mask"])) >= 0.5
        receiver, object_mask = _support(row, gt.shape, args.mask_to_receiver, args.exclude_object)
        gt &= receiver
        gt &= ~object_mask
        for name, spec in methods.items():
            probability = _resize_probability(_prediction(row, spec), gt.shape)
            for index, threshold in enumerate(candidates):
                pred = probability >= threshold
                pred &= receiver
                pred &= ~object_mask
                scores[name][index] += _metrics(pred, gt, args.boundary_radius)["dice"]
    return {
        name: float(candidates[int(np.argmax(values))])
        for name, values in scores.items()
    }


def main() -> int:
    args = parse_args()
    methods = _pairs(args.method)
    thresholds = {name: float(value) for name, value in _pairs(args.threshold, cast=float).items()}
    rows = read_jsonl(args.manifest)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("No rows to evaluate")
    unknown = set(thresholds) - set(methods)
    if unknown:
        raise KeyError(f"Thresholds supplied for unknown methods: {sorted(unknown)}")
    calibration_rows: list[dict[str, Any]] | None = None
    if args.calibrate:
        calibration_rows = (
            read_jsonl(args.calibration_manifest.resolve())
            if args.calibration_manifest
            else rows
        )
        if args.calibration_limit > 0:
            calibration_rows = calibration_rows[: args.calibration_limit]
        if not calibration_rows:
            raise RuntimeError("No rows available for threshold calibration")
        thresholds.update(_calibrate(calibration_rows, methods, args))
    for name in methods:
        thresholds.setdefault(name, float(args.default_threshold))

    records: list[dict[str, Any]] = []
    grouped: dict[str, dict[float, list[dict[str, float]]]] = {
        name: defaultdict(list) for name in methods
    }
    overall: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        gt = _gray(Path(row["gt_shadow_mask"])) >= 0.5
        receiver, object_mask = _support(row, gt.shape, args.mask_to_receiver, args.exclude_object)
        gt &= receiver
        gt &= ~object_mask
        height = round(light_position(row)[2], 6)
        for name, spec in methods.items():
            probability = _resize_probability(_prediction(row, spec), gt.shape)
            pred = probability >= thresholds[name]
            pred &= receiver
            pred &= ~object_mask
            values = _metrics(pred, gt, args.boundary_radius)
            overall[name].append(values)
            grouped[name][height].append(values)
            records.append(
                {
                    "sample_id": sample_key(row),
                    "scene_id": row.get("scene_id"),
                    "sample_name": row.get("sample_name"),
                    "light_height": height,
                    "method": name,
                    "threshold": thresholds[name],
                    **values,
                }
            )

    summary = {
        "schema": "tokenlight_shadow_c2f_metrics_v1",
        "manifest": args.manifest.resolve().as_posix(),
        "row_count": len(rows),
        "thresholds": thresholds,
        "calibrated": bool(args.calibrate),
        "calibration_manifest": (
            args.calibration_manifest.resolve().as_posix()
            if args.calibration_manifest
            else args.manifest.resolve().as_posix() if args.calibrate else None
        ),
        "calibration_row_count": len(calibration_rows) if calibration_rows is not None else 0,
        "boundary_radius": args.boundary_radius,
        "support": {
            "receiver_intersection": bool(args.mask_to_receiver),
            "object_exclusion": bool(args.exclude_object),
        },
        "methods": {
            name: {
                "overall": _mean(values),
                "by_height": {
                    str(height): {"count": len(items), **_mean(items)}
                    for height, items in sorted(grouped[name].items())
                },
            }
            for name, values in overall.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
