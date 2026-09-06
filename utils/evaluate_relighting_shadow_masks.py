#!/usr/bin/env python3
"""Evaluate RGB relighting predictions in renderer-defined lighting regions.

The RGB baseline evaluated by this script does not emit a shadow mask.  The
primary metrics therefore compare prediction and target RGB inside renderer
AOV masks.  A secondary, explicitly diagnostic, pseudo-shadow attenuation map
is estimated from all available lights of a scene and compared with both the
target-derived proxy and the binary renderer shadow mask.

Internal mask polarity is always 1 = shadow/occluded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


REPO_ROOT = Path(__file__).resolve().parents[1]
LUMA_WEIGHTS = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
DEFAULT_MANIFEST = "data_train/objaverse_fixed32_part2000_2099_png_infer/metadata.jsonl"
DEFAULT_BASE_PATH = "data/objaverse_fixed32_part2000_2099_png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-dir", required=True, help="Directory containing raw scene_*_light_*.png files.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--base-path", default=DEFAULT_BASE_PATH)
    parser.add_argument("--output-dir", default="", help="Default: <infer-dir>/shadow_mask_eval_seed<seed>")
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--scene-count", type=int, default=10)
    parser.add_argument("--lights-per-scene", type=int, default=10)
    parser.add_argument("--normalization-percentile", type=float, default=90.0)
    parser.add_argument("--envelope-percentile", type=float, default=90.0)
    parser.add_argument("--log-epsilon", type=float, default=0.003)
    parser.add_argument("--normalization-floor", type=float, default=0.03)
    parser.add_argument("--envelope-epsilon", type=float, default=0.05)
    parser.add_argument("--gaussian-sigma", type=float, default=0.5)
    parser.add_argument("--boundary-width", type=int, default=2)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--save-all-soft-masks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--panels-per-scene", type=int, default=1)
    return parser.parse_args()


def resolve_repo(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_rows(path: Path) -> list[dict[str, Any]]:
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
    return rows


def resolve_data(value: Any, base_path: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base_path / path


def prediction_name(row: dict[str, Any]) -> str:
    return f"{row['scene_id']}_light_{int(row['light_id']):03d}.png"


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def load_mask(path: Path, threshold: float = 0.5) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return values > threshold


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def linear_luminance(values: np.ndarray) -> np.ndarray:
    return srgb_to_linear(values) @ LUMA_WEIGHTS


def target_path(row: dict[str, Any], base_path: Path) -> Path:
    value = row.get("target_image") or row.get("video")
    if not value:
        raise KeyError(f"Manifest row {row.get('_manifest_index')} has no target_image/video")
    return resolve_data(value, base_path)


def scene_mask_path(base_path: Path, scene_id: str, filename: str) -> Path:
    return base_path / "scenes" / scene_id / "masks" / filename


def scene_domains(row: dict[str, Any], base_path: Path) -> dict[str, np.ndarray]:
    scene_id = str(row["scene_id"])
    object_mask = load_mask(resolve_data(row["mask"], base_path))
    receiver = load_mask(scene_mask_path(base_path, scene_id, "receiver_mask.png")) & ~object_mask
    floor = load_mask(scene_mask_path(base_path, scene_id, "floor_mask.png")) & receiver
    # Antialiased floor/wall AOVs overlap by a thin horizon fringe after hard
    # thresholding.  Make the normalization domains disjoint so the second
    # plane cannot silently overwrite the first plane's scale.
    wall = load_mask(scene_mask_path(base_path, scene_id, "wall_mask.png")) & receiver & ~floor
    remainder = receiver & ~(floor | wall)
    planes = [plane for plane in (floor, wall, remainder) if np.any(plane)]
    return {
        "object": object_mask,
        "receiver": receiver,
        "floor": floor,
        "wall": wall,
        "remainder": remainder,
        "planes": planes,
    }


def pseudo_shadow_stack(
    rows: list[dict[str, Any]],
    *,
    image_path_for_row: Any,
    base_path: Path,
    domains: dict[str, np.ndarray],
    normalization_percentile: float,
    envelope_percentile: float,
    log_epsilon: float,
    normalization_floor: float,
    envelope_epsilon: float,
    gaussian_sigma: float,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Build a multi-light low-gain attenuation proxy, not a physical alpha AOV.

    Source is ambient-only.  For each target light we measure positive
    log-luminance gain over source, normalize floor and wall independently, and
    use the across-light upper envelope as a pseudo unoccluded reference.
    """

    ordered = sorted(rows, key=lambda item: int(item["light_id"]))
    source = linear_luminance(load_rgb(resolve_data(ordered[0]["input_image"], base_path)))
    normalized_gains: list[np.ndarray] = []
    for row in ordered:
        luminance = linear_luminance(load_rgb(image_path_for_row(row)))
        gain = np.maximum(np.log((luminance + log_epsilon) / (source + log_epsilon)), 0.0)
        normalized = np.zeros_like(gain, dtype=np.float32)
        for plane in domains["planes"]:
            scale = float(np.percentile(gain[plane], normalization_percentile))
            normalized[plane] = gain[plane] / max(scale, normalization_floor)
        normalized_gains.append(normalized)

    gain_stack = np.stack(normalized_gains, axis=0)
    envelope = np.percentile(gain_stack, envelope_percentile, axis=0).astype(np.float32)
    attenuation = np.clip(1.0 - gain_stack / (envelope[None, ...] + envelope_epsilon), 0.0, 1.0)
    receiver_weights = domains["receiver"].astype(np.float32)
    attenuation[:, ~domains["receiver"]] = 0.0
    if gaussian_sigma > 0:
        smooth_weights = ndimage.gaussian_filter(receiver_weights, sigma=gaussian_sigma, mode="nearest")
        for index in range(len(attenuation)):
            numerator = ndimage.gaussian_filter(
                attenuation[index] * receiver_weights, sigma=gaussian_sigma, mode="nearest"
            )
            attenuation[index] = numerator / np.maximum(smooth_weights, 1e-6)
    attenuation[:, ~domains["receiver"]] = 0.0
    return {
        int(row["light_id"]): attenuation[index]
        for index, row in enumerate(ordered)
    }, envelope


def safe_divide(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def confusion_counts(
    probability: np.ndarray, target: np.ndarray, domain: np.ndarray, threshold: float
) -> dict[str, int]:
    predicted = (probability >= threshold) & domain
    positive = target & domain
    negative = ~target & domain
    return {
        "tp": int(np.count_nonzero(predicted & positive)),
        "fp": int(np.count_nonzero(predicted & negative)),
        "fn": int(np.count_nonzero(~predicted & positive)),
        "tn": int(np.count_nonzero(~predicted & negative)),
    }


def metrics_from_confusion(counts: dict[str, int]) -> dict[str, float | None]:
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    tpr = safe_divide(tp, tp + fn)
    tnr = safe_divide(tn, tn + fp)
    return {
        "iou": safe_divide(tp, tp + fp + fn),
        "dice": safe_divide(2 * tp, 2 * tp + fp + fn),
        "precision": safe_divide(tp, tp + fp),
        "recall": tpr,
        "specificity": tnr,
        "ber": None if tpr is None or tnr is None else float(0.5 * ((1.0 - tpr) + (1.0 - tnr))),
        "predicted_fraction": safe_divide(tp + fp, tp + fp + fn + tn),
        "target_fraction": safe_divide(tp + fn, tp + fp + fn + tn),
    }


def binary_metrics(
    probability: np.ndarray, target: np.ndarray, domain: np.ndarray, threshold: float
) -> dict[str, float | int | None]:
    counts = confusion_counts(probability, target, domain, threshold)
    return {**metrics_from_confusion(counts), **counts}


def soft_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    domain: np.ndarray,
    *,
    support_threshold: float = 0.05,
) -> dict[str, float | int | None]:
    p = prediction[domain].astype(np.float64)
    g = target[domain].astype(np.float64)
    if p.size == 0:
        return {
            "pixel_count": 0,
            "rmse": None,
            "mae": None,
            "soft_iou_minmax": None,
            "zncc": None,
            "scale_aligned_rmse": None,
            "scale": None,
            "wse": None,
            "shadow_support_fraction": None,
        }
    difference = p - g
    denominator = float(np.dot(p, p))
    scale = float(np.dot(p, g) / denominator) if denominator > 1e-12 else 1.0
    centered_p = p - p.mean()
    centered_g = g - g.mean()
    zncc_denominator = float(np.sqrt(np.dot(centered_p, centered_p) * np.dot(centered_g, centered_g)))
    support = g > support_threshold
    support_fraction = float(support.mean())
    if np.any(support) and np.any(~support) and support_fraction > 0:
        rmse_shadow = float(np.sqrt(np.mean(np.square(difference[support]))))
        rmse_nonshadow = float(np.sqrt(np.mean(np.square(difference[~support]))))
        wse = float(((1.0 - support_fraction) / support_fraction) * rmse_shadow + rmse_nonshadow)
    else:
        wse = None
    union = float(np.maximum(p, g).sum())
    return {
        "pixel_count": int(p.size),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "mae": float(np.mean(np.abs(difference))),
        "soft_iou_minmax": safe_divide(float(np.minimum(p, g).sum()), union),
        "zncc": safe_divide(float(np.dot(centered_p, centered_g)), zncc_denominator),
        "scale_aligned_rmse": float(np.sqrt(np.mean(np.square(scale * p - g)))),
        "scale": scale,
        "wse": wse,
        "shadow_support_fraction": support_fraction,
    }


def region_rgb_metrics(
    squared_error: np.ndarray,
    absolute_error: np.ndarray,
    mask: np.ndarray,
    full_squared_error_sum: float,
) -> dict[str, float | int | None]:
    pixel_count = int(np.count_nonzero(mask))
    if pixel_count == 0:
        return {
            "pixel_count": 0,
            "pixel_fraction": 0.0,
            "rmse": None,
            "rmse_255": None,
            "mae": None,
            "psnr": None,
            "error_mass_fraction": 0.0,
        }
    values = squared_error[mask]
    mse = float(values.mean())
    squared_error_sum = float(values.sum())
    rmse = math.sqrt(mse)
    return {
        "pixel_count": pixel_count,
        "pixel_fraction": float(pixel_count / mask.size),
        "rmse": rmse,
        "rmse_255": rmse * 255.0,
        "mae": float(absolute_error[mask].mean()),
        "psnr": float("inf") if mse <= 0 else float(-10.0 * math.log10(mse)),
        "error_mass_fraction": safe_divide(squared_error_sum, full_squared_error_sum),
    }


def rgb_region_blocks(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    domains: dict[str, np.ndarray],
    shadow: np.ndarray,
    shadow_padded: np.ndarray,
    direct_lit: np.ndarray,
    boundary_width: int,
    target_proxy: np.ndarray,
    proxy_threshold: float,
) -> dict[str, dict[str, float | int | None]]:
    squared_error = np.square(prediction - target)
    absolute_error = np.abs(prediction - target)
    full_squared_error_sum = float(squared_error.sum())
    structure = ndimage.generate_binary_structure(2, 1)
    dilated = ndimage.binary_dilation(shadow, structure=structure, iterations=boundary_width)
    eroded = ndimage.binary_erosion(shadow, structure=structure, iterations=boundary_width)
    boundary = (dilated & ~eroded) & domains["receiver"]
    proxy_transition = (
        (np.abs(target_proxy - proxy_threshold) <= 0.1) & domains["receiver"]
    )
    masks = {
        "full": np.ones(shadow.shape, dtype=bool),
        "object": domains["object"],
        "direct_lit_object": direct_lit,
        "shadow_core": shadow,
        "shadow_pad02": shadow_padded,
        "shadow_boundary_2px": boundary,
        "proxy_transition_band": proxy_transition,
        "lit_receiver": domains["receiver"] & ~shadow_padded,
        "receiver": domains["receiver"],
        "direct_or_shadow_pad02": direct_lit | shadow_padded,
        "outside_direct_or_shadow_pad02": ~(direct_lit | shadow_padded),
    }
    return {
        name: region_rgb_metrics(squared_error, absolute_error, mask, full_squared_error_sum)
        for name, mask in masks.items()
    }


def finite_mean(values: Iterable[Any]) -> float | None:
    valid: list[float] = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            valid.append(number)
    return float(np.mean(valid)) if valid else None


def flatten(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def numeric_metric_keys(records: list[dict[str, Any]]) -> list[str]:
    excluded_suffixes = ("_pixel_count", "_tp", "_fp", "_fn", "_tn", "_scale")
    excluded = {
        "manifest_index",
        "light_id",
        "original_light_id",
        "scene_sample_rank",
        "scene_light_count",
    }
    keys: list[str] = []
    for key in records[0]:
        if key in excluded or key.endswith(excluded_suffixes):
            continue
        values = [record.get(key) for record in records]
        if any(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values if value is not None):
            keys.append(key)
    return keys


def summarize_values(values: list[Any]) -> dict[str, float | int | None]:
    finite = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {"count": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "median": float(np.median(finite)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def bootstrap_scene_ci(
    scene_records: list[dict[str, Any]], metric_keys: list[str], *, seed: int, replicates: int
) -> dict[str, dict[str, float | int | None]]:
    rng = np.random.default_rng(seed)
    shared_indices = (
        rng.integers(0, len(scene_records), size=(replicates, len(scene_records)))
        if replicates > 0 and scene_records
        else None
    )
    result: dict[str, dict[str, float | int | None]] = {}
    for key in metric_keys:
        values = np.asarray([
            float(record[key])
            if record.get(key) is not None and math.isfinite(float(record[key]))
            else np.nan
            for record in scene_records
        ], dtype=np.float64)
        valid_count = int(np.count_nonzero(np.isfinite(values)))
        if valid_count == 0:
            result[key] = {"scene_count": 0, "mean": None, "ci95_low": None, "ci95_high": None}
            continue
        if shared_indices is not None:
            with np.errstate(invalid="ignore"):
                means = np.nanmean(values[shared_indices], axis=1)
            means = means[np.isfinite(means)]
            low, high = np.percentile(means, (2.5, 97.5))
        else:
            low = high = np.nanmean(values)
        result[key] = {
            "scene_count": valid_count,
            "mean": float(np.nanmean(values)),
            "ci95_low": float(low),
            "ci95_high": float(high),
        }
    return result


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(key for record in records for key in record))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def calibration_counts(probability: np.ndarray, target: np.ndarray, domain: np.ndarray, thresholds: np.ndarray) -> list[dict[str, int]]:
    positive = np.sort(probability[target & domain])
    negative = np.sort(probability[(~target) & domain])
    positive_below = np.searchsorted(positive, thresholds, side="left")
    negative_below = np.searchsorted(negative, thresholds, side="left")
    return [
        {
            "tp": int(len(positive) - positive_below[index]),
            "fn": int(positive_below[index]),
            "fp": int(len(negative) - negative_below[index]),
            "tn": int(negative_below[index]),
        }
        for index in range(len(thresholds))
    ]


def calibrate_threshold(
    scene_rows: dict[str, list[dict[str, Any]]],
    scene_ids: list[str],
    *,
    thresholds: np.ndarray,
    base_path: Path,
    proxy_kwargs: dict[str, Any],
) -> tuple[float, list[dict[str, Any]]]:
    per_scene_curves: list[list[dict[str, float | None]]] = []
    for scene_number, scene_id in enumerate(scene_ids, start=1):
        rows = scene_rows[scene_id]
        domains = scene_domains(rows[0], base_path)
        proxies, _ = pseudo_shadow_stack(
            rows,
            image_path_for_row=lambda row: target_path(row, base_path),
            base_path=base_path,
            domains=domains,
            **proxy_kwargs,
        )
        sums = [{"iou": [], "ber": []} for _ in thresholds]
        for row in rows:
            shadow = load_mask(resolve_data(row["shadow_mask"], base_path)) & domains["receiver"]
            curves = calibration_counts(proxies[int(row["light_id"])], shadow, domains["receiver"], thresholds)
            for index, counts in enumerate(curves):
                metrics = metrics_from_confusion(counts)
                sums[index]["iou"].append(metrics["iou"])
                sums[index]["ber"].append(metrics["ber"])
        per_scene_curves.append(
            [
                {"iou": finite_mean(item["iou"]), "ber": finite_mean(item["ber"])}
                for item in sums
            ]
        )
        print(f"[calibrate] {scene_number}/{len(scene_ids)} {scene_id}", flush=True)

    curve: list[dict[str, Any]] = []
    for index, threshold in enumerate(thresholds):
        scene_ious = [items[index]["iou"] for items in per_scene_curves]
        scene_bers = [items[index]["ber"] for items in per_scene_curves]
        curve.append(
            {
                "threshold": float(threshold),
                "scene_macro_iou": finite_mean(scene_ious),
                "scene_macro_ber": finite_mean(scene_bers),
            }
        )
    valid = [item for item in curve if item["scene_macro_ber"] is not None]
    if not valid:
        raise RuntimeError("Threshold calibration produced no finite BER values")
    selected = min(valid, key=lambda item: (float(item["scene_macro_ber"]), -float(item["scene_macro_iou"])))
    return float(selected["threshold"]), curve


def optional_existing_metrics(infer_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    path = infer_dir / "metrics.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for record in payload.get("records", []):
        key = (str(record.get("scene_id")), int(record.get("light_id")))
        result[key] = record
    return result


def grayscale_image(values: np.ndarray) -> Image.Image:
    return Image.fromarray(np.round(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").convert("RGB")


def add_label(image: Image.Image, text: str) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, result.width, 28), fill=(0, 0, 0))
    draw.text((8, 7), text, fill=(255, 255, 255))
    return result


def save_panel(
    path: Path,
    prediction: np.ndarray,
    target: np.ndarray,
    predicted_proxy: np.ndarray,
    target_proxy: np.ndarray,
    geometry: np.ndarray,
) -> None:
    rgb_prediction = Image.fromarray(np.round(np.clip(prediction, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
    rgb_target = Image.fromarray(np.round(np.clip(target, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
    panels = [
        add_label(rgb_prediction, "Output RGB"),
        add_label(rgb_target, "GT RGB"),
        add_label(grayscale_image(predicted_proxy), "Output pseudo attenuation"),
        add_label(grayscale_image(target_proxy), "GT pseudo attenuation"),
        add_label(grayscale_image(geometry.astype(np.float32)), "Renderer shadow AOV"),
    ]
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> int:
    args = parse_args()
    infer_dir = resolve_repo(args.infer_dir)
    manifest = resolve_repo(args.manifest)
    base_path = resolve_repo(args.base_path)
    output_dir = (
        resolve_repo(args.output_dir)
        if args.output_dir
        else infer_dir / f"shadow_mask_eval_seed{int(args.seed)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(manifest)
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_predictions: list[str] = []
    for row in rows:
        prediction = infer_dir / prediction_name(row)
        if prediction.is_file():
            by_scene[str(row["scene_id"])].append(row)
        else:
            missing_predictions.append(str(prediction))

    eligible = sorted(
        scene_id for scene_id, items in by_scene.items() if len(items) >= int(args.lights_per_scene)
    )
    if len(eligible) < int(args.scene_count):
        raise ValueError(
            f"Need {args.scene_count} scenes with >= {args.lights_per_scene} predictions; found {len(eligible)}"
        )
    rng = random.Random(int(args.seed))
    selected_scene_ids = rng.sample(eligible, int(args.scene_count))
    selected: dict[str, list[dict[str, Any]]] = {}
    for scene_id in selected_scene_ids:
        selected[scene_id] = rng.sample(
            sorted(by_scene[scene_id], key=lambda item: int(item["light_id"])),
            int(args.lights_per_scene),
        )

    calibration_scene_ids = [scene_id for scene_id in eligible if scene_id not in selected_scene_ids]
    thresholds = np.arange(
        float(args.threshold_min),
        float(args.threshold_max) + float(args.threshold_step) * 0.5,
        float(args.threshold_step),
        dtype=np.float64,
    )
    proxy_kwargs = {
        "normalization_percentile": float(args.normalization_percentile),
        "envelope_percentile": float(args.envelope_percentile),
        "log_epsilon": float(args.log_epsilon),
        "normalization_floor": float(args.normalization_floor),
        "envelope_epsilon": float(args.envelope_epsilon),
        "gaussian_sigma": float(args.gaussian_sigma),
    }
    selected_threshold, calibration_curve = calibrate_threshold(
        by_scene,
        calibration_scene_ids,
        thresholds=thresholds,
        base_path=base_path,
        proxy_kwargs=proxy_kwargs,
    )
    write_csv(output_dir / "calibration_curve.csv", calibration_curve)
    print(f"[calibrate] selected threshold={selected_threshold:.4f}", flush=True)

    existing_metrics = optional_existing_metrics(infer_dir)
    records: list[dict[str, Any]] = []
    selected_manifest: list[dict[str, Any]] = []
    mask_dir = output_dir / "soft_masks"
    panel_dir = output_dir / "panels"
    if args.save_all_soft_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)

    for scene_number, scene_id in enumerate(selected_scene_ids, start=1):
        all_scene_rows = by_scene[scene_id]
        domains = scene_domains(all_scene_rows[0], base_path)
        gt_proxies, _ = pseudo_shadow_stack(
            all_scene_rows,
            image_path_for_row=lambda row: target_path(row, base_path),
            base_path=base_path,
            domains=domains,
            **proxy_kwargs,
        )
        pred_proxies, _ = pseudo_shadow_stack(
            all_scene_rows,
            image_path_for_row=lambda row: infer_dir / prediction_name(row),
            base_path=base_path,
            domains=domains,
            **proxy_kwargs,
        )
        for rank, row in enumerate(selected[scene_id], start=1):
            light_id = int(row["light_id"])
            stem = Path(prediction_name(row)).stem
            prediction_path = infer_dir / prediction_name(row)
            prediction = load_rgb(prediction_path)
            target = load_rgb(target_path(row, base_path))
            shadow = load_mask(resolve_data(row["shadow_mask"], base_path)) & domains["receiver"]
            shadow_padded = load_mask(resolve_data(row["shadow_mask_pad16"], base_path)) & domains["receiver"]
            direct_lit = load_mask(resolve_data(row["inf_mask"], base_path)) & domains["object"]
            gt_proxy = gt_proxies[light_id]
            pred_proxy = pred_proxies[light_id]

            geometry_prediction = binary_metrics(pred_proxy, shadow, domains["receiver"], selected_threshold)
            geometry_reference = binary_metrics(gt_proxy, shadow, domains["receiver"], selected_threshold)
            proxy_binary = binary_metrics(
                pred_proxy,
                gt_proxy >= selected_threshold,
                domains["receiver"],
                selected_threshold,
            )
            proxy_soft = soft_metrics(
                pred_proxy,
                gt_proxy,
                domains["receiver"],
                support_threshold=selected_threshold,
            )
            geometry_soft = soft_metrics(pred_proxy, shadow.astype(np.float32), domains["receiver"])
            rgb_blocks = rgb_region_blocks(
                prediction,
                target,
                domains=domains,
                shadow=shadow,
                shadow_padded=shadow_padded,
                direct_lit=direct_lit,
                boundary_width=int(args.boundary_width),
                target_proxy=gt_proxy,
                proxy_threshold=selected_threshold,
            )

            record: dict[str, Any] = {
                "scene_id": scene_id,
                "light_id": light_id,
                "original_light_id": row.get("original_light_id"),
                "manifest_index": int(row["_manifest_index"]),
                "scene_sample_rank": rank,
                "scene_light_count": len(all_scene_rows),
                "prediction": str(prediction_path),
                "target": str(target_path(row, base_path)),
                "source": str(resolve_data(row["input_image"], base_path)),
                "renderer_shadow_mask": str(resolve_data(row["shadow_mask"], base_path)),
                "renderer_direct_lit_mask": str(resolve_data(row["inf_mask"], base_path)),
                "pseudo_threshold": selected_threshold,
                **flatten("geometry_pred", geometry_prediction),
                **flatten("geometry_gt_proxy_reference", geometry_reference),
                **flatten("proxy_binary_pred_vs_gt", proxy_binary),
                **flatten("proxy_soft_pred_vs_gt", proxy_soft),
                **flatten("geometry_soft_pred", geometry_soft),
            }
            for region_name, block in rgb_blocks.items():
                record.update(flatten(f"rgb_{region_name}", block))
            existing = existing_metrics.get((scene_id, light_id))
            if existing:
                for region_name in ("full_image", "object_only", "background_only"):
                    for metric_name, value in existing.get(region_name, {}).items():
                        record[f"existing_{region_name}_{metric_name}"] = value
            records.append(record)

            selected_manifest.append(
                {
                    **{key: value for key, value in row.items() if key != "_manifest_index"},
                    "selection_seed": int(args.seed),
                    "selection_scene_rank": scene_number,
                    "selection_light_rank": rank,
                    "prediction": str(prediction_path),
                }
            )
            if args.save_all_soft_masks:
                grayscale_image(pred_proxy).save(mask_dir / f"{stem}_pred_proxy.png")
                grayscale_image(gt_proxy).save(mask_dir / f"{stem}_gt_proxy.png")
                grayscale_image(shadow.astype(np.float32)).save(mask_dir / f"{stem}_geometry.png")
            if rank <= int(args.panels_per_scene):
                save_panel(
                    panel_dir / f"{stem}_panel.png",
                    prediction,
                    target,
                    pred_proxy,
                    gt_proxy,
                    shadow,
                )
        print(f"[evaluate] {scene_number}/{len(selected_scene_ids)} {scene_id}", flush=True)

    with (output_dir / "selected_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected_manifest:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    write_csv(output_dir / "per_sample.csv", records)

    metric_keys = numeric_metric_keys(records)
    pair_macro = {key: summarize_values([record.get(key) for record in records]) for key in metric_keys}
    scene_records: list[dict[str, Any]] = []
    for scene_id in selected_scene_ids:
        items = [record for record in records if record["scene_id"] == scene_id]
        scene_record: dict[str, Any] = {"scene_id": scene_id, "sample_count": len(items)}
        for key in metric_keys:
            scene_record[key] = finite_mean(record.get(key) for record in items)
        scene_records.append(scene_record)
    write_csv(output_dir / "per_scene.csv", scene_records)

    scene_macro = {key: summarize_values([record.get(key) for record in scene_records]) for key in metric_keys}
    headline_keys = [
        "geometry_pred_iou",
        "geometry_pred_dice",
        "geometry_pred_ber",
        "geometry_gt_proxy_reference_iou",
        "geometry_gt_proxy_reference_ber",
        "proxy_binary_pred_vs_gt_iou",
        "proxy_binary_pred_vs_gt_ber",
        "proxy_soft_pred_vs_gt_rmse",
        "proxy_soft_pred_vs_gt_mae",
        "proxy_soft_pred_vs_gt_soft_iou_minmax",
        "rgb_full_rmse",
        "rgb_object_rmse",
        "rgb_direct_lit_object_rmse",
        "rgb_shadow_core_rmse",
        "rgb_shadow_pad02_rmse",
        "rgb_shadow_boundary_2px_rmse",
        "rgb_lit_receiver_rmse",
        "rgb_direct_or_shadow_pad02_error_mass_fraction",
        "existing_full_image_psnr",
        "existing_full_image_ssim",
        "existing_full_image_lpips",
    ]
    headline_keys = [key for key in headline_keys if key in metric_keys]
    bootstrap = bootstrap_scene_ci(
        scene_records,
        headline_keys,
        seed=int(args.seed) + 1,
        replicates=int(args.bootstrap_replicates),
    )
    selected_lights = {
        scene_id: sorted(int(row["light_id"]) for row in selected[scene_id])
        for scene_id in selected_scene_ids
    }
    payload = {
        "schema_version": 1,
        "evaluation_kind": "rgb_aov_primary_plus_multilight_pseudo_shadow_diagnostic",
        "mask_polarity": "1=shadow_or_occluded",
        "paths": {
            "infer_dir": str(infer_dir),
            "manifest": str(manifest),
            "base_path": str(base_path),
            "output_dir": str(output_dir),
        },
        "selection": {
            "seed": int(args.seed),
            "scene_count": len(selected_scene_ids),
            "lights_per_scene": int(args.lights_per_scene),
            "pair_count": len(records),
            "eligible_scene_count": len(eligible),
            "selected_scene_ids_in_draw_order": selected_scene_ids,
            "selected_lights": selected_lights,
            "missing_prediction_count": len(missing_predictions),
        },
        "pseudo_mask": {
            "name": "multi-light low-positive-gain attenuation proxy",
            "warning": (
                "Diagnostic only: the RGB baseline has no explicit mask output, and this proxy may mix cast shadow, "
                "attached shading, material response, and generation error."
            ),
            "linear_luminance": True,
            "formula": (
                "g=max(log((Y_image+eps)/(Y_ambient_source+eps)),0); normalize g per floor/wall and light; "
                "U=across-light percentile(g); A=clip(1-g/(U+eps_U),0,1)"
            ),
            **proxy_kwargs,
            "threshold_selection": "minimum scene-macro BER on disjoint calibration scenes",
            "selected_threshold": selected_threshold,
            "calibration_scene_count": len(calibration_scene_ids),
            "calibration_scene_ids": calibration_scene_ids,
            "soft_iou_definition": "sum(min(pred,target))/sum(max(pred,target)) within receiver",
            "soft_mask_png_note": (
                "soft_masks/*.png are 8-bit visualizations only; exact metrics use in-memory float32 proxies and "
                "are reproducible by rerunning this script."
            ),
            "soft_support_threshold_for_wse": selected_threshold,
            "wse_note": (
                "Uses the FSD weighting form, but support is the disjoint-calibrated proxy threshold rather than "
                "FSD's physical-alpha threshold 0.05. It is retained in detailed tables but excluded from "
                "headline metrics because small proxy support makes per-sample WSE unstable."
            ),
        },
        "rgb_metric_definition": {
            "value_space": "sRGB normalized to [0,1]",
            "rmse": "sqrt(mean((prediction-target)^2)) over RGB values in the region",
            "mae": "mean(abs(prediction-target)) over RGB values in the region",
            "psnr": "-10*log10(region RGB MSE), data_range=1",
            "boundary": f"dilate(renderer_shadow,{int(args.boundary_width)}px) minus erode(renderer_shadow,{int(args.boundary_width)}px)",
            "proxy_transition_band": "abs(target_proxy-selected_threshold) <= 0.1; diagnostic, not physical penumbra",
            "shadow_pad02_note": "Manifest field is named shadow_mask_pad16, but these data paths are the renderer's 2px dilation.",
        },
        "aggregation": {
            "pair_macro": "mean/std/median/min/max across selected scene-light pairs",
            "scene_macro": "average lights within each scene, then summarize equally across scenes",
            "bootstrap": f"{int(args.bootstrap_replicates)} scene-level resamples; percentile 95% CI",
        },
        "headline_scene_macro_bootstrap95": bootstrap,
        "pair_macro": pair_macro,
        "scene_macro": scene_macro,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["headline_scene_macro_bootstrap95"], indent=2), flush=True)
    print(output_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
