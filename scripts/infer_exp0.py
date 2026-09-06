#!/usr/bin/env python3
"""Infer exp_0 checkpoints and build metrics, 5-panels, and 3-panel videos."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exp_0 inference, full/object/background eval, 5-panels, and Output|GT|MSE videos."
    )
    parser.add_argument("--train-root", default="outputs/train/exp_0")
    parser.add_argument("--output-root", default="outputs/infer/exp_0")
    parser.add_argument(
        "--manifest",
        default="data_train/objaverse_fixed32_part2000_2099_png_infer/metadata.jsonl",
    )
    parser.add_argument("--base-path", default="data/objaverse_fixed32_part2000_2099_png")
    parser.add_argument("--gpu-devices", default="0")
    parser.add_argument("--checkpoint-mode", choices=("latest", "all"), default="latest")
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Infer exactly epoch-N from every selected run; overrides --checkpoint-mode.",
    )
    parser.add_argument("--run-filter", default="", help="Only process run directory names containing this text.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--gap", type=int, default=40)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate existing predictions for every selected checkpoint without inference or rendering panels/videos.",
    )
    parser.add_argument("--skip-panels", action="store_true")
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument(
        "--eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute image metrics during inference (default: enabled).",
    )
    parser.add_argument(
        "--metric-device",
        default="auto",
        help="Torch device used for PSNR/SSIM/LPIPS evaluation, e.g. cuda:5 or cpu.",
    )
    parser.add_argument(
        "--with-gt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also save the legacy source|output|GT composites (default: enabled).",
    )
    parser.add_argument(
        "--video-grouping",
        choices=("auto", "scene", "light-variant"),
        default="auto",
        help=(
            "Video grouping. 'auto' splits light variants when light_variant_id is present, "
            "otherwise it preserves one video per scene."
        ),
    )
    parser.add_argument(
        "--video-order",
        choices=("auto", "light-id", "zxy", "spatial-zigzag"),
        default="auto",
        help=(
            "Frame order. 'auto' uses z/x/y grid order when grid metadata is present, "
            "otherwise light-id order."
        ),
    )
    parser.add_argument(
        "--validate-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate complete grid/variant metadata when the manifest provides it.",
    )
    parser.add_argument(
        "--comparison-panels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After all jobs finish, create one panel containing every model output plus GT; "
            "the same panels are encoded into variant-aware videos."
        ),
    )
    return parser.parse_args()


def checkpoint_number(path: Path) -> int:
    try:
        return int(path.stem.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def discover_jobs(
    train_root: Path, mode: str, run_filter: str, epoch: int | None = None
) -> list[tuple[Path, Path]]:
    jobs: list[tuple[Path, Path]] = []
    for run_dir in sorted(path for path in train_root.iterdir() if path.is_dir()):
        if run_filter and run_filter not in run_dir.name:
            continue
        checkpoints = sorted(run_dir.glob("epoch-*.safetensors"), key=checkpoint_number)
        if not checkpoints:
            checkpoints = sorted(run_dir.glob("step-*.safetensors"), key=checkpoint_number)
        if not checkpoints:
            print(f"[skip] no checkpoints: {run_dir}", file=sys.stderr)
            continue
        if epoch is not None:
            selected = [path for path in checkpoints if path.name == f"epoch-{epoch}.safetensors"]
            if not selected:
                print(f"[skip] epoch-{epoch} is missing: {run_dir}", file=sys.stderr)
                continue
        else:
            selected = checkpoints if mode == "all" else [checkpoints[-1]]
        jobs.extend((run_dir, checkpoint) for checkpoint in selected)
    return jobs


def resolved_args(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "train_config_resolved.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    legacy = payload.get("resolved_args")
    if isinstance(legacy, dict):
        return legacy
    # Current resolved configs retain the original sectioned layout. Flatten
    # those sections so condition selection is based on the saved config rather
    # than on a run-directory naming convention.
    flattened: dict[str, Any] = {}
    for value in payload.values():
        if isinstance(value, dict):
            flattened.update(value)
    return flattened


def condition_args(run_dir: Path) -> list[str]:
    values = resolved_args(run_dir)
    mask_tokens = bool(values.get("tokenlight_mask_tokens", "shadow_mask" in run_dir.name))
    if not mask_tokens:
        return ["--no-tokenlight_mask_tokens", "--no-use-mask-input"]
    mask_key = str(values.get("tokenlight_mask_image_key") or "shadow_mask")
    extra_keys = str(values.get("tokenlight_extra_mask_image_keys") or "")
    result = [
        "--tokenlight_mask_tokens",
        "--use-mask-input",
        "--mask-key",
        mask_key,
        "--mask-fallback-key",
        mask_key,
    ]
    if extra_keys:
        result.extend(["--extra-mask-keys", extra_keys])
    return result


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("valid") is False:
                continue
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def prediction_name(row: dict[str, Any]) -> str:
    scene = str(row["scene_id"])
    light_id = row.get("light_id")
    return f"{scene}.png" if light_id is None else f"{scene}_light_{int(light_id):03d}.png"


def resolve_data(value: Any, base_path: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base_path / path


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> Image.Image:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BICUBIC)
    return image


def mse_map(pred: Image.Image, target: Image.Image) -> np.ndarray:
    pred_array = np.asarray(pred, dtype=np.float32) / 255.0
    target_array = np.asarray(target, dtype=np.float32) / 255.0
    return np.mean(np.square(pred_array - target_array), axis=-1)


def heatmap_image(values: np.ndarray, vmax: float) -> Image.Image:
    """Small dependency-free magma-like color map."""
    normalized = np.clip(values / max(vmax, 1e-12), 0.0, 1.0)
    anchors = np.asarray(
        ((0, 0, 4), (51, 15, 92), (137, 34, 106), (221, 73, 74), (252, 166, 54), (252, 253, 191)),
        dtype=np.float32,
    )
    scaled = normalized * (len(anchors) - 1)
    lower = np.floor(scaled).astype(np.int32)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    fraction = (scaled - lower)[..., None]
    rgb = anchors[lower] * (1.0 - fraction) + anchors[upper] * fraction
    return Image.fromarray(np.round(rgb).astype(np.uint8), mode="RGB")


def global_mse_vmax(rows: list[dict[str, Any]], predictions: Path, base_path: Path) -> float:
    samples: list[np.ndarray] = []
    for row in rows:
        pred_path = predictions / prediction_name(row)
        target_value = row.get("target_image") or row.get("video")
        if not pred_path.is_file() or not target_value:
            continue
        pred = load_rgb(pred_path)
        target = load_rgb(resolve_data(target_value, base_path), pred.size)
        flat = mse_map(pred, target).reshape(-1)
        stride = max(1, math.ceil(flat.size / 20000))
        samples.append(flat[::stride])
    if not samples:
        return 1.0
    value = float(np.percentile(np.concatenate(samples), 99.0))
    return value if math.isfinite(value) and value > 1e-12 else 1.0


def font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def label(image: Image.Image, text: str) -> None:
    draw = ImageDraw.Draw(image)
    text_font = font(max(20, image.height // 20))
    x, y = image.width // 2, max(8, image.height // 40)
    draw.text((x, y), text, font=text_font, anchor="ma", fill="white", stroke_width=3, stroke_fill="black")


def light_scene_image(row: dict[str, Any], size: int) -> Image.Image:
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    margin = size // 7
    center = (size // 2, size // 2)
    draw.rectangle((margin, margin, size - margin, size - margin), outline=(120, 120, 120), width=2)
    object_half = size // 12
    draw.rectangle(
        (center[0] - object_half, center[1] - object_half, center[0] + object_half, center[1] + object_half),
        fill=(216, 146, 55),
        outline=(90, 55, 15),
        width=2,
    )
    attrs = row.get("attrs_json") or {}
    if isinstance(attrs, str):
        try:
            attrs = json.loads(attrs)
        except json.JSONDecodeError:
            attrs = {}
    for index, light in enumerate(attrs.get("lights", []) if isinstance(attrs, dict) else []):
        if not isinstance(light, dict):
            continue
        try:
            x, y, z = (float(light[key]) for key in ("x", "y", "z"))
        except (KeyError, TypeError, ValueError):
            continue
        px = int(margin + (np.clip(x, -1, 1) + 1) * 0.5 * (size - margin * 2))
        py = int(size - margin - (np.clip(y, -1, 1) + 1) * 0.5 * (size - margin * 2))
        color = tuple(int(np.clip(float(light.get(key, 1.0)), 0, 1) * 255) for key in ("r", "g", "b"))
        draw.line((center[0], center[1], px, py), fill=color, width=3)
        radius = max(7, size // 35)
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color, outline="black", width=2)
        try:
            power = float(light.get("lambda"))
            light_text = f"L{index} z={z:g} p={power:g}"
        except (TypeError, ValueError):
            light_text = f"L{index} z={z:g}"
        draw.text(
            (px + radius + 3, py),
            light_text,
            fill="black",
            anchor="lm",
            font=font(max(12, size // 36)),
        )
    metadata = ["top view"]
    if row.get("light_variant_id") is not None:
        metadata.append(f"variant={int(row['light_variant_id'])}")
    if isinstance(row.get("grid_cell"), (list, tuple)) and len(row["grid_cell"]) == 3:
        metadata.append("cell=" + ",".join(str(int(value)) for value in row["grid_cell"]))
    draw.text(
        (12, size - 12),
        " | ".join(metadata),
        fill=(70, 70, 70),
        anchor="ls",
        font=font(max(12, size // 36)),
    )
    label(image, "Light scene")
    return image


def make_five_panel_frames(
    rows: list[dict[str, Any]], predictions: Path, base_path: Path, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    vmax = global_mse_vmax(rows, predictions, base_path)
    for index, row in enumerate(rows, start=1):
        pred_path = predictions / prediction_name(row)
        source_value = row.get("input_image")
        target_value = row.get("target_image") or row.get("video")
        if not pred_path.is_file() or not source_value or not target_value:
            continue
        prediction = load_rgb(pred_path)
        source = load_rgb(resolve_data(source_value, base_path), prediction.size)
        target = load_rgb(resolve_data(target_value, base_path), prediction.size)
        error = heatmap_image(mse_map(prediction, target), vmax)
        panels = [source, light_scene_image(row, prediction.width), prediction, target, error]
        for panel, text in zip(panels, ("Source", "Light scene", "Output", "GT", "MSE")):
            label(panel, text)
        canvas = Image.new("RGB", (prediction.width * 5, prediction.height), "white")
        for panel_index, panel in enumerate(panels):
            canvas.paste(panel, (panel_index * prediction.width, 0))
        canvas.save(output_dir / f"{Path(prediction_name(row)).stem}_5panel.png")
        if index == 1 or index % 100 == 0 or index == len(rows):
            print(f"[5panel] {index}/{len(rows)}", flush=True)


def make_three_panel_frames(
    rows: list[dict[str, Any]], predictions: Path, base_path: Path, output_dir: Path, gap: int
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    vmax = global_mse_vmax(rows, predictions, base_path)
    for index, row in enumerate(rows, start=1):
        pred_path = predictions / prediction_name(row)
        target_value = row.get("target_image") or row.get("video")
        if not pred_path.is_file() or not target_value:
            continue
        prediction = load_rgb(pred_path)
        target = load_rgb(resolve_data(target_value, base_path), prediction.size)
        values = np.clip(mse_map(prediction, target) / vmax, 0.0, 1.0)
        heatmap = heatmap_image(values, 1.0)
        panels = [prediction.copy(), target.copy(), heatmap]
        for panel, text in zip(panels, ("Output", "GT", "MSE")):
            label(panel, text)
        width, height = prediction.size
        canvas = Image.new("RGB", (width * 3 + gap * 2, height), "white")
        for panel_index, panel in enumerate(panels):
            canvas.paste(panel, (panel_index * (width + gap), 0))
        canvas.save(output_dir / f"{Path(prediction_name(row)).stem}_3panel.png")
        if index == 1 or index % 100 == 0 or index == len(rows):
            print(f"[3panel] {index}/{len(rows)}", flush=True)
    (output_dir / "three_panel_config.json").write_text(
        json.dumps({"layout": "Output | GT | MSE", "gap": gap, "mse_vmax_p99": vmax}, indent=2) + "\n",
        encoding="utf-8",
    )


def comparison_run_sort_key(job: tuple[Path, Path]) -> tuple[int, str]:
    name = job[0].name
    if "rgb_baseline" in name:
        rank = 0
    elif "rgb_decoder_loss" in name:
        rank = 1
    elif "rgb_shadow_mask" in name:
        rank = 2
    elif "rgb_delta_flow" in name:
        rank = 3
    else:
        rank = 10
    return rank, name


def comparison_run_label(run_name: str) -> str:
    if "rgb_baseline" in run_name:
        return "baseline"
    if "rgb_decoder_loss" in run_name:
        return "decoder loss"
    if "rgb_shadow_mask" in run_name:
        return "shadow mask"
    if "rgb_delta_flow_lambda0p2" in run_name:
        return "delta flow lambda=0.2"
    if "rgb_delta_flow" in run_name:
        return "delta flow"
    return run_name


def bottom_labeled_panel(image: Image.Image, text: str, bar_height: int = 35) -> Image.Image:
    panel = Image.new("RGB", (image.width, image.height + bar_height), "black")
    panel.paste(image, (0, 0))
    ImageDraw.Draw(panel).text(
        (panel.width // 2, image.height + bar_height // 2),
        text,
        fill=(210, 210, 210),
        anchor="mm",
        font=font(max(14, bar_height // 2)),
    )
    return panel


def make_model_comparison_frames(
    rows: list[dict[str, Any]],
    jobs: list[tuple[Path, Path]],
    output_root: Path,
    base_path: Path,
    output_dir: Path,
) -> str:
    ordered_jobs = sorted(jobs, key=comparison_run_sort_key)
    labels = [comparison_run_label(run_dir.name) for run_dir, _ in ordered_jobs]
    labels.append("gt")
    model_count = len(ordered_jobs)
    frame_suffix = f"_{model_count}outputs_gt_{model_count + 1}panel.png"
    output_dir.mkdir(parents=True, exist_ok=True)

    for row_index, row in enumerate(rows, start=1):
        frame_name = prediction_name(row)
        prediction_paths = [
            output_root / run_dir.name / checkpoint.stem / "predictions" / frame_name
            for run_dir, checkpoint in ordered_jobs
        ]
        missing = [path for path in prediction_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing comparison prediction: {missing[0]}")
        target_value = row.get("target_image") or row.get("video")
        if not target_value:
            raise ValueError(f"Missing GT path for {row.get('scene_id')}/{row.get('light_id')}")

        predictions = [load_rgb(path) for path in prediction_paths]
        size = predictions[0].size
        predictions = [
            image if image.size == size else image.resize(size, Image.Resampling.BICUBIC)
            for image in predictions
        ]
        target = load_rgb(resolve_data(target_value, base_path), size)
        labeled = [
            bottom_labeled_panel(image, panel_label)
            for image, panel_label in zip([*predictions, target], labels)
        ]
        canvas = Image.new("RGB", (size[0] * len(labeled), labeled[0].height), "black")
        for panel_index, panel in enumerate(labeled):
            canvas.paste(panel, (panel_index * size[0], 0))
        canvas.save(output_dir / f"{Path(frame_name).stem}{frame_suffix}")
        if row_index == 1 or row_index % 100 == 0 or row_index == len(rows):
            print(f"[comparison] {row_index}/{len(rows)}", flush=True)

    (output_dir / "comparison_config.json").write_text(
        json.dumps(
            {
                "layout": " | ".join(labels),
                "models": [
                    {
                        "label": label_value,
                        "run": run_dir.name,
                        "checkpoint": checkpoint.name,
                    }
                    for label_value, (run_dir, checkpoint) in zip(labels[:-1], ordered_jobs)
                ],
                "row_count": len(rows),
                "frame_suffix": frame_suffix,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return frame_suffix


def row_light(row: dict[str, Any]) -> dict[str, Any]:
    attrs = row.get("attrs_json") or {}
    if isinstance(attrs, str):
        try:
            attrs = json.loads(attrs)
        except json.JSONDecodeError:
            return {}
    lights = attrs.get("lights", []) if isinstance(attrs, dict) else []
    return lights[0] if lights and isinstance(lights[0], dict) else {}


def row_xyz(row: dict[str, Any]) -> tuple[float, float, float]:
    light = row_light(row)
    try:
        return tuple(float(light[key]) for key in ("x", "y", "z"))  # type: ignore[return-value]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Missing finite XYZ for {row.get('scene_id')}/{row.get('light_id')}"
        ) from exc


def row_power(row: dict[str, Any]) -> float | None:
    value = row.get("power_scale")
    if value is None:
        value = row_light(row).get("lambda")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def grid_cell(row: dict[str, Any]) -> tuple[int, int, int] | None:
    value = row.get("grid_cell")
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError):
        return None
    return result  # type: ignore[return-value]


def validate_grid_rows(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    has_grid = [grid_cell(row) is not None for row in rows]
    if not any(has_grid):
        return None
    if not all(has_grid):
        raise ValueError("Manifest mixes rows with and without grid_cell metadata")

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    resolutions: dict[str, tuple[int, int, int]] = {}
    variants_by_scene: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        scene = str(row["scene_id"])
        if row.get("light_variant_id") is None:
            raise ValueError(f"Grid row is missing light_variant_id: {scene}/{row.get('light_id')}")
        variant = int(row["light_variant_id"])
        resolution_value = row.get("grid_resolution")
        if not isinstance(resolution_value, (list, tuple)) or len(resolution_value) != 3:
            raise ValueError(f"Grid row is missing grid_resolution: {scene}/{row.get('light_id')}")
        resolution = tuple(int(item) for item in resolution_value)
        if any(item <= 0 for item in resolution):
            raise ValueError(f"Invalid grid_resolution {resolution} in scene {scene}")
        previous = resolutions.setdefault(scene, resolution)
        if previous != resolution:
            raise ValueError(f"Inconsistent grid_resolution in scene {scene}: {previous} vs {resolution}")
        grouped[(scene, variant)].append(row)
        variants_by_scene[scene].add(variant)

    for (scene, variant), items in sorted(grouped.items()):
        resolution = resolutions[scene]
        expected = {
            (x, y, z)
            for x in range(resolution[0])
            for y in range(resolution[1])
            for z in range(resolution[2])
        }
        cells = [grid_cell(row) for row in items]
        unique_cells = set(cells)
        duplicates = len(cells) - len(unique_cells)
        missing = sorted(expected - unique_cells)
        extra = sorted(unique_cells - expected)
        if duplicates or missing or extra:
            raise ValueError(
                f"Incomplete grid for {scene} variant {variant}: rows={len(items)}, "
                f"unique={len(unique_cells)}, duplicates={duplicates}, "
                f"missing={missing[:8]}, extra={extra[:8]}"
            )

    for scene, variants in sorted(variants_by_scene.items()):
        cell_sets = {
            variant: {grid_cell(row) for row in grouped[(scene, variant)]}
            for variant in variants
        }
        reference = next(iter(cell_sets.values()))
        if any(cells != reference for cells in cell_sets.values()):
            raise ValueError(f"Light variants do not cover identical grid cells in scene {scene}")

    random2 = any("random2" in str(row.get("light_candidate_source", "")) for row in rows)
    if random2:
        if not all("random2" in str(row.get("light_candidate_source", "")) for row in rows):
            raise ValueError("Manifest mixes random2 and non-random2 grid rows")
        for scene, variants in sorted(variants_by_scene.items()):
            if resolutions[scene] != (4, 4, 2):
                raise ValueError(
                    f"random2 scene {scene} must use grid_resolution [4, 4, 2], "
                    f"found {list(resolutions[scene])}"
                )
            if variants != {0, 1}:
                raise ValueError(
                    f"random2 scene {scene} must contain variants [0, 1], found {sorted(variants)}"
                )
            variant_powers: dict[int, float] = {}
            for variant in sorted(variants):
                powers = {row_power(row) for row in grouped[(scene, variant)]}
                powers.discard(None)
                if len(powers) != 1:
                    raise ValueError(
                        f"random2 scene {scene} variant {variant} must have one power, "
                        f"found {sorted(powers)}"
                    )
                variant_powers[variant] = next(iter(powers))
            if len(set(variant_powers.values())) != 2:
                raise ValueError(
                    f"random2 scene {scene} variants must have distinct powers, found {variant_powers}"
                )

    return {
        "scenes": len(variants_by_scene),
        "rows": len(rows),
        "sequences": len(grouped),
        "random2": random2,
        "variants_per_scene": {
            scene: sorted(variants) for scene, variants in sorted(variants_by_scene.items())
        },
        "grid_resolution": {
            scene: list(resolution) for scene, resolution in sorted(resolutions.items())
        },
    }


def spatial_zigzag_keyed_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_z: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in items:
        by_z[row_xyz(row)[2]].append(row)
    ordered: list[dict[str, Any]] = []
    for layer_index, z in enumerate(sorted(by_z)):
        by_x: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for row in by_z[z]:
            by_x[row_xyz(row)[0]].append(row)
        for column_index, x in enumerate(sorted(by_x, reverse=bool(layer_index % 2))):
            ordered.extend(
                sorted(
                    by_x[x],
                    key=lambda row: row_xyz(row)[1],
                    reverse=bool(column_index % 2),
                )
            )
    return ordered


def ordered_video_rows(items: list[dict[str, Any]], order: str) -> list[dict[str, Any]]:
    if order == "auto":
        order = "zxy" if all(grid_cell(row) is not None for row in items) else "light-id"
    if order == "light-id":
        return sorted(items, key=lambda row: int(row["light_id"]))
    if order == "zxy":
        if all(grid_cell(row) is not None for row in items):
            return sorted(
                items,
                key=lambda row: (
                    grid_cell(row)[2],
                    grid_cell(row)[0],
                    grid_cell(row)[1],
                    int(row["light_id"]),
                ),
            )
        return sorted(
            items,
            key=lambda row: (
                row_xyz(row)[2],
                row_xyz(row)[0],
                row_xyz(row)[1],
                int(row["light_id"]),
            ),
        )
    if order == "spatial-zigzag":
        return spatial_zigzag_keyed_rows(items)
    raise ValueError(f"Unsupported video order: {order}")


def float_token(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return text


def make_videos(
    rows: list[dict[str, Any]],
    frames_dir: Path,
    videos_dir: Path,
    fps: float,
    crf: int,
    *,
    grouping: str = "auto",
    order: str = "auto",
    frame_suffix: str = "_3panel.png",
    video_suffix: str = "output_gt_mse",
) -> int:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not installed")
    if grouping == "auto":
        grouping = "light-variant" if rows and all(row.get("light_variant_id") is not None for row in rows) else "scene"
    grouped: dict[tuple[str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        light_id = row.get("light_id")
        if light_id is None:
            continue
        frame = frames_dir / f"{Path(prediction_name(row)).stem}{frame_suffix}"
        if not frame.is_file():
            raise FileNotFoundError(f"Missing 3-panel frame: {frame}")
        variant: int | None = None
        if grouping == "light-variant":
            if row.get("light_variant_id") is None:
                raise ValueError(
                    f"--video-grouping light-variant requires light_variant_id: "
                    f"{row.get('scene_id')}/{light_id}"
                )
            variant = int(row["light_variant_id"])
        grouped[(str(row["scene_id"]), variant)].append(row)
    videos_dir.mkdir(parents=True, exist_ok=True)
    index_records: list[dict[str, Any]] = []
    for (scene, variant), items in sorted(grouped.items()):
        ordered_rows = ordered_video_rows(items, order)
        suffix = ""
        if variant is not None:
            suffix = f"_variant_{variant:02d}"
            powers = {row_power(row) for row in ordered_rows}
            powers.discard(None)
            if len(powers) != 1:
                raise ValueError(
                    f"Expected one power per {scene} variant {variant}, found {sorted(powers)}"
                )
            suffix += f"_lambda_{float_token(next(iter(powers)))}"
        destination = videos_dir / f"{scene}{suffix}_{video_suffix}.mp4"
        with tempfile.TemporaryDirectory(prefix=f"{scene}{suffix}_3panel_") as temp:
            stage = Path(temp)
            for index, row in enumerate(ordered_rows):
                source = frames_dir / f"{Path(prediction_name(row)).stem}{frame_suffix}"
                (stage / f"{index:06d}.png").symlink_to(source.resolve())
                xyz = row_xyz(row)
                index_records.append(
                    {
                        "video": destination.name,
                        "frame_index": index,
                        "scene_id": scene,
                        "light_variant_id": variant,
                        "light_id": int(row["light_id"]),
                        "original_light_id": row.get("original_light_id"),
                        "grid_cell": list(grid_cell(row)) if grid_cell(row) is not None else None,
                        "xyz": list(xyz),
                        "power_scale": row_power(row),
                        "frame": source.name,
                    }
                )
            run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-framerate",
                    str(fps),
                    "-i",
                    str(stage / "%06d.png"),
                    "-frames:v",
                    str(len(ordered_rows)),
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    str(crf),
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(destination),
                ]
            )
    (videos_dir / "sequence_index.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in index_records),
        encoding="utf-8",
    )
    (videos_dir / "video_summary.json").write_text(
        json.dumps(
            {
                "grouping": grouping,
                "order": order,
                "fps": fps,
                "video_count": len(grouped),
                "frame_count": len(index_records),
                "frame_suffix": frame_suffix,
                "video_suffix": video_suffix,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return len(grouped)


def process_job(args: argparse.Namespace, run_dir: Path, checkpoint: Path, rows: list[dict[str, Any]]) -> None:
    tag = checkpoint.stem
    root = repo_path(args.output_root) / run_dir.name / tag
    predictions = root / "predictions"
    panels = root / "5panel"
    three_panels = root / "3panel_frames"
    videos = root / f"videos_{args.fps:g}fps"
    common = [
        "--manifest", str(repo_path(args.manifest)),
        "--base-path", str(repo_path(args.base_path)),
        "--output-dir", str(predictions),
        "--checkpoint", str(checkpoint),
        "--height", str(args.height),
        "--width", str(args.width),
        "--num_inference_steps", str(args.num_inference_steps),
        "--cfg_scale", str(args.cfg_scale),
        "--seed", str(args.seed),
        "--gpu-devices", args.gpu_devices,
        "--tokenlight_max_lights", "2",
        "--eval-mask-key", "mask",
        "--metric-device", args.metric_device,
    ]
    common.append("--with-gt" if args.with_gt else "--no-with-gt")
    common.append("--no-skip-existing" if args.overwrite else "--skip-existing")
    if args.eval:
        common.append("--eval")
    if args.limit > 0:
        common.extend(["--limit", str(args.limit)])
    if args.eval_only:
        run([sys.executable, "scripts/infer_manifest.py", *common, "--eval-only"])
        print(f"[eval done] {run_dir.name}/{tag}: {predictions / 'metrics.json'}", flush=True)
        return
    if not args.skip_infer:
        run([sys.executable, "scripts/infer_manifest.py", *common, *condition_args(run_dir)])
    if not args.skip_panels:
        make_five_panel_frames(rows, predictions, repo_path(args.base_path), panels)
    make_three_panel_frames(rows, predictions, repo_path(args.base_path), three_panels, args.gap)
    if not args.skip_videos:
        make_videos(
            rows,
            three_panels,
            videos,
            args.fps,
            args.crf,
            grouping=args.video_grouping,
            order=args.video_order,
        )
    print(f"[done] {run_dir.name}/{tag}: {root}", flush=True)


def main() -> int:
    args = parse_args()
    if args.fps <= 0 or args.gap < 0 or not 0 <= args.crf <= 51:
        raise ValueError("fps must be positive, gap non-negative, and crf in [0, 51]")
    train_root = repo_path(args.train_root)
    if not train_root.is_dir():
        raise FileNotFoundError(train_root)
    if args.epoch is not None and args.epoch < 0:
        raise ValueError("epoch must be non-negative")
    jobs = discover_jobs(train_root, args.checkpoint_mode, args.run_filter, args.epoch)
    if not jobs:
        raise RuntimeError(f"No checkpoints found under {train_root}")
    rows = load_rows(repo_path(args.manifest), args.limit)
    if args.validate_grid and args.limit == 0:
        grid_summary = validate_grid_rows(rows)
        if grid_summary is not None:
            print(
                "grid=" + json.dumps(
                    {
                        "scenes": grid_summary["scenes"],
                        "rows": grid_summary["rows"],
                        "sequences": grid_summary["sequences"],
                        "resolution": sorted({tuple(value) for value in grid_summary["grid_resolution"].values()}),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            validation_path = repo_path(args.output_root) / "grid_validation.json"
            validation_path.parent.mkdir(parents=True, exist_ok=True)
            validation_path.write_text(
                json.dumps(grid_summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    elif args.validate_grid and args.limit > 0:
        print("[grid] skipped completeness validation because --limit is active", flush=True)
    print(f"jobs={len(jobs)} rows={len(rows)} fps={args.fps}", flush=True)
    for run_dir, checkpoint in jobs:
        process_job(args, run_dir, checkpoint, rows)
    if args.comparison_panels:
        if len(jobs) < 2:
            raise ValueError("--comparison-panels requires at least two selected model checkpoints")
        output_root = repo_path(args.output_root)
        model_count = len(jobs)
        comparison_dir = output_root / f"{model_count}models_vs_gt_{model_count + 1}panel"
        frame_suffix = make_model_comparison_frames(
            rows,
            jobs,
            output_root,
            repo_path(args.base_path),
            comparison_dir,
        )
        if not args.skip_videos:
            make_videos(
                rows,
                comparison_dir,
                output_root / f"{model_count}models_vs_gt_videos_{args.fps:g}fps",
                args.fps,
                args.crf,
                grouping=args.video_grouping,
                order=args.video_order,
                frame_suffix=frame_suffix,
                video_suffix=f"{model_count}outputs_gt",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
