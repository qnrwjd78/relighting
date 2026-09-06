#!/usr/bin/env python3
"""Run selected lighting conditions through multiple inference seeds."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.infer_exp0 import condition_args, discover_jobs, repo_path  # noqa: E402


def comma_ints(value: str, label: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError(f"{label} cannot be empty")
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError(f"{label} cannot contain duplicates")
    return result


def scene_name(value: str) -> str:
    text = value.strip()
    if text.startswith("scene_"):
        return text
    try:
        number = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid scene id: {value!r}") from exc
    # Short ids refer to offsets inside the part2000_2099 evaluation split.
    if 0 <= number < 100:
        number += 2000
    return f"scene_{number:06d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer 5 positions from scenes 28 and 30 over ten seeds."
    )
    parser.add_argument("--train-root", default="outputs/train/exp_0")
    parser.add_argument("--output-root", default="outputs/infer/seed_sweep")
    parser.add_argument(
        "--manifest",
        default="data_train/objaverse_fixed32_part2000_2099_png_infer/metadata.jsonl",
    )
    parser.add_argument("--base-path", default="data/objaverse_fixed32_part2000_2099_png")
    parser.add_argument("--scene-ids", default="28,30")
    parser.add_argument("--light-ids", default="0,7,14,21,27")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument(
        "--metrics-selection",
        default="",
        help="Select exact scene/light pairs from the lowest background PSNR records in metrics.json.",
    )
    parser.add_argument("--lowest-background-count", type=int, default=5)
    parser.add_argument(
        "--exclude-scenes",
        default="",
        help="Comma-separated scene ids to exclude from metrics-based selection.",
    )
    parser.add_argument(
        "--exclude-pairs",
        default="",
        help="Comma-separated scene:light pairs to exclude, e.g. scene_002045:11.",
    )
    parser.add_argument("--gpu-devices", default="7")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--checkpoint-mode", choices=("latest", "all"), default="latest")
    parser.add_argument("--run-filter", default="")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--eval", action="store_true", help="Also compute PSNR/SSIM/LPIPS.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-panels", action="store_true")
    parser.add_argument(
        "--panels-only",
        action="store_true",
        help="Build comparison panels from existing predictions without loading a model.",
    )
    return parser.parse_args()


def load_selected_rows(
    manifest: Path, scenes: list[str], light_ids: list[int], seeds: list[int]
) -> list[dict[str, Any]]:
    return load_selected_pairs(
        manifest,
        [(scene, light_id, None) for scene in scenes for light_id in light_ids],
        seeds,
    )


def load_selected_pairs(
    manifest: Path,
    pairs: list[tuple[str, int, float | None]],
    seeds: list[int],
) -> list[dict[str, Any]]:
    wanted = {(scene, light_id) for scene, light_id, _ in pairs}
    selection_scores = {(scene, light_id): score for scene, light_id, score in pairs}
    found: dict[tuple[str, int], dict[str, Any]] = {}
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("valid") is False or row.get("task") != "position":
                continue
            key = (str(row.get("scene_id")), int(row.get("light_id", -1)))
            if key in wanted:
                found[key] = row

    missing = sorted(wanted - set(found))
    if missing:
        raise ValueError(f"Missing scene/light conditions in manifest: {missing}")

    expanded: list[dict[str, Any]] = []
    for scene, light_id, _ in pairs:
        for seed in seeds:
            row = dict(found[(scene, light_id)])
            row["_seed"] = seed
            score = selection_scores[(scene, light_id)]
            if score is not None:
                row["_selection_background_psnr"] = score
            row["_prediction_name"] = (
                f"{scene}_position_{light_id:03d}_seed_{seed:04d}.png"
            )
            expanded.append(row)
    return expanded


def lowest_background_pairs(
    metrics_path: Path,
    count: int,
    excluded_scenes: set[str] | None = None,
    excluded_pairs: set[tuple[str, int]] | None = None,
) -> list[tuple[str, int, float]]:
    if count <= 0:
        raise ValueError("lowest-background-count must be positive")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    records = payload.get("records", []) if isinstance(payload, dict) else []
    excluded_scenes = excluded_scenes or set()
    excluded_pairs = excluded_pairs or set()
    ranked: list[tuple[float, int, str, int]] = []
    for record in records:
        try:
            score = float(record["background_only"]["psnr"])
            index = int(record.get("index", 0))
            scene = str(record["scene_id"])
            light_id = int(record["light_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if scene in excluded_scenes or (scene, light_id) in excluded_pairs:
            continue
        if math.isfinite(score):
            ranked.append((score, index, scene, light_id))
    ranked.sort()
    if len(ranked) < count:
        raise ValueError(f"Only {len(ranked)} finite background PSNR records found")
    return [(scene, light_id, score) for score, _, scene, light_id in ranked[:count]]


def parse_excluded_pairs(value: str) -> set[tuple[str, int]]:
    pairs: set[tuple[str, int]] = set()
    for item in (part.strip() for part in value.split(",")):
        if not item:
            continue
        try:
            raw_scene, raw_light = item.rsplit(":", 1)
            pairs.add((scene_name(raw_scene), int(raw_light)))
        except (ValueError, argparse.ArgumentTypeError) as exc:
            raise ValueError(f"Invalid excluded pair {item!r}; expected scene:light") from exc
    return pairs


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_font(panel_height: int) -> ImageFont.ImageFont:
    size = max(18, panel_height // 20)
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def paste_labeled(
    canvas: Image.Image,
    image: Image.Image,
    xy: tuple[int, int],
    label: str,
    font: ImageFont.ImageFont,
) -> None:
    x, y = xy
    canvas.paste(image, xy)
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    margin = max(6, image.height // 80)
    draw.rectangle(
        (x, y, x + text_width + margin * 2, y + text_height + margin * 2),
        fill=(0, 0, 0),
    )
    draw.text(
        (x + margin - bbox[0], y + margin - bbox[1]),
        label,
        fill=(255, 255, 255),
        font=font,
    )


def mse_values(pred: Image.Image, target: Image.Image) -> np.ndarray:
    pred_array = np.asarray(pred, dtype=np.float32) / 255.0
    target_array = np.asarray(target, dtype=np.float32) / 255.0
    return np.mean(np.square(pred_array - target_array), axis=-1)


def mse_heatmap(values: np.ndarray, vmax: float) -> Image.Image:
    normalized = np.clip(values / max(vmax, 1e-12), 0.0, 1.0)
    anchors = np.asarray(
        (
            (0, 0, 4),
            (51, 15, 92),
            (137, 34, 106),
            (221, 73, 74),
            (252, 166, 54),
            (252, 253, 191),
        ),
        dtype=np.float32,
    )
    scaled = normalized * (len(anchors) - 1)
    lower = np.floor(scaled).astype(np.int32)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    fraction = (scaled - lower)[..., None]
    rgb = anchors[lower] * (1.0 - fraction) + anchors[upper] * fraction
    return Image.fromarray(np.round(rgb).astype(np.uint8), mode="RGB")


def make_seed_comparisons(
    rows: list[dict[str, Any]],
    predictions: Path,
    destination: Path,
    mse_destination: Path,
    base_path: Path,
    seeds: list[int],
) -> None:
    if len(seeds) != 10:
        raise ValueError("The 5x2 comparison layout requires exactly 10 seeds")
    conditions: dict[tuple[str, int], dict[int, dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["scene_id"]), int(row["light_id"]))
        conditions.setdefault(key, {})[int(row["_seed"])] = row

    destination.mkdir(parents=True, exist_ok=True)
    mse_destination.mkdir(parents=True, exist_ok=True)
    gap = 8
    for (scene, light_id), seed_rows in conditions.items():
        missing = [seed for seed in seeds if seed not in seed_rows]
        if missing:
            raise ValueError(f"Missing seeds for {scene}/position-{light_id}: {missing}")
        first_pred = predictions / str(seed_rows[seeds[0]]["_prediction_name"])
        if not first_pred.is_file():
            raise FileNotFoundError(first_pred)
        with Image.open(first_pred) as opened:
            panel_size = opened.size
        panel_width, panel_height = panel_size
        grid_width = panel_width * 5 + gap * 4
        grid_height = panel_height * 2 + gap
        canvas_width = grid_width + gap * 3 + panel_width
        canvas = Image.new("RGB", (canvas_width, grid_height), (255, 255, 255))
        mse_canvas = Image.new("RGB", (canvas_width, grid_height), (255, 255, 255))
        font = load_font(panel_height)

        reference = seed_rows[seeds[0]]
        target_value = reference.get("target_image") or reference.get("video")
        if not target_value:
            raise KeyError(f"Missing GT for {scene}/position-{light_id}")
        target_path = Path(str(target_value))
        if not target_path.is_absolute():
            target_path = base_path / target_path
        with Image.open(target_path) as opened:
            target = opened.convert("RGB").resize(panel_size, Image.Resampling.BICUBIC)

        predictions_by_seed: dict[int, Image.Image] = {}
        mse_by_seed: dict[int, np.ndarray] = {}
        for seed in seeds:
            pred_path = predictions / str(seed_rows[seed]["_prediction_name"])
            if not pred_path.is_file():
                raise FileNotFoundError(pred_path)
            with Image.open(pred_path) as opened:
                pred = opened.convert("RGB").resize(panel_size, Image.Resampling.BICUBIC)
            predictions_by_seed[seed] = pred
            mse_by_seed[seed] = mse_values(pred, target)
        sampled = [
            values.reshape(-1)[:: max(1, values.size // 20000)]
            for values in mse_by_seed.values()
        ]
        vmax = float(np.percentile(np.concatenate(sampled), 99.0))
        if not np.isfinite(vmax) or vmax <= 1e-12:
            vmax = 1.0

        for index, seed in enumerate(seeds):
            column = index % 5
            row_index = index // 5
            xy = (column * (panel_width + gap), row_index * (panel_height + gap))
            paste_labeled(
                canvas,
                predictions_by_seed[seed],
                xy,
                f"Seed {seed}",
                font,
            )
            values = mse_by_seed[seed]
            paste_labeled(
                mse_canvas,
                mse_heatmap(values, vmax),
                xy,
                f"Seed {seed} | MSE {float(values.mean()):.5f}",
                font,
            )
        gt_x = grid_width + gap * 3
        gt_y = (grid_height - panel_height) // 2
        paste_labeled(canvas, target, (gt_x, gt_y), "GT", font)
        paste_labeled(mse_canvas, target, (gt_x, gt_y), "GT", font)

        canvas.save(destination / f"{scene}_position_{light_id:03d}_seeds_vs_gt.png")
        mse_canvas.save(
            mse_destination / f"{scene}_position_{light_id:03d}_mse_vs_gt.png"
        )


def main() -> int:
    args = parse_args()
    seeds = comma_ints(args.seeds, "seeds")
    if args.epoch is not None and args.epoch < 0:
        raise ValueError("epoch must be non-negative")

    jobs = discover_jobs(
        repo_path(args.train_root), args.checkpoint_mode, args.run_filter, args.epoch
    )
    if not jobs:
        raise RuntimeError("No matching checkpoints found")

    if args.metrics_selection:
        excluded_scenes = {
            scene_name(item) for item in args.exclude_scenes.split(",") if item.strip()
        }
        excluded_pairs = parse_excluded_pairs(args.exclude_pairs)
        pairs = lowest_background_pairs(
            repo_path(args.metrics_selection),
            args.lowest_background_count,
            excluded_scenes,
            excluded_pairs,
        )
        rows = load_selected_pairs(repo_path(args.manifest), pairs, seeds)
        selection_description = [
            f"{scene}/light-{light_id}:background_psnr={score:.6f}"
            for scene, light_id, score in pairs
        ]
    else:
        scenes = [scene_name(item) for item in args.scene_ids.split(",") if item.strip()]
        light_ids = comma_ints(args.light_ids, "light ids")
        rows = load_selected_rows(repo_path(args.manifest), scenes, light_ids, seeds)
        selection_description = [
            f"{scene}/light-{light_id}" for scene in scenes for light_id in light_ids
        ]
    output_root = repo_path(args.output_root)
    sweep_manifest = output_root / "selection_manifest.jsonl"
    write_manifest(sweep_manifest, rows)
    print(
        f"jobs={len(jobs)} selection={selection_description} seeds={seeds} "
        f"samples_per_model={len(rows)}",
        flush=True,
    )

    for run_dir, checkpoint in jobs:
        output_dir = output_root / run_dir.name / checkpoint.stem / "predictions"
        command = [
            sys.executable,
            "scripts/infer_manifest.py",
            "--manifest",
            str(sweep_manifest),
            "--base-path",
            str(repo_path(args.base_path)),
            "--output-dir",
            str(output_dir),
            "--checkpoint",
            str(checkpoint),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--num_inference_steps",
            str(args.num_inference_steps),
            "--cfg_scale",
            str(args.cfg_scale),
            "--gpu-devices",
            args.gpu_devices,
            "--tokenlight_max_lights",
            "2",
            "--seed-key",
            "_seed",
            "--prediction-name-key",
            "_prediction_name",
            "--eval-mask-key",
            "mask",
        ]
        command.append("--no-skip-existing" if args.overwrite else "--skip-existing")
        if args.eval:
            command.append("--eval")
        command.extend(condition_args(run_dir))
        if not args.panels_only:
            print("\n$ " + " ".join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
        if not args.skip_panels:
            make_seed_comparisons(
                rows,
                output_dir,
                output_dir.parent / "seed_comparison_5x2_gt",
                output_dir.parent / "mse_comparison_5x2_gt",
                repo_path(args.base_path),
                seeds,
            )

    print(f"[done] results: {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
