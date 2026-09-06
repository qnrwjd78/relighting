#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create source/light-scene/prediction/GT/error 5-panels directly from inference outputs."
    )
    parser.add_argument("--infer-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prediction-suffix", default="")
    parser.add_argument("--source-key", default="input_image")
    parser.add_argument("--target-key", default="target_image")
    parser.add_argument("--target-fallback-key", default="video")
    parser.add_argument("--attrs-key", default="attrs_json")
    parser.add_argument("--target-transform", choices=("none", "luminance", "log_luminance"), default="none")
    parser.add_argument("--heatmap-scale", choices=("per_image", "global"), default="global")
    parser.add_argument("--heatmap-percentile", type=float, default=99.0)
    parser.add_argument("--heatmap-vmax", type=float, default=0.0)
    parser.add_argument("--heatmap-samples-per-image", type=int, default=20000)
    parser.add_argument("--caption-height", type=int, default=80)
    parser.add_argument("--caption-font-size", type=int, default=28)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
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


def output_name(row: dict[str, Any], suffix: str) -> str:
    return f"{Path(prediction_name(row, suffix)).stem}_5panel.png"


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> Image.Image:
    with Image.open(path) as image:
        output = image.convert("RGB")
    if size is not None and output.size != size:
        output = output.resize(size, Image.Resampling.BICUBIC)
    return output


def image_to_unit(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0


def transform_image(image: Image.Image, transform: str) -> Image.Image:
    if transform == "none":
        return image.convert("RGB")
    rgb = image_to_unit(image.convert("RGB"))
    luminance = np.sum(rgb * np.array((0.2126, 0.7152, 0.0722), dtype=np.float32), axis=-1)
    if transform == "log_luminance":
        eps = 1e-3
        log_eps = math.log(eps)
        luminance = (np.log(np.clip(luminance, eps, 1.0)) - log_eps) / -log_eps
    output = np.repeat(np.clip(luminance, 0.0, 1.0)[..., None], 3, axis=-1)
    return Image.fromarray(np.round(output * 255.0).astype(np.uint8), mode="RGB")


def mse_map(prediction: Image.Image, target: Image.Image) -> np.ndarray:
    return np.mean(np.square(image_to_unit(prediction) - image_to_unit(target)), axis=-1)


def robust_vmax(values: np.ndarray, percentile: float) -> float:
    value = float(np.percentile(values, percentile)) if values.size else 0.0
    if not math.isfinite(value) or value <= 1e-12:
        value = float(values.max()) if values.size else 1.0
    return value if math.isfinite(value) and value > 1e-12 else 1.0


def heatmap_image(values: np.ndarray, vmax: float) -> Image.Image:
    normalized = np.clip(values / max(vmax, 1e-12), 0.0, 1.0)
    rgb = (plt.get_cmap("magma")(normalized)[..., :3] * 255.0).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def parse_attrs(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return {}


def light_scene_image(row: dict[str, Any], attrs_key: str, size: int) -> Image.Image:
    attrs = parse_attrs(row.get(attrs_key) or row.get("attrs"))
    lights = [light for light in attrs.get("lights", []) if isinstance(light, dict)]
    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
    axis = fig.add_subplot(111, projection="3d")

    bounds = (-1.0, 1.0)
    edges = (
        ((-1, -1, -1), (1, -1, -1)), ((-1, 1, -1), (1, 1, -1)),
        ((-1, -1, 1), (1, -1, 1)), ((-1, 1, 1), (1, 1, 1)),
        ((-1, -1, -1), (-1, 1, -1)), ((1, -1, -1), (1, 1, -1)),
        ((-1, -1, 1), (-1, 1, 1)), ((1, -1, 1), (1, 1, 1)),
        ((-1, -1, -1), (-1, -1, 1)), ((1, -1, -1), (1, -1, 1)),
        ((-1, 1, -1), (-1, 1, 1)), ((1, 1, -1), (1, 1, 1)),
    )
    for start, end in edges:
        axis.plot(*zip(start, end), color="#777777", linewidth=0.7, alpha=0.55)
    axis.scatter((0,), (0,), (0,), s=900, marker="s", color="#d89237", alpha=0.60, edgecolors="#6b4518")

    for index, light in enumerate(lights):
        try:
            x, y, z = (float(light[key]) for key in ("x", "y", "z"))
        except (KeyError, TypeError, ValueError):
            continue
        color = np.clip([float(light.get(key, 1.0)) for key in ("r", "g", "b")], 0.0, 1.0)
        axis.scatter((x,), (y,), (z,), s=150, color=color, edgecolors="black", linewidths=0.8)
        axis.plot((0, x), (0, y), (0, z), color=color, linewidth=1.2, alpha=0.8)
        axis.text(x, y, z, f" L{index}", fontsize=8)

    ambient = attrs.get("a")
    task = str(row.get("task") or "")
    axis.text2D(0.03, 0.96, f"{task or 'lighting'}\nambient={ambient if ambient is not None else 'n/a'}", transform=axis.transAxes, va="top", fontsize=9)
    axis.set(xlim=bounds, ylim=bounds, zlim=bounds)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xticks((-1, 0, 1))
    axis.set_yticks((-1, 0, 1))
    axis.set_zticks((-1, 0, 1))
    axis.tick_params(labelsize=7, pad=-2)
    axis.view_init(elev=22, azim=-72)
    axis.set_proj_type("ortho")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def load_font(size: int) -> ImageFont.ImageFont:
    for value in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(value).is_file():
            return ImageFont.truetype(value, size=size)
    return ImageFont.load_default()


def compose(panels: list[Image.Image], output: Path, caption_height: int, font_size: int) -> None:
    tile = panels[0].width
    labels = ("source", "light scene", "prediction", "GT", "error MSE heatmap")
    canvas = Image.new("RGB", (tile * 5, tile + caption_height), (12, 12, 12))
    for index, panel in enumerate(panels):
        canvas.paste(panel.resize((tile, tile), Image.Resampling.BICUBIC), (index * tile, 0))
    draw = ImageDraw.Draw(canvas)
    font = load_font(font_size)
    for index, label in enumerate(labels):
        draw.text(
            (index * tile + tile / 2, tile + caption_height / 2),
            label,
            fill=(235, 235, 235),
            font=font,
            anchor="mm",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def sample_global_vmax(
    rows: list[dict[str, Any]], infer_dir: Path, base_path: Path, args: argparse.Namespace
) -> float:
    if args.heatmap_vmax > 0:
        return float(args.heatmap_vmax)
    if args.heatmap_scale != "global":
        return 0.0
    samples: list[np.ndarray] = []
    max_samples = max(1, int(args.heatmap_samples_per_image))
    for row in rows:
        prediction_path = infer_dir / prediction_name(row, args.prediction_suffix)
        target_value = row_value(row, args.target_key, args.target_fallback_key)
        if not prediction_path.is_file() or not target_value:
            continue
        target_path = resolve_data(target_value, base_path)
        if not target_path.is_file():
            continue
        prediction = load_rgb(prediction_path)
        target = transform_image(load_rgb(target_path, prediction.size), args.target_transform)
        flat = mse_map(prediction, target).reshape(-1)
        stride = max(1, math.ceil(flat.size / max_samples))
        samples.append(flat[::stride])
    return robust_vmax(np.concatenate(samples), args.heatmap_percentile) if samples else 1.0


def main() -> int:
    args = parse_args()
    infer_dir = resolve_repo(args.infer_dir)
    manifest = resolve_repo(args.manifest)
    base_path = resolve_repo(args.base_path)
    output_dir = resolve_repo(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(manifest, int(args.limit))
    global_vmax = sample_global_vmax(rows, infer_dir, base_path, args)

    records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        prediction_path = infer_dir / prediction_name(row, args.prediction_suffix)
        source_value = row.get(args.source_key)
        target_value = row_value(row, args.target_key, args.target_fallback_key)
        source_path = resolve_data(source_value, base_path) if source_value else None
        target_path = resolve_data(target_value, base_path) if target_value else None
        if not prediction_path.is_file() or source_path is None or not source_path.is_file() or target_path is None or not target_path.is_file():
            item = {"manifest_index": row.get("_manifest_index"), "prediction": str(prediction_path), "source": None if source_path is None else str(source_path), "target": None if target_path is None else str(target_path)}
            missing.append(item)
            if not args.allow_missing:
                raise FileNotFoundError(item)
            continue

        prediction = load_rgb(prediction_path)
        source = load_rgb(source_path, prediction.size)
        target = transform_image(load_rgb(target_path, prediction.size), args.target_transform)
        values = mse_map(prediction, target)
        vmax = global_vmax if global_vmax > 0 else robust_vmax(values, args.heatmap_percentile)
        output_path = output_dir / output_name(row, args.prediction_suffix)
        if args.overwrite or not output_path.exists():
            compose(
                [source, light_scene_image(row, args.attrs_key, prediction.width), prediction, target, heatmap_image(values, vmax)],
                output_path,
                int(args.caption_height),
                int(args.caption_font_size),
            )
        records.append(
            {
                "scene_id": row.get("scene_id"),
                "light_id": row.get("light_id"),
                "source": str(source_path),
                "prediction": str(prediction_path),
                "target": str(target_path),
                "output_5panel": str(output_path),
                "mse_mean": float(values.mean()),
                "heatmap_vmax": vmax,
            }
        )
        if index == 1 or index % 50 == 0 or index == len(rows):
            print(f"{index}/{len(rows)} panels, missing={len(missing)}")

    with (output_dir / "index.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": 1,
        "layout": "source | light scene | prediction | GT | error MSE heatmap",
        "infer_dir": str(infer_dir),
        "manifest": str(manifest),
        "base_path": str(base_path),
        "output_dir": str(output_dir),
        "expected_count": len(rows),
        "record_count": len(records),
        "missing_count": len(missing),
        "heatmap_scale": args.heatmap_scale,
        "heatmap_percentile": args.heatmap_percentile,
        "global_heatmap_vmax": global_vmax,
        "target_transform": args.target_transform,
        "missing": missing,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output_dir)
    return 1 if missing and not args.allow_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
