#!/usr/bin/env python3
"""Create one luminance Output | GT video per scene and light height."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.make_5panel import transform_image


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def height_label(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def prediction_name(row: dict) -> str:
    return f"{row['scene_id']}_light_{int(row['light_id']):03d}.png"


def ordered_rows(rows: list[dict]) -> list[dict]:
    by_x: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        cell = row.get("grid_cell")
        if not isinstance(cell, list) or len(cell) != 3:
            raise ValueError(f"Missing grid_cell for {row['scene_id']}/{row['light_id']}")
        by_x[int(cell[0])].append(row)
    output: list[dict] = []
    for column_index, x_index in enumerate(sorted(by_x)):
        output.extend(
            sorted(
                by_x[x_index],
                key=lambda row: int(row["grid_cell"][1]),
                reverse=bool(column_index % 2),
            )
        )
    return output


def label_bar(image: Image.Image, text: str, height: int = 38) -> Image.Image:
    output = Image.new("RGB", (image.width, image.height + height), "black")
    output.paste(image, (0, 0))
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default(size=20)
    draw.text((image.width // 2, image.height + height // 2), text, fill="white", anchor="mm", font=font)
    return output


def make_frame(prediction: Path, target: Path, destination: Path) -> None:
    with Image.open(prediction) as image:
        output = image.convert("RGB")
    with Image.open(target) as image:
        gt = transform_image(image.convert("RGB"), "luminance")
    if gt.size != output.size:
        gt = gt.resize(output.size, Image.Resampling.BICUBIC)
    panels = [label_bar(output, "Output"), label_bar(gt, "GT")]
    canvas = Image.new("RGB", (panels[0].width * 2, panels[0].height), "black")
    canvas.paste(panels[0], (0, 0))
    canvas.paste(panels[1], (panels[0].width, 0))
    canvas.save(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.fps <= 0 or not 0 <= args.crf <= 51:
        raise ValueError("fps must be positive and crf must be in [0,51]")

    predictions = resolve(args.predictions)
    manifest = resolve(args.manifest)
    base_path = resolve(args.base_path)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("valid") is False:
                continue
            groups[(str(row["scene_id"]), round(float(row["light_height"]), 6))].append(row)

    completed = 0
    for (scene_id, height), rows in sorted(groups.items()):
        destination = output_dir / f"{scene_id}_height_{height_label(height)}_{args.fps:g}fps.mp4"
        if destination.exists() and not args.overwrite:
            completed += 1
            continue
        sequence = ordered_rows(rows)
        with tempfile.TemporaryDirectory(prefix=f"{scene_id}_height_") as temporary:
            frame_dir = Path(temporary)
            for index, row in enumerate(sequence):
                prediction = predictions / prediction_name(row)
                target = base_path / str(row.get("target_image") or row["video"])
                if not prediction.is_file() or not target.is_file():
                    raise FileNotFoundError(prediction if not prediction.is_file() else target)
                make_frame(prediction, target, frame_dir / f"{index:06d}.png")
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y" if args.overwrite else "-n",
                "-framerate", str(args.fps), "-i", str(frame_dir / "%06d.png"),
                "-frames:v", str(len(sequence)), "-c:v", "libx264", "-preset", "medium",
                "-crf", str(args.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(destination),
            ]
            subprocess.run(command, check=True)
        completed += 1
        print(f"[video] {completed}/{len(groups)} {destination.name}", flush=True)

    summary = {
        "layout": "Output | GT",
        "target_transform": "luminance",
        "fps": args.fps,
        "video_count": completed,
        "group_count": len(groups),
    }
    (output_dir / "video_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
