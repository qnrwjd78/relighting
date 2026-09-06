#!/usr/bin/env python3
"""Create 10fps RGB Output | GT videos for each selected scene-height."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.make_height_output_gt_videos import (
    height_label,
    label_bar,
    ordered_rows,
    prediction_name,
    resolve,
)


def make_frame(prediction: Path, target: Path, destination: Path) -> None:
    with Image.open(prediction) as image:
        output = image.convert("RGB")
    with Image.open(target) as image:
        gt = image.convert("RGB")
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
    predictions, manifest, base_path = resolve(args.predictions), resolve(args.manifest), resolve(args.base_path)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
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
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y" if args.overwrite else "-n",
                 "-framerate", str(args.fps), "-i", str(frame_dir / "%06d.png"), "-frames:v", str(len(sequence)),
                 "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", str(destination)],
                check=True,
            )
        completed += 1
        print(f"[video] {completed}/{len(groups)} {destination.name}", flush=True)
    summary = {"layout": "Output | GT", "target_transform": "rgb", "fps": args.fps,
               "video_count": completed, "group_count": len(groups)}
    (output_dir / "video_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
