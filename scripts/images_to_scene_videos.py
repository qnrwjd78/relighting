#!/usr/bin/env python3
"""Create per-scene output and GT videos from relighting PNG results.

Expected input names::

    <scene>_light_<number>.png
    <scene>_light_<number>_withgt.png

The regular PNG is used for the output video.  The rightmost panel of the
``_withgt`` PNG is cropped to the regular PNG size and used for the GT video.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from PIL import Image


FRAME_RE = re.compile(
    r"^(?P<scene>.+)_light_(?P<light>\d+)(?P<withgt>_withgt)?\.png$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one output MP4 and one GT MP4 for every scene."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing result PNGs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination root (default: <input_dir>_videos)",
    )
    parser.add_argument("--fps", type=float, default=6.0, help="Frames per second (default: 6)")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Target duration in seconds; overrides --fps for each scene",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="H.264 constant-rate factor; lower is higher quality (default: 18)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace MP4 files that already exist"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="JSONL manifest containing scene_id, light_id, and attrs_json",
    )
    parser.add_argument(
        "--order",
        choices=("light-id", "spatial-zigzag"),
        default="light-id",
        help="Frame ordering (default: light-id)",
    )
    parser.add_argument(
        "--split-layers",
        action="store_true",
        help="Create separate low/high videos; requires --manifest",
    )
    return parser.parse_args()


def discover_frames(input_dir: Path) -> dict[str, dict[int, dict[str, Path]]]:
    scenes: dict[str, dict[int, dict[str, Path]]] = defaultdict(lambda: defaultdict(dict))
    for path in input_dir.iterdir():
        if not path.is_file():
            continue
        match = FRAME_RE.match(path.name)
        if not match:
            continue
        kind = "withgt" if match.group("withgt") else "output"
        scenes[match.group("scene")][int(match.group("light"))][kind] = path.resolve()
    return scenes


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def load_light_positions(manifest: Path) -> dict[tuple[str, int], tuple[float, float, float]]:
    positions: dict[tuple[str, int], tuple[float, float, float]] = {}
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                attrs = row.get("attrs_json", {})
                if isinstance(attrs, str):
                    attrs = json.loads(attrs)
                light = attrs["lights"][0]
                key = (str(row["scene_id"]), int(row["light_id"]))
                positions[key] = (float(light["x"]), float(light["y"]), float(light["z"]))
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid manifest row {line_number}: {exc}") from exc
    return positions


def spatial_zigzag_order(
    scene: str,
    light_ids: list[int],
    positions: dict[tuple[str, int], tuple[float, float, float]],
) -> list[int]:
    missing = [light_id for light_id in light_ids if (scene, light_id) not in positions]
    if missing:
        raise ValueError(f"manifest has no positions for light IDs: {missing}")

    by_z: dict[float, list[int]] = defaultdict(list)
    for light_id in light_ids:
        by_z[positions[(scene, light_id)][2]].append(light_id)

    ordered: list[int] = []
    for layer_index, z in enumerate(sorted(by_z)):
        by_x: dict[float, list[int]] = defaultdict(list)
        for light_id in by_z[z]:
            by_x[positions[(scene, light_id)][0]].append(light_id)
        x_values = sorted(by_x, reverse=bool(layer_index % 2))
        for column_index, x in enumerate(x_values):
            column = sorted(
                by_x[x],
                key=lambda light_id: positions[(scene, light_id)][1],
                reverse=bool(column_index % 2),
            )
            ordered.extend(column)
    return ordered


def split_height_layers(
    scene: str,
    light_ids: list[int],
    positions: dict[tuple[str, int], tuple[float, float, float]],
    known_z_values: list[float],
) -> list[tuple[str, list[int]]]:
    by_z: dict[float, list[int]] = defaultdict(list)
    for light_id in light_ids:
        try:
            z = positions[(scene, light_id)][2]
        except KeyError as exc:
            raise ValueError(f"manifest has no position for light ID {light_id}") from exc
        by_z[z].append(light_id)
    z_values = sorted(by_z)
    if len(z_values) == 1 and len(known_z_values) == 2:
        label = "low" if z_values[0] == known_z_values[0] else "high"
        return [(label, by_z[z_values[0]])]
    if len(z_values) != 2:
        raise ValueError(f"expected 1 or 2 known height layers, found {len(z_values)}")
    return [("low", by_z[z_values[0]]), ("high", by_z[z_values[1]])]


def stage_sequence(paths: list[Path], directory: Path) -> str:
    for index, source in enumerate(paths):
        (directory / f"{index:06d}.png").symlink_to(source)
    return str(directory / "%06d.png")


def encode(
    ffmpeg: str,
    pattern: str,
    destination: Path,
    frame_count: int,
    fps: float,
    crf: int,
    video_filter: str,
    overwrite: bool,
) -> None:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-framerate",
        str(fps),
        "-start_number",
        "0",
        "-i",
        pattern,
        "-frames:v",
        str(frame_count),
        "-vf",
        video_filter,
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
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        message = result.stderr.strip() or "unknown ffmpeg error"
        raise RuntimeError(f"Failed to create {destination}: {message}")


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        print(f"error: input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    if args.fps <= 0:
        print("error: --fps must be positive", file=sys.stderr)
        return 2
    if args.duration is not None and args.duration <= 0:
        print("error: --duration must be positive", file=sys.stderr)
        return 2
    if not 0 <= args.crf <= 51:
        print("error: --crf must be between 0 and 51", file=sys.stderr)
        return 2
    if (args.order == "spatial-zigzag" or args.split_layers) and args.manifest is None:
        print("error: spatial ordering and layer splitting require --manifest", file=sys.stderr)
        return 2

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("error: ffmpeg is not installed or not on PATH", file=sys.stderr)
        return 2

    scenes = discover_frames(input_dir)
    if not scenes:
        print(f"error: no matching PNG files found in {input_dir}", file=sys.stderr)
        return 2

    light_positions: dict[tuple[str, int], tuple[float, float, float]] = {}
    if args.manifest is not None:
        manifest = args.manifest.expanduser().resolve()
        if not manifest.is_file():
            print(f"error: manifest does not exist: {manifest}", file=sys.stderr)
            return 2
        try:
            light_positions = load_light_positions(manifest)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    known_z_values = sorted({position[2] for position in light_positions.values()})

    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_dir.with_name(f"{input_dir.name}_videos")
    )
    output_video_dir = output_root / "output"
    gt_video_dir = output_root / "gt"

    created = 0
    skipped = 0
    failed = 0
    for scene, light_records in sorted(scenes.items()):
        paired_ids = sorted(
            light_id
            for light_id, records in light_records.items()
            if "output" in records and "withgt" in records
        )
        if not paired_ids:
            print(f"[skip] {scene}: no output/_withgt frame pairs", file=sys.stderr)
            failed += 1
            continue

        if args.order == "spatial-zigzag":
            try:
                paired_ids = spatial_zigzag_order(scene, paired_ids, light_positions)
            except ValueError as exc:
                print(f"[error] {scene}: {exc}", file=sys.stderr)
                failed += 1
                continue

        unpaired = len(light_records) - len(paired_ids)
        if unpaired:
            print(f"[warn] {scene}: ignored {unpaired} unpaired light frame(s)", file=sys.stderr)

        all_output_frames = [light_records[i]["output"] for i in paired_ids]
        all_withgt_frames = [light_records[i]["withgt"] for i in paired_ids]
        width, height = image_size(all_output_frames[0])
        if any(image_size(path) != (width, height) for path in all_output_frames):
            print(f"[error] {scene}: output frame sizes are inconsistent", file=sys.stderr)
            failed += 1
            continue
        if any(
            composite_width < width or composite_height < height
            for composite_width, composite_height in map(image_size, all_withgt_frames)
        ):
            print(f"[error] {scene}: a _withgt image is smaller than its output", file=sys.stderr)
            failed += 1
            continue

        try:
            sequences = (
                split_height_layers(scene, paired_ids, light_positions, known_z_values)
                if args.split_layers
                else [("all", paired_ids)]
            )
        except ValueError as exc:
            print(f"[error] {scene}: {exc}", file=sys.stderr)
            failed += 1
            continue

        try:
            scene_created = False
            layer_summaries = []
            for layer, layer_ids in sequences:
                output_dir = output_video_dir / layer if args.split_layers else output_video_dir
                gt_dir = gt_video_dir / layer if args.split_layers else gt_video_dir
                output_dir.mkdir(parents=True, exist_ok=True)
                gt_dir.mkdir(parents=True, exist_ok=True)
                output_mp4 = output_dir / f"{scene}.mp4"
                gt_mp4 = gt_dir / f"{scene}.mp4"
                if not args.overwrite and output_mp4.exists() and gt_mp4.exists():
                    layer_summaries.append(f"{layer}=skipped")
                    continue

                output_frames = [light_records[i]["output"] for i in layer_ids]
                withgt_frames = [light_records[i]["withgt"] for i in layer_ids]
                scene_fps = (
                    len(layer_ids) / args.duration if args.duration is not None else args.fps
                )
                with tempfile.TemporaryDirectory(
                    prefix=f"{scene}_{layer}_"
                ) as temp_root_string:
                    temp_root = Path(temp_root_string)
                    # Staging also supports non-contiguous light IDs and paths with spaces.
                    output_stage = temp_root / "output"
                    gt_stage = temp_root / "gt"
                    output_stage.mkdir()
                    gt_stage.mkdir()
                    output_pattern = stage_sequence(output_frames, output_stage)
                    gt_pattern = stage_sequence(withgt_frames, gt_stage)

                    if args.overwrite or not output_mp4.exists():
                        encode(
                            ffmpeg,
                            output_pattern,
                            output_mp4,
                            len(output_frames),
                            scene_fps,
                            args.crf,
                            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                            args.overwrite,
                        )
                    if args.overwrite or not gt_mp4.exists():
                        encode(
                            ffmpeg,
                            gt_pattern,
                            gt_mp4,
                            len(withgt_frames),
                            scene_fps,
                            args.crf,
                            f"crop={width}:{height}:iw-{width}:0,"
                            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                            args.overwrite,
                        )
                scene_created = True
                layer_summaries.append(f"{layer}={len(layer_ids)}")
            duration_text = (
                f", target {args.duration:g}s" if args.duration is not None else ""
            )
            print(f"[ok] {scene}: {', '.join(layer_summaries)}{duration_text}")
            if scene_created:
                created += 1
            else:
                skipped += 1
        except Exception as exc:
            print(f"[error] {scene}: {exc}", file=sys.stderr)
            failed += 1

    print(
        f"done: {created} scene(s) created, {skipped} skipped, {failed} failed; "
        f"videos: {output_root}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
