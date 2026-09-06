#!/usr/bin/env python3
"""Assemble training-aligned GenLit chunks into 54-frame scene videos.

Each raw GenLit chunk contains ``000.png`` (the source image) followed by 25
generated frames.  Generated frames ``001.png`` through ``007.png`` are the
intensity warm-up and are intentionally discarded.  Frames ``008.png`` through
``025.png`` are the 18 steady-intensity targets retained from each of three
chunks, producing 54 output frames per scene.

All raw inputs for every requested scene are validated before any destination
is created.  Videos are assembled in temporary directories and published only
after ffprobe confirms their dimensions, frame rate, frame count, codec, and
pixel format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import NoReturn


DEFAULT_RAW_ROOT = Path(
    "/workspace/outputs/infer/relighting_external/"
    "genlit_dense_circle_r1.2_e60_54_scenes_002536_002553"
)
DEFAULT_OUTPUT_ROOT = Path("/workspace/outputs/infer/relighting_external")
DEFAULT_SCENES = ("scene_002536", "scene_002553")

RUN_NAME = "genlit_dense_circle_r1.2_e60_54"
CHUNK_INDICES = range(3)
RAW_FRAME_INDICES = range(26)
SELECTED_FRAME_INDICES = range(8, 26)
RAW_WIDTH = 640
RAW_HEIGHT = 448
OUTPUT_FRAME_COUNT = 54
FPS = 6
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class VideoSpec:
    suffix: str
    width: int
    height: int
    video_filter: str | None = None


VIDEO_SPECS = (
    VideoSpec(suffix="6fps", width=640, height=448),
    VideoSpec(
        suffix="480x480_6fps",
        width=480,
        height=480,
        video_filter="scale=480:480:flags=lanczos",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select frames 008..025 from each of three 25-frame GenLit chunks "
            "and create 54-frame 6 fps videos."
        )
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help=f"Root containing chunk_00..chunk_02 (default: {DEFAULT_RAW_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Per-scene output root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=list(DEFAULT_SCENES),
        help="Scene directory names (default: scene_002536 scene_002553)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate all raw PNG inputs without creating frames or videos",
    )
    return parser.parse_args()


def fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def png_dimensions(path: Path) -> tuple[int, int]:
    """Return PNG dimensions after validating its signature and IHDR header."""
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
    except OSError as exc:
        fail(f"Cannot read PNG {path}: {exc}")
    if len(header) != 24 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        fail(f"Invalid PNG signature/IHDR: {path}")
    return struct.unpack(">II", header[16:24])


def raw_frame_dir(raw_root: Path, scene: str, chunk_index: int) -> Path:
    return (
        raw_root
        / f"chunk_{chunk_index:02d}"
        / "validation_images_multi"
        / "checkpoint"
        / "videos"
        / scene
        / "0_images"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        fail(f"Cannot hash {path}: {exc}")
    return digest.hexdigest()


def validate_png_sequence(frame_dir: Path) -> None:
    """Fully decode all 26 PNGs so truncated/corrupt warm-up frames also fail."""
    run_checked(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-xerror",
            "-framerate",
            "1",
            "-start_number",
            "0",
            "-i",
            str(frame_dir / "%03d.png"),
            "-frames:v",
            str(len(RAW_FRAME_INDICES)),
            "-f",
            "null",
            "-",
        ],
        f"Decoding raw PNG sequence {frame_dir}",
    )


def validate_raw_inputs(raw_root: Path, scenes: list[str]) -> dict[str, list[Path]]:
    """Validate complete inference output and return selected frames in order."""
    if not raw_root.is_dir():
        fail(f"Raw inference root does not exist: {raw_root}")
    if not scenes:
        fail("At least one scene is required")
    if len(scenes) != len(set(scenes)):
        fail(f"Duplicate scene names are not allowed: {scenes}")

    selected_by_scene: dict[str, list[Path]] = {}
    expected_names = {f"{index:03d}.png" for index in RAW_FRAME_INDICES}

    for scene in scenes:
        if not scene or scene in {".", ".."} or Path(scene).name != scene:
            fail(f"Scene must be a single safe directory name: {scene!r}")
        selected: list[Path] = []
        source_digests: list[str] = []
        for chunk_index in CHUNK_INDICES:
            frame_dir = raw_frame_dir(raw_root, scene, chunk_index)
            if not frame_dir.is_dir():
                fail(f"Missing raw frame directory: {frame_dir}")

            actual_names = {
                path.name
                for path in frame_dir.iterdir()
                if path.is_file() and path.suffix.lower() == ".png"
            }
            missing = sorted(expected_names - actual_names)
            unexpected = sorted(actual_names - expected_names)
            if missing or unexpected:
                details = []
                if missing:
                    details.append(f"missing={missing}")
                if unexpected:
                    details.append(f"unexpected={unexpected}")
                fail(f"Expected exactly 000.png..025.png in {frame_dir}: " + ", ".join(details))

            for frame_index in RAW_FRAME_INDICES:
                frame = frame_dir / f"{frame_index:03d}.png"
                dimensions = png_dimensions(frame)
                if dimensions != (RAW_WIDTH, RAW_HEIGHT):
                    fail(
                        f"Expected {RAW_WIDTH}x{RAW_HEIGHT} PNG, got "
                        f"{dimensions[0]}x{dimensions[1]}: {frame}"
                    )
            validate_png_sequence(frame_dir)
            source_digests.append(file_sha256(frame_dir / "000.png"))
            selected.extend(
                (frame_dir / f"{frame_index:03d}.png").resolve(strict=True)
                for frame_index in SELECTED_FRAME_INDICES
            )

        if len(set(source_digests)) != 1:
            fail(
                f"Source frame 000.png differs between chunks for {scene}: "
                f"{source_digests}"
            )
        if len(selected) != OUTPUT_FRAME_COUNT:
            fail(f"Internal frame-selection error for {scene}: got {len(selected)} frames")
        selected_by_scene[scene] = selected

    return selected_by_scene


def run_checked(command: list[str], description: str) -> None:
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "no stderr"
        fail(f"{description} failed (exit {result.returncode}): {stderr}")


def encode_video(frame_pattern: Path, destination: Path, spec: VideoSpec) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(FPS),
        "-start_number",
        "0",
        "-i",
        str(frame_pattern),
        "-frames:v",
        str(OUTPUT_FRAME_COUNT),
    ]
    if spec.video_filter is not None:
        command.extend(["-vf", spec.video_filter])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )
    run_checked(command, f"Encoding {destination}")


def probe_video(path: Path) -> dict[str, object]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "no stderr"
        fail(f"Probing {path} failed (exit {result.returncode}): {stderr}")
    try:
        payload = json.loads(result.stdout)
        streams = payload["streams"]
        if len(streams) != 1:
            fail(f"Expected one video stream in {path}, found {len(streams)}")
        return streams[0]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        fail(f"Invalid ffprobe response for {path}: {exc}")


def validate_video(path: Path, spec: VideoSpec) -> dict[str, object]:
    stream = probe_video(path)
    expected_fields: dict[str, object] = {
        "codec_name": "h264",
        "width": spec.width,
        "height": spec.height,
        "pix_fmt": "yuv420p",
        "nb_read_frames": str(OUTPUT_FRAME_COUNT),
    }
    mismatches = [
        f"{key}={stream.get(key)!r} (expected {expected!r})"
        for key, expected in expected_fields.items()
        if stream.get(key) != expected
    ]
    try:
        actual_fps = Fraction(str(stream["avg_frame_rate"]))
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        fail(f"Invalid frame rate reported for {path}: {exc}")
    if actual_fps != FPS:
        mismatches.append(f"avg_frame_rate={actual_fps} (expected {FPS})")
    if mismatches:
        fail(f"Video validation failed for {path}: " + ", ".join(mismatches))
    return stream


def stage_scene(
    output_root: Path,
    scene: str,
    selected_frames: list[Path],
) -> tuple[tempfile.TemporaryDirectory[str], Path, list[dict[str, object]]]:
    scene_root = output_root / scene
    destination = scene_root / RUN_NAME
    if destination.exists() or destination.is_symlink():
        fail(f"Refusing to replace existing destination: {destination}")
    scene_root.mkdir(parents=True, exist_ok=True)

    temporary = tempfile.TemporaryDirectory(prefix=f".{RUN_NAME}.", dir=scene_root)
    stage = Path(temporary.name)
    frames_dir = stage / "frames"
    frames_dir.mkdir()
    for output_index, source in enumerate(selected_frames):
        link = frames_dir / f"{output_index:03d}.png"
        link.symlink_to(source)
        if link.resolve(strict=True) != source:
            fail(f"Staged frame link does not resolve to its source: {link}")

    staged_names = sorted(path.name for path in frames_dir.iterdir())
    expected_staged_names = [f"{index:03d}.png" for index in range(OUTPUT_FRAME_COUNT)]
    if staged_names != expected_staged_names:
        fail(f"Staged frame sequence is incomplete for {scene}")

    probes: list[dict[str, object]] = []
    pattern = frames_dir / "%03d.png"
    for spec in VIDEO_SPECS:
        video = stage / f"{scene}_{RUN_NAME}f_{spec.suffix}.mp4"
        encode_video(pattern, video, spec)
        probes.append(validate_video(video, spec))
    return temporary, destination, probes


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if shutil.which("ffmpeg") is None:
        fail("Required executable is not on PATH: ffmpeg")

    selected_by_scene = validate_raw_inputs(raw_root, args.scenes)
    print(
        f"Validated {len(args.scenes)} scenes x 3 chunks x 26 PNGs; "
        f"selected {OUTPUT_FRAME_COUNT} frames per scene (008..025 per chunk)."
    )
    if args.check_only:
        return 0

    if shutil.which("ffprobe") is None:
        fail("Required executable is not on PATH: ffprobe")

    destinations = {
        scene: output_root / scene / RUN_NAME
        for scene in args.scenes
    }
    for destination in destinations.values():
        if destination.exists() or destination.is_symlink():
            fail(f"Refusing to replace existing destination: {destination}")

    staged: list[
        tuple[tempfile.TemporaryDirectory[str], Path, list[dict[str, object]]]
    ] = []
    try:
        # Encode and validate every scene before publishing any destination.
        for scene in args.scenes:
            staged.append(stage_scene(output_root, scene, selected_by_scene[scene]))

        published: list[tuple[Path, Path]] = []
        try:
            for temporary, destination, _ in staged:
                stage = Path(temporary.name)
                os.rename(stage, destination)
                published.append((destination, stage))
        except OSError as exc:
            rollback_errors = []
            for destination, stage in reversed(published):
                try:
                    os.rename(destination, stage)
                except OSError as rollback_exc:
                    rollback_errors.append(
                        f"could not roll back {destination}: {rollback_exc}"
                    )
            detail = f"Publishing scene outputs failed: {exc}"
            if rollback_errors:
                detail += "; " + "; ".join(rollback_errors)
            fail(detail)

        for scene, (_, destination, probes) in zip(args.scenes, staged):
            print(f"scene={scene}")
            print(f"frames={destination / 'frames'}")
            for spec, probe in zip(VIDEO_SPECS, probes):
                video = destination / f"{scene}_{RUN_NAME}f_{spec.suffix}.mp4"
                print(f"video={video}")
                print(f"video_stream={json.dumps(probe, sort_keys=True)}")
    finally:
        for temporary, _, _ in staged:
            temporary.cleanup()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
