#!/usr/bin/env python3
"""Attach exact per-position cast-shadow paths to the exp_1 scene-cache metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = WORKSPACE / "data_train/objaverse_fixed_7x7x5_power06_rgb_480/metadata.jsonl"
DEFAULT_OUTPUT = (
    WORKSPACE
    / "data_train/objaverse_fixed_7x7x5_power06_rgb_shadow_mask_480/metadata.jsonl"
)
DEFAULT_ROOTS = (
    WORKSPACE
    / "downloads/blender_relight/objaverse_245/objaverse_245_train_pbr_masks_0000_0999",
    WORKSPACE / "downloads/objaverse_245_train_pbr_masks_1000_1999",
)
SHADOW_NAME = "object_shadow_geometry_ray_clean_minarea00.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mask-root", action="append", type=Path, dest="mask_roots")
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    return parser.parse_args()


def _workspace_relative(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Mask path must be under workspace {workspace}: {path}") from error


def discover_scene_roots(mask_roots: list[Path]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for root in mask_roots:
        scenes = root.resolve() / "scenes"
        if not scenes.is_dir():
            raise FileNotFoundError(f"Missing mask scenes directory: {scenes}")
        for scene in scenes.iterdir():
            if not scene.is_dir() or not scene.name.startswith("scene_"):
                continue
            previous = result.setdefault(scene.name, root.resolve())
            if previous != root.resolve():
                raise ValueError(f"Scene {scene.name} occurs in both {previous} and {root}")
    if not result:
        raise ValueError("No scene directories found in the supplied mask roots")
    return result


def shadow_path(row: dict[str, Any], scene_roots: dict[str, Path]) -> Path:
    scene_id = str(row.get("scene_id") or row.get("scene_folder") or "")
    sample_name = str(row.get("sample_name") or "")
    if not scene_id or not sample_name.startswith("position_"):
        raise ValueError(f"Malformed scene/sample identity: {scene_id!r}, {sample_name!r}")
    root = scene_roots.get(scene_id)
    if root is None:
        raise KeyError(f"No mask shard contains {scene_id}")
    return (
        root
        / "scenes"
        / scene_id
        / "samples"
        / "position"
        / f"{sample_name}_masks"
        / SHADOW_NAME
    )


def prepare(input_path: Path, output_path: Path, mask_roots: list[Path], workspace: Path) -> dict:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    workspace = workspace.resolve()
    scene_roots = discover_scene_roots(mask_roots)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    input_hash = hashlib.sha256()
    output_hash = hashlib.sha256()
    rows = 0
    scenes: set[str] = set()
    shard_rows: dict[str, int] = {
        _workspace_relative(root.resolve(), workspace): 0 for root in mask_roots
    }
    seen: set[tuple[str, str]] = set()

    try:
        with input_path.open("rb") as source, temporary.open("wb") as target:
            for line_number, raw_line in enumerate(source, start=1):
                input_hash.update(raw_line)
                if not raw_line.strip():
                    continue
                row = json.loads(raw_line)
                scene_id = str(row.get("scene_id") or row.get("scene_folder") or "")
                sample_name = str(row.get("sample_name") or "")
                identity = (scene_id, sample_name)
                if identity in seen:
                    raise ValueError(f"Duplicate scene/sample at line {line_number}: {identity}")
                seen.add(identity)
                path = shadow_path(row, scene_roots)
                if not path.is_file():
                    raise FileNotFoundError(f"Missing shadow mask at line {line_number}: {path}")
                row["shadow_mask"] = _workspace_relative(path, workspace)
                encoded = (
                    json.dumps(row, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
                    + "\n"
                ).encode("utf-8")
                target.write(encoded)
                output_hash.update(encoded)
                rows += 1
                scenes.add(scene_id)
                shard_rows[_workspace_relative(scene_roots[scene_id], workspace)] += 1
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    summary = {
        "schema": "tokenlight_exp1_shadow_mask_metadata_v1",
        "input": input_path.as_posix(),
        "output": output_path.as_posix(),
        "workspace": workspace.as_posix(),
        "mask_roots": [root.resolve().as_posix() for root in mask_roots],
        "shadow_filename": SHADOW_NAME,
        "row_count": rows,
        "scene_count": len(scenes),
        "shard_rows": dict(sorted(shard_rows.items())),
        "input_sha256": input_hash.hexdigest(),
        "output_sha256": output_hash.hexdigest(),
    }
    summary_path = output_path.with_name("metadata_summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    roots = args.mask_roots or list(DEFAULT_ROOTS)
    summary = prepare(args.input, args.output, roots, args.workspace)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
