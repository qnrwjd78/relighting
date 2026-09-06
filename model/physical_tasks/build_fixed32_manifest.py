#!/usr/bin/env python3
"""Build geometry-ray visibility/shadow metadata from objaverse_fixed32_png."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/objaverse_fixed32_png")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data_train/objaverse_fixed32_physical/metadata.jsonl"
    )
    parser.add_argument("--reject-metadata", type=Path, default=None)
    parser.add_argument("--val-scene-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=480)
    parser.add_argument("--include-rejected", action="store_true")
    parser.add_argument("--no-check-files", action="store_true")
    return parser.parse_args()


def relative(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def stable_split(value: str, fraction: float, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    unit = int.from_bytes(digest[:8], "big") / float(2**64)
    return "val" if unit < fraction else "train"


def vector(value: Any, length: int, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field} must have {length} values, got {value!r}")
    return [float(item) for item in value]


def build_scene(scene_dir: Path, split: str) -> list[dict[str, Any]]:
    meta = json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))
    scene_id = str(meta.get("scene_id") or scene_dir.name)
    camera = meta["camera"]
    common = meta["common"]
    pbr = common["pbr_maps"]
    depth_range = pbr["depth_png_range"]
    rotation = camera.get("similarity_transform", {}).get("rotation_matrix")
    if rotation is None:
        rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    rotation = [vector(row, 3, f"{scene_id}.rotation") for row in rotation]
    base = {
        "dataset": "objaverse_fixed32_png",
        "scene_id": scene_id,
        "split": split,
        "source_image": relative(scene_dir / meta["source"]["ambient_only"]["render"]),
        "depth_image": relative(scene_dir / pbr["depth"]),
        "normal_image": relative(scene_dir / pbr["normal"]),
        "object_mask": relative(scene_dir / common["object_mask"]),
        "receiver_mask": relative(scene_dir / common["receiver_masks"]["receiver"]),
        "depth_min_meters": float(depth_range["min_meters"]),
        "depth_max_meters": float(depth_range["max_meters"]),
        "depth_encoding": "depth_percentile_1_99_near_white",
        "fov_degrees": float(camera.get("fov_degrees", 39.6)),
        "canonical_scale": 1.0,
        "canonical_camera_location": vector(camera["location"], 3, "camera.location"),
        "camera_ray_to_geometry": rotation,
        "normal_world_to_canonical": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }
    rows: list[dict[str, Any]] = []
    for sample in meta.get("samples", []):
        if sample.get("task") != "position":
            continue
        light = sample.get("light", {})
        masks = sample.get("masks", {})
        row = {
            **base,
            "sample_id": f"objaverse_fixed32_png/{scene_id}/{sample['name']}",
            "visibility_mask": relative(scene_dir / masks["object_direct_lit_clean"]),
            "shadow_mask": relative(scene_dir / masks["object_shadow_clean"]),
            "light_id": int(light["id"]),
            "light_position": vector(light["world_position"], 3, "light_position"),
        }
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.val_scene_fraction < 1.0:
        raise ValueError("--val-scene-fraction must be in [0, 1)")
    data_root = args.data_root.expanduser().resolve()
    reject_path = args.reject_metadata or data_root / "reject_metadata.txt"
    rejected = set()
    if reject_path.is_file() and not args.include_rejected:
        rejected = {line.strip() for line in reject_path.read_text().splitlines() if line.strip()}
    rows: list[dict[str, Any]] = []
    scene_counts = Counter()
    for meta_path in sorted((data_root / "scenes").glob("*/meta.json")):
        scene_id = meta_path.parent.name
        if scene_id in rejected:
            continue
        split = stable_split(scene_id, args.val_scene_fraction, args.seed)
        current = build_scene(meta_path.parent, split)
        rows.extend(current)
        scene_counts[split] += 1
    keys = (
        "source_image", "depth_image", "normal_image",
        "object_mask", "receiver_mask", "visibility_mask", "shadow_mask",
    )
    if not args.no_check_files:
        missing = [(row["sample_id"], key, row[key]) for row in rows for key in keys if not (ROOT / row[key]).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing files (first 20): {missing[:20]}")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    row_counts = Counter(row["split"] for row in rows)
    print(f"Wrote {len(rows)} rows to {output}")
    print(f"Excluded {len(rejected)} scene ids from {reject_path}")
    for split in sorted(scene_counts):
        print(f"  {split}: scenes={scene_counts[split]} rows={row_counts[split]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
