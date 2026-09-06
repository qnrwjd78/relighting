#!/usr/bin/env python3
"""Build a single-light manifest from composed TokenLight PNG datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS = (
    ROOT / "data/portrait_png",
    ROOT / "data/objaverse_test_seen",
    ROOT / "data/objaverse_test_unseen",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-roots", nargs="+", type=Path, default=list(DEFAULT_ROOTS))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data_train/physical_tasks_single_light/metadata.jsonl",
    )
    parser.add_argument("--val-scene-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=480)
    parser.add_argument("--no-check-files", action="store_true")
    return parser.parse_args()


def stable_unit_interval(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def relative_or_absolute(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def require_vector(value: Any, length: int, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field} must be a {length}-vector, got {value!r}")
    return [float(item) for item in value]


def scene_rows(data_root: Path, scene_dir: Path, val_fraction: float, seed: int) -> list[dict[str, Any]]:
    samples_path = scene_dir / "samples_manifest.json"
    meta_path = scene_dir / "meta.json"
    samples_doc = load_object(samples_path)
    meta = load_object(meta_path)
    scene_id = str(samples_doc.get("scene_id") or meta.get("scene_id") or scene_dir.name)
    dataset_name = data_root.name
    split_key = f"{dataset_name}/{scene_id}"
    split = "val" if stable_unit_interval(split_key, seed) < val_fraction else "train"

    camera = meta.get("camera", {})

    common = {
        "dataset": dataset_name,
        "scene_id": scene_id,
        "split": split,
        "source_image": relative_or_absolute(scene_dir / str(samples_doc.get("source_image", "source.png"))),
        "canonical_camera_location": require_vector(
            camera.get("canonical_position", [0.0, -3.5, 0.0]), 3, f"{scene_id}.canonical_camera_location"
        ),
    }

    rows: list[dict[str, Any]] = []
    for sample in samples_doc.get("samples", []):
        image_value = str(sample.get("image", ""))
        if sample.get("task") != "single_light" or not Path(image_value).name.startswith("light_"):
            continue
        lights = sample.get("lights")
        if not isinstance(lights, list) or len(lights) != 1:
            continue
        light = lights[0]
        row = {
            **common,
            "sample_id": f"{dataset_name}/{scene_id}/{Path(image_value).stem}",
            "target_image": relative_or_absolute(scene_dir / image_value),
            "light_id": int(light.get("id", len(rows))),
            "light_position": require_vector(light.get("position"), 3, f"{scene_id}.{image_value}.position"),
            "light_color": require_vector(light.get("color"), 3, f"{scene_id}.{image_value}.color"),
            "light_intensity": float(light.get("intensity", 1.0)),
            "light_radius": float(light.get("radius", 0.1)),
            "ambient_scale": float(sample.get("global_control", {}).get("ambient_scale", 1.0)),
            "tonemap": samples_doc.get("lighting_png_tonemap", "reinhard_gamma"),
            "gamma": float(samples_doc.get("lighting_png_gamma", 2.2)),
        }
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.val_scene_fraction < 1.0:
        raise ValueError("--val-scene-fraction must be in [0, 1)")
    rows: list[dict[str, Any]] = []
    for root in args.data_roots:
        data_root = root.expanduser().resolve()
        manifests = sorted((data_root / "scenes").glob("*/samples_manifest.json"))
        if not manifests:
            raise FileNotFoundError(f"No scene manifests found under {data_root / 'scenes'}")
        for samples_path in manifests:
            rows.extend(scene_rows(data_root, samples_path.parent, args.val_scene_fraction, args.seed))

    if not args.no_check_files:
        path_keys = ("source_image", "target_image")
        missing = [(row["sample_id"], key, row[key]) for row in rows for key in path_keys if not (ROOT / row[key]).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing referenced files (first 20): {missing[:20]}")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    counts = Counter((row["dataset"], row["split"]) for row in rows)
    print(f"Wrote {len(rows)} single-light rows to {output}")
    for key, count in sorted(counts.items()):
        print(f"  {key[0]:24s} {key[1]:5s} {count:6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
