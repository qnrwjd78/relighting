#!/usr/bin/env python3
"""Build a standard TokenLight eval manifest from the Objaverse-245 bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch


DEFAULT_PROMPT = "photorealistic object relighting, preserve geometry and materials"


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def attrs_for_sample(sample: Mapping[str, Any], ambient: float) -> str:
    light = sample["light"]
    position = light["canonical_position"]
    color = light.get("component_color", light.get("render_color", [1.0, 1.0, 1.0]))
    return compact_json(
        {
            "a": float(ambient),
            "dg": 0.0,
            "lights": [
                {
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "z": float(position[2]),
                    "r": float(color[0]),
                    "g": float(color[1]),
                    "b": float(color[2]),
                    "lambda": float(light.get("power_scale", 0.6)),
                    "d": float(light.get("canonical_radius", 0.06)),
                }
            ],
            "t": 1.0,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    bundle = args.bundle_root.resolve()
    png_root = bundle / "unseen_7x7x5_power06_png_exclude_requested"
    cache_root = bundle / "objaverse_245_eval_cache_2500_2999"
    point_root = bundle / "unseen_7x7x5_power06_source"
    scene_files = sorted((cache_root / "scenes").glob("scene_*.pt"))
    if not scene_files:
        raise FileNotFoundError(f"No eval scene caches under {cache_root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    scene_count = 0
    row_count = 0
    height_values: set[float] = set()
    with temporary.open("w", encoding="utf-8") as handle:
        for scene_file in scene_files:
            cache = torch.load(scene_file, map_location="cpu", weights_only=False, mmap=True)
            scene_id = str(cache["scene_id"])
            scene_dir = png_root / "scenes" / scene_id
            meta_path = scene_dir / "meta.json"
            point_path = point_root / scene_id / "source.npy"
            if not meta_path.is_file() or not point_path.is_file():
                raise FileNotFoundError(f"Incomplete eval scene {scene_id}")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            ambient = float(meta["source"]["ambient_source"]["strength"])
            samples = cache["samples"]
            sample_images = cache["sample_images"]
            if len(samples) != len(sample_images) or len(samples) != int(cache["sample_latents"].shape[0]):
                raise ValueError(f"Cache row mismatch in {scene_file}")
            for sample_index, (sample, image_path) in enumerate(zip(samples, sample_images)):
                light = sample["light"]
                masks = sample.get("masks", {})
                position = [float(value) for value in light["canonical_position"]]
                height_values.add(position[2])
                row = {
                    "scene_id": scene_id,
                    "scene_folder": scene_id,
                    "task": str(sample.get("task", "position")),
                    "sample_name": str(sample.get("name", Path(image_path).stem)),
                    "light_id": int(light["id"]),
                    "input_image": f"scenes/{scene_id}/{cache['source_image']}",
                    "video": f"scenes/{scene_id}/{image_path}",
                    "mask": f"scenes/{scene_id}/masks/object_mask.png",
                    "inf_mask": f"scenes/{scene_id}/{masks['object_direct_lit_clean']}",
                    "shadow_mask": f"scenes/{scene_id}/{masks['object_shadow_clean']}",
                    "shadow_mask_pad16": f"scenes/{scene_id}/{masks['object_shadow_clean_pad16']}",
                    "prompt": DEFAULT_PROMPT,
                    "attrs_json": attrs_for_sample(sample, ambient),
                    "light_position": position,
                    "light_height": position[2],
                    "grid_cell": light.get("grid_cell"),
                    "grid_resolution": light.get("grid_resolution"),
                    "valid": True,
                    "point_map": point_path.resolve().as_posix(),
                    "_scene_cache_file": scene_file.resolve().as_posix(),
                    "_scene_cache_sample_index": sample_index,
                    "_image_transform": str(cache.get("image_transform", "")),
                }
                handle.write(compact_json(row) + "\n")
                row_count += 1
            scene_count += 1
    temporary.replace(args.output)
    summary = {
        "schema": "objaverse245_eval_manifest_v1",
        "bundle_root": bundle.as_posix(),
        "png_root": png_root.as_posix(),
        "cache_root": cache_root.as_posix(),
        "point_map_root": point_root.as_posix(),
        "scene_count": scene_count,
        "row_count": row_count,
        "height_values": sorted(height_values),
        "image_transform": "luminance",
    }
    args.output.with_name("manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
