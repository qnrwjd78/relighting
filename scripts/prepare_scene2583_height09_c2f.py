#!/usr/bin/env python3
"""Create the ordered 49-frame C2F manifest for scene 2583 at z=0.9."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/workspace")
RAW = ROOT / "data_train/objaverse_245_eval_2500_2999_rgb_random3/metadata_random3_heights.jsonl"
TRAJECTORY = ROOT / "outputs/infer/relighting_external/scene_002583/height_0.9_49/trajectory_manifest.json"
OUT = ROOT / "data_train/shadow_c2f_scene2583_height09/trajectory.jsonl"
DATA = ROOT / "data/objaverse_245_eval_2500_2999/unseen_7x7x5_power06_png_exclude_requested"
RUN = ROOT / "outputs/infer/shadow_c2f/scene_002583_height_0.9_epoch001"


def main() -> int:
    wanted = json.loads(TRAJECTORY.read_text())["frames"]
    ids = {int(frame["position_id"]) for frame in wanted}
    raw = {}
    with RAW.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("scene_id") == "scene_002583" and int(row.get("light_id", -1)) in ids:
                raw[int(row["light_id"])] = row
    rows = []
    scene = DATA / "scenes/scene_002583"
    for frame in wanted:
        light_id = int(frame["position_id"])
        row = dict(raw[light_id])
        row.update({
            "frame_index": int(frame["frame_index"]),
            "source_image": str(scene / "source.png"),
            "target_image": str(scene / f"samples/position/position_{light_id:03d}.png"),
            "object_mask": str(scene / "masks/object_mask.png"),
            "receiver_mask": str(scene / "masks/receiver_mask.png"),
            "gt_shadow_mask": str(scene / f"samples/position/position_{light_id:03d}_masks/object_shadow_geometry_ray_clean_minarea00.png"),
            "pbr_depth": str(scene / "pbr/depth.png"),
            "pbr_normal": str(scene / "pbr/normal.png"),
            "scene_meta": str(scene / "meta.json"),
            "predicted_shadow_mask": str(RUN / f"masks/{int(frame['frame_index']):03d}.png"),
            "refined_probability": str(RUN / f"probabilities/{int(frame['frame_index']):03d}.npy"),
        })
        rows.append(row)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
    temporary.replace(OUT)
    print(json.dumps({"output": str(OUT), "rows": len(rows), "position_ids": [r["light_id"] for r in rows]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
