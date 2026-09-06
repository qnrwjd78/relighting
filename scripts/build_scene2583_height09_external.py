#!/usr/bin/env python3
"""Build the exact 49-frame z=0.9 zigzag used by the RGB baseline video."""

import json
import math
from pathlib import Path

import numpy as np


SCENE = Path("/workspace/data/objaverse_245_eval_2500_2999/unseen_7x7x5_power06_png_exclude_requested/scenes/scene_002583")
OUT = Path("/workspace/outputs/infer/relighting_external/scene_002583/height_0.9_49")


def main() -> None:
    meta = json.loads((SCENE / "meta.json").read_text())
    samples = {tuple(s["light"]["grid_cell"]): s for s in meta["samples"]}
    cells = []
    for xi in range(7):
        ys = range(7) if xi % 2 == 0 else range(6, -1, -1)
        cells.extend((xi, yi, 4) for yi in ys)
    if any(cell not in samples for cell in cells):
        raise RuntimeError("The z=0.9 grid is not complete")

    camera = meta["camera"]
    transform = camera["similarity_transform"]
    rotation = np.asarray(transform["rotation_matrix"], dtype=np.float64)
    center = np.asarray(transform["target_center"], dtype=np.float64)
    camera_world = np.asarray(camera["location"], dtype=np.float64)
    right, forward, up = rotation[:, 0], rotation[:, 1], rotation[:, 2]
    pivot_forward = float(np.dot(center - camera_world, forward))
    focal = 0.5 / math.tan(math.radians(float(camera["fov_degrees"])) / 2)

    frames = []
    trajectory = []
    for index, cell in enumerate(cells):
        sample = samples[cell]
        xyz = np.asarray(sample["light"]["canonical_position"], dtype=np.float64)
        world = np.asarray(sample["light"]["world_position"], dtype=np.float64)
        delta = world - camera_world
        depth = float(np.dot(delta, forward))
        u = 0.5 + focal * float(np.dot(delta, right)) / depth
        v = 0.5 - focal * float(np.dot(delta, up)) / depth
        z_rel = depth / pivot_forward
        horizontal = math.hypot(float(xyz[0]), float(xyz[1]))
        radius = float(np.linalg.norm(xyz))
        azimuth = math.degrees(math.atan2(float(xyz[0]), -float(xyz[1]))) if horizontal else 0.0
        elevation = math.degrees(math.atan2(float(xyz[2]), horizontal))
        trajectory.append([f"{radius:.15g} {azimuth:.15g} {elevation:.15g}", "1", "75"])
        frames.append({
            "frame_index": index,
            "position_id": int(sample["light"]["id"]),
            "grid_cell": list(cell),
            "canonical_position": xyz.tolist(),
            "gt_path": str((SCENE / sample["image"]).resolve()),
            "genlit": {"radius": radius, "azimuth_deg": azimuth, "elevation_deg": elevation,
                       "in_nominal_domain": 0.8 <= radius <= 1.5 and 45 <= elevation <= 80},
            "livelight": {"u": u, "v": v, "z_rel": z_rel, "intensity": 1.0,
                          "color": [1.0, 1.0, 1.0]},
        })

    OUT.mkdir(parents=True, exist_ok=True)
    rows = np.asarray(trajectory, dtype=np.str_)
    # GenLit multi is fixed at 25 frames. Pad only the final frame of chunk 2.
    np.save(OUT / "genlit_chunk_00.npy", rows[:25][None])
    chunk_01 = np.concatenate([rows[25:], rows[-1:]], axis=0)
    np.save(OUT / "genlit_chunk_01.npy", chunk_01[None])
    # Training-aligned multi clips: seven dimming/ramp frames followed by at
    # most eighteen evaluated target positions. Warm-up outputs are discarded.
    official_chunks = []
    target_groups = [rows[i : i + 18] for i in range(0, len(rows), 18)]
    env_ramp = np.linspace(1.0, 0.2, 7)
    point_ramp = np.linspace(0.0, 75.0, 7)
    for chunk_index, targets in enumerate(target_groups):
        first_coord = targets[0, 0]
        warmup = np.asarray(
            [[first_coord, f"{env:.15g}", f"{point:.15g}"] for env, point in zip(env_ramp, point_ramp)],
            dtype=np.str_,
        )
        steady = targets.copy()
        steady[:, 1] = "0.2"
        steady[:, 2] = "75"
        clip = np.concatenate([warmup, steady], axis=0)
        valid_targets = len(targets)
        if len(clip) < 25:
            clip = np.concatenate([clip, np.repeat(clip[-1:], 25 - len(clip), axis=0)], axis=0)
        path = OUT / f"genlit_official_chunk_{chunk_index:02d}.npy"
        np.save(path, clip[None])
        official_chunks.append({
            "path": str(path.resolve()), "warmup_frames": 7,
            "valid_target_frames": valid_targets, "padding_frames": 18 - valid_targets,
        })
    manifest = {
        "schema": "external_relighting_height09_v1",
        "scene_id": meta["scene_id"],
        "source_path": str((SCENE / "source.png").resolve()),
        "frame_count": 49,
        "fps": 6,
        "ordering": "x-major serpentine at grid z index 4 (canonical z=0.9)",
        "genlit_chunks": [
            {"path": str((OUT / "genlit_chunk_00.npy").resolve()), "valid_frames": 25},
            {"path": str((OUT / "genlit_chunk_01.npy").resolve()), "valid_frames": 24, "padding_frames": 1},
        ],
        "genlit_training_aligned_chunks": official_chunks,
        "frames": frames,
    }
    (OUT / "trajectory_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "ids": [f["position_id"] for f in frames],
                      "genlit_in_domain": sum(f["genlit"]["in_nominal_domain"] for f in frames)}, indent=2))


if __name__ == "__main__":
    main()
