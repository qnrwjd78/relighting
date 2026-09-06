#!/usr/bin/env python3
"""Create three training-aligned GenLit clips forming one dense 54-frame circle."""

import json
from pathlib import Path

import numpy as np


OUT = Path(
    "/workspace/outputs/infer/relighting_external/scene_002583/"
    "genlit_dense_circle_r1.2_e60_54"
)


def main() -> None:
    radius = 1.2
    elevation = 60.0
    frame_count = 54
    environment_ramp = np.linspace(1.0, 0.2, 7)
    point_ramp = np.linspace(0.0, 75.0, 7)
    azimuths = np.linspace(0.0, 360.0, frame_count, endpoint=False)

    OUT.mkdir(parents=True, exist_ok=True)
    chunks = []
    for chunk_index in range(3):
        start = chunk_index * 18
        target_azimuths = azimuths[start : start + 18]
        first_position = f"{radius:.15g} {target_azimuths[0]:.15g} {elevation:.15g}"
        warmup = [
            [first_position, f"{env:.15g}", f"{point:.15g}"]
            for env, point in zip(environment_ramp, point_ramp)
        ]
        targets = [
            [f"{radius:.15g} {azimuth:.15g} {elevation:.15g}", "0.2", "75"]
            for azimuth in target_azimuths
        ]
        trajectory = np.asarray([warmup + targets], dtype=np.str_)
        if trajectory.shape != (1, 25, 3):
            raise RuntimeError(f"Unexpected chunk shape: {trajectory.shape}")
        path = OUT / f"trajectory_chunk_{chunk_index:02d}.npy"
        np.save(path, trajectory)
        chunks.append(
            {
                "chunk_index": chunk_index,
                "path": str(path.resolve()),
                "warmup_frames": 7,
                "target_frames": 18,
                "target_global_range": [start, start + 17],
            }
        )

    manifest = {
        "schema": "genlit_native_dense_circle_v1",
        "scene_id": "scene_002583",
        "frame_count": frame_count,
        "fps": 6,
        "model_mode": "multi",
        "one_gpu_sequential": True,
        "trajectory": {
            "radius": radius,
            "elevation_deg": elevation,
            "azimuth_deg": azimuths.tolist(),
            "azimuth_step_deg": float(360.0 / frame_count),
            "all_in_nominal_domain": True,
        },
        "steady_intensity": {"environment": 0.2, "point": 75.0},
        "chunks": chunks,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
