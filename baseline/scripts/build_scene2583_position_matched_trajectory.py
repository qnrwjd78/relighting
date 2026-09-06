#!/usr/bin/env python3
"""Build the shared scene_002583 trajectory for GenLit-multi and LiveLight.

The script only reads the extracted evaluation scene and writes derived artifacts
under the position-matched output directory.  Dataset candidate IDs are retained;
they are not renumbered after rejected candidates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_SCENE = Path(
    "/workspace/data/objaverse_245_eval_2500_2999/"
    "unseen_7x7x5_power06_png_exclude_requested/scenes/scene_002583"
)
DEFAULT_OUTPUT = Path(
    "/workspace/outputs/infer/relighting_external/scene_002583/position_matched"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--render-width", type=int, default=512)
    parser.add_argument("--render-height", type=int, default=512)
    parser.add_argument("--genlit-environment-intensity", type=float, default=1.0)
    parser.add_argument("--genlit-point-intensity", type=float, default=75.0)
    parser.add_argument("--livelight-intensity", type=float, default=1.0)
    return parser.parse_args()


def as_float_list(array: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(array).reshape(-1)]


def display_luminance(path: Path) -> dict[str, float]:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    lum = rgb @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return {
        "mean": float(lum.mean()),
        "p50": float(np.percentile(lum, 50)),
        "p95": float(np.percentile(lum, 95)),
        "p99": float(np.percentile(lum, 99)),
    }


def main() -> None:
    args = parse_args()
    meta_path = args.scene_dir / "meta.json"
    source_path = args.scene_dir / "source.png"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata["scene_id"] != "scene_002583":
        raise ValueError(f"Unexpected scene: {metadata['scene_id']}")

    # A constant-height 5x5 path. Alternating y order makes every step one grid
    # cell long, including row transitions. The exact candidate IDs contain gaps
    # elsewhere in the dataset, so samples are resolved by grid_cell, never ordinal.
    selected_cells: list[tuple[int, int, int]] = []
    for x_index in range(1, 6):
        y_indices = range(1, 6) if (x_index - 1) % 2 == 0 else range(5, 0, -1)
        selected_cells.extend((x_index, y_index, 4) for y_index in y_indices)
    if len(selected_cells) != 25 or len(set(selected_cells)) != 25:
        raise RuntimeError("Trajectory selection must contain 25 unique cells")

    sample_by_cell = {
        tuple(int(value) for value in sample["light"]["grid_cell"]): sample
        for sample in metadata["samples"]
        if sample.get("task") == "position"
    }
    missing_cells = [cell for cell in selected_cells if cell not in sample_by_cell]
    if missing_cells:
        raise FileNotFoundError(f"Selected cells were rejected or are missing: {missing_cells}")

    camera = metadata["camera"]
    transform = camera["similarity_transform"]
    rotation = np.asarray(transform["rotation_matrix"], dtype=np.float64)
    scale = float(transform["scale"])
    target_center = np.asarray(transform["target_center"], dtype=np.float64)
    camera_world = np.asarray(camera["location"], dtype=np.float64)
    canonical_camera = np.asarray(camera["canonical_position"], dtype=np.float64)
    canonical_pivot = np.asarray(transform["canonical_center"], dtype=np.float64)
    if not np.allclose(canonical_camera, [0.0, -3.5, 0.0], atol=1e-7):
        raise ValueError(f"Unexpected canonical camera: {canonical_camera}")
    if not np.allclose(canonical_pivot, [0.0, 0.0, 0.0], atol=1e-7):
        raise ValueError(f"Unexpected canonical pivot: {canonical_pivot}")

    # R columns are canonical camera-right, camera-forward, and camera-up axes.
    right_axis = rotation[:, 0]
    forward_axis = rotation[:, 1]
    up_axis = rotation[:, 2]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-6):
        raise ValueError("Similarity rotation is not orthonormal")
    expected_camera_world = target_center + scale * rotation @ canonical_camera
    camera_reconstruction_error = float(np.linalg.norm(expected_camera_world - camera_world))
    if camera_reconstruction_error > 2e-6:
        raise ValueError(f"Camera reconstruction error is too large: {camera_reconstruction_error}")

    fov_degrees = float(camera["fov_degrees"])
    focal_scale = 0.5 / math.tan(math.radians(fov_degrees) * 0.5)
    fx = float(args.render_width) * focal_scale
    fy = float(args.render_height) * focal_scale
    cx = (float(args.render_width) - 1.0) * 0.5
    cy = (float(args.render_height) - 1.0) * 0.5
    pivot_forward_world = float(np.dot(target_center - camera_world, forward_axis))

    frames: list[dict] = []
    trajectory_rows: list[list[str]] = []
    world_reconstruction_errors: list[float] = []
    camera_coordinate_errors: list[float] = []
    selected_positions: list[np.ndarray] = []
    for frame_index, cell in enumerate(selected_cells):
        sample = sample_by_cell[cell]
        light = sample["light"]
        position_id = int(light["id"])
        expected_id = ((cell[0] * 7) + cell[1]) * 5 + cell[2]
        if position_id != expected_id:
            raise ValueError(
                f"Candidate ID mismatch for cell {cell}: {position_id} != {expected_id}"
            )
        position_name = f"position_{position_id:03d}"
        if sample["name"] != position_name:
            raise ValueError(f"Sample name mismatch: {sample['name']} != {position_name}")

        canonical = np.asarray(light["canonical_position"], dtype=np.float64)
        world = np.asarray(light["world_position"], dtype=np.float64)
        selected_positions.append(canonical)
        reconstructed_world = target_center + scale * rotation @ (canonical - canonical_pivot)
        world_error = float(np.linalg.norm(reconstructed_world - world))
        world_reconstruction_errors.append(world_error)

        delta_world = world - camera_world
        camera_coords_world = np.asarray(
            [
                np.dot(delta_world, forward_axis),
                np.dot(delta_world, right_axis),
                np.dot(delta_world, up_axis),
            ],
            dtype=np.float64,
        )
        camera_coords_expected = scale * np.asarray(
            [
                canonical[1] - canonical_camera[1],
                canonical[0] - canonical_camera[0],
                canonical[2] - canonical_camera[2],
            ],
            dtype=np.float64,
        )
        camera_error = float(np.linalg.norm(camera_coords_world - camera_coords_expected))
        camera_coordinate_errors.append(camera_error)
        forward, right, up = camera_coords_world
        if forward <= 0:
            raise ValueError(f"Light is behind the camera at frame {frame_index}")

        u_pixel = cx + fx * right / forward
        v_pixel = cy - fy * up / forward
        u = float(u_pixel / max(args.render_width - 1, 1))
        v = float(v_pixel / max(args.render_height - 1, 1))
        z_rel = float(forward / pivot_forward_world)

        horizontal = float(math.hypot(canonical[0], canonical[1]))
        radius = float(np.linalg.norm(canonical - canonical_pivot))
        if horizontal < 1e-10:
            azimuth = 0.0
            azimuth_defined = False
        else:
            # Camera-relative convention: azimuth 0 is toward the camera (-Y),
            # positive angles rotate toward image-right (+X).
            azimuth = float(math.degrees(math.atan2(canonical[0], -canonical[1])))
            azimuth_defined = True
        elevation = float(math.degrees(math.atan2(canonical[2], horizontal)))
        in_domain = (0.8 <= radius <= 1.5) and (45.0 <= elevation <= 80.0)

        env_intensity = float(args.genlit_environment_intensity)
        point_intensity = float(args.genlit_point_intensity)
        coord_string = f"{radius:.15g} {azimuth:.15g} {elevation:.15g}"
        trajectory_rows.append([coord_string, f"{env_intensity:.15g}", f"{point_intensity:.15g}"])

        gt_path = (args.scene_dir / sample["image"]).resolve()
        if not gt_path.is_file():
            raise FileNotFoundError(gt_path)
        frames.append(
            {
                "frame_index": frame_index,
                "position_id": position_id,
                "position_name": position_name,
                "grid_cell": list(cell),
                "canonical_position": as_float_list(canonical),
                "world_position": as_float_list(world),
                "camera_position": {
                    "convention": "[forward, right, up] in world-length units",
                    "forward_right_up": as_float_list(camera_coords_world),
                },
                "livelight": {
                    "u": u,
                    "v": v,
                    "z_rel": z_rel,
                    "u_pixel_at_512": float(u_pixel),
                    "v_pixel_at_512": float(v_pixel),
                    "intensity_provisional": float(args.livelight_intensity),
                    "color": [1.0, 1.0, 1.0],
                    "in_normalized_image_bounds": bool(0.0 <= u <= 1.0 and 0.0 <= v <= 1.0),
                },
                "genlit": {
                    "radius": radius,
                    "azimuth_deg": azimuth,
                    "azimuth_defined": azimuth_defined,
                    "elevation_deg": elevation,
                    "environment_intensity": env_intensity,
                    "point_intensity": point_intensity,
                    "checkpoint_domain": {
                        "radius_range": [0.8, 1.5],
                        "elevation_deg_range": [45.0, 80.0],
                        "in_domain": in_domain,
                    },
                    "loader_normalized_5d": [
                        0.1 + ((radius - 0.8) / (1.5 - 0.8)) * 0.8,
                        0.1 + ((azimuth % 360.0) / 360.0) * 0.8,
                        elevation / 90.0,
                        env_intensity,
                        point_intensity / 75.0,
                    ],
                },
                "dataset_light": {
                    "render_world_energy": float(light["render_world_energy"]),
                    "world_energy_before_power_scale": float(light["world_energy"]),
                    "canonical_energy_before_power_scale": float(light["canonical_energy"]),
                    "power_scale": float(light["power_scale"]),
                    "effective_canonical_energy": float(
                        light["canonical_energy"] * light["power_scale"]
                    ),
                    "world_radius": float(light["world_radius"]),
                    "render_color": [float(value) for value in light["render_color"]],
                },
                "gt_path": str(gt_path),
                "gt_display_luminance": display_luminance(gt_path),
            }
        )

    trajectory = np.asarray([trajectory_rows], dtype=np.str_)
    if trajectory.shape != (1, 25, 3):
        raise RuntimeError(f"Unexpected GenLit trajectory shape: {trajectory.shape}")

    selected_positions_array = np.stack(selected_positions)
    steps = np.linalg.norm(np.diff(selected_positions_array, axis=0), axis=1)
    genlit_in_domain_count = sum(
        frame["genlit"]["checkpoint_domain"]["in_domain"] for frame in frames
    )
    normalized_uv = np.asarray(
        [[frame["livelight"]["u"], frame["livelight"]["v"]] for frame in frames]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = args.output_dir / "genlit_multi_scene002583_positions.npy"
    manifest_path = args.output_dir / "trajectory_manifest.json"
    np.save(trajectory_path, trajectory)

    manifest = {
        "schema": "scene_position_matched_trajectory_v1",
        "scene_id": metadata["scene_id"],
        "source_path": str(source_path.resolve()),
        "meta_path": str(meta_path.resolve()),
        "scene_dir": str(args.scene_dir.resolve()),
        "genlit_trajectory_path": str(trajectory_path.resolve()),
        "frame_count": 25,
        "position_ids_in_order": [frame["position_id"] for frame in frames],
        "selection": {
            "description": "z=0.9, central 5x5 canonical grid, x-major serpentine",
            "x_grid_indices": [1, 2, 3, 4, 5],
            "y_grid_indices": [1, 2, 3, 4, 5],
            "z_grid_index": 4,
            "unique_positions": True,
            "all_dataset_candidates_accepted": True,
            "canonical_step_distance": {
                "min": float(steps.min()),
                "max": float(steps.max()),
                "mean": float(steps.mean()),
            },
            "genlit_multi_domain_count": genlit_in_domain_count,
            "genlit_multi_out_of_domain_frames": [
                frame["frame_index"]
                for frame in frames
                if not frame["genlit"]["checkpoint_domain"]["in_domain"]
            ],
            "out_of_domain_reason": (
                "The grid center is at elevation 90 degrees; its azimuth is undefined and set to 0. "
                "The other 24 frames are inside public multi trajectory radius/elevation ranges."
            ),
        },
        "coordinate_systems": {
            "pivot": {
                "choice": "dataset canonical-rig origin / camera look-at target",
                "canonical": as_float_list(canonical_pivot),
                "world": as_float_list(target_center),
                "not_object_bbox_center_world": as_float_list(
                    np.asarray(metadata["object"]["center"], dtype=np.float64)
                ),
            },
            "canonical_to_world": {
                "formula": "p_world = target_center + scale * R @ (p_canonical - canonical_pivot)",
                "scale": scale,
                "rotation_matrix": rotation.tolist(),
                "max_validation_error": float(max(world_reconstruction_errors)),
            },
            "camera": {
                "world_location": as_float_list(camera_world),
                "canonical_location": as_float_list(canonical_camera),
                "right_axis_world": as_float_list(right_axis),
                "forward_axis_world": as_float_list(forward_axis),
                "up_axis_world": as_float_list(up_axis),
                "camera_reconstruction_error": camera_reconstruction_error,
                "max_frame_coordinate_error": float(max(camera_coordinate_errors)),
                "pivot_forward_world": pivot_forward_world,
            },
            "livelight_projection": {
                "formula": {
                    "u_px": "cx + fx * right / forward",
                    "v_px": "cy - fy * up / forward",
                    "u": "u_px / (width - 1)",
                    "v": "v_px / (height - 1)",
                    "z_rel": "light_forward / pivot_forward",
                },
                "width": args.render_width,
                "height": args.render_height,
                "fov_degrees": fov_degrees,
                "focal_scale": focal_scale,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "u_range": [float(normalized_uv[:, 0].min()), float(normalized_uv[:, 0].max())],
                "v_range": [float(normalized_uv[:, 1].min()), float(normalized_uv[:, 1].max())],
                "all_in_image_bounds": bool(
                    np.logical_and(normalized_uv >= 0.0, normalized_uv <= 1.0).all()
                ),
                "metric_scale_caveat": (
                    "z_rel preserves the dataset rig ratio. The prepared MoGe depth uses a center-surface "
                    "anchor of 256, so LiveLight remains a relative-depth approximation, not Blender metric depth."
                ),
            },
            "genlit_polar": {
                "pivot": "canonical rig origin",
                "radius": "norm([x,y,z])",
                "azimuth_deg": "degrees(atan2(x, -y)); 0=toward camera, positive=toward image right",
                "elevation_deg": "degrees(atan2(z, hypot(x,y)))",
                "center_azimuth_rule": "set to 0 when hypot(x,y)=0 because azimuth is undefined at zenith",
            },
        },
        "intensity_strategy": {
            "dataset": {
                "ambient": {
                    "type": metadata["source"]["ambient_source"]["type"],
                    "path": metadata["source"]["ambient_source"]["path"],
                    "strength": float(metadata["source"]["ambient_source"]["strength"]),
                    "unchanged_between_source_and_gt": True,
                },
                "point": {
                    "base_energy": float(metadata["sampling"]["base_energy"]),
                    "power_scale": float(metadata["sampling"]["power_values"][0]),
                    "effective_canonical_energy": float(
                        metadata["sampling"]["base_energy"]
                        * metadata["sampling"]["power_values"][0]
                    ),
                    "render_world_energy": float(frames[0]["dataset_light"]["render_world_energy"]),
                    "color": [1.0, 1.0, 1.0],
                },
                "png_encoding": "reinhard_gamma, gamma=2.2",
            },
            "genlit": {
                "selected": {
                    "environment_intensity": float(args.genlit_environment_intensity),
                    "point_intensity": float(args.genlit_point_intensity),
                    "constant_for_all_25_frames": True,
                },
                "reason": (
                    "GT keeps the same HDRI at full strength, so environment intensity stays 1.0. "
                    "Dataset point energy exceeds the public multi trajectory's 75-unit maximum, so point "
                    "intensity is clamped to 75. This prioritizes per-position GT alignment over the official "
                    "six-frame fade schedule."
                ),
                "checkpoint_semantics": (
                    "The shipped multi trajectories hold position fixed while environment ramps 1.0->0.2 "
                    "and point ramps 0->75 over frames 0..6, then move the light with [0.2,75]."
                ),
                "caveat": (
                    "[environment=1.0, point=75] is outside the shipped trajectory's correlated intensity "
                    "schedule, and Blender energy units are not guaranteed to equal GenLit intensity units."
                ),
                "optional_global_point_candidates": [25.0, 50.0, 75.0],
            },
            "livelight": {
                "selected_provisional": float(args.livelight_intensity),
                "color": [1.0, 1.0, 1.0],
                "optional_global_candidates": [0.25, 0.5, 1.0, 2.0, 4.0],
                "caveat": "MPLI intensity is learned/control-space scale, not Blender point-light energy.",
            },
            "calibration_policy": (
                "If candidate inference is affordable, choose one intensity globally per method across all "
                "25 frames using the same object+receiver evaluation region. Never fit a separate exposure "
                "per frame, because that leaks GT and causes temporal flicker."
            ),
        },
        "source_display_luminance": display_luminance(source_path),
        "frames": frames,
        "validation": {
            "npy_shape": list(trajectory.shape),
            "npy_dtype": str(trajectory.dtype),
            "all_gt_files_exist": True,
            "all_position_ids_match_candidate_formula": True,
            "all_world_positions_match_similarity_transform": bool(
                max(world_reconstruction_errors) < 2e-6
            ),
            "all_camera_coordinates_match_canonical_rig": bool(
                max(camera_coordinate_errors) < 2e-6
            ),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Reload both artifacts so a serialization error cannot silently pass validation.
    reloaded_trajectory = np.load(trajectory_path, allow_pickle=False)
    reloaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if reloaded_trajectory.shape != (1, 25, 3):
        raise RuntimeError(f"Reloaded trajectory has wrong shape: {reloaded_trajectory.shape}")
    if len(reloaded_manifest["frames"]) != 25:
        raise RuntimeError("Reloaded manifest does not contain 25 frames")

    print(json.dumps({
        "manifest": str(manifest_path.resolve()),
        "trajectory": str(trajectory_path.resolve()),
        "trajectory_shape": list(reloaded_trajectory.shape),
        "position_ids": reloaded_manifest["position_ids_in_order"],
        "genlit_in_domain": f"{genlit_in_domain_count}/25",
        "max_world_error": max(world_reconstruction_errors),
        "max_camera_error": max(camera_coordinate_errors),
    }, indent=2))


if __name__ == "__main__":
    main()
