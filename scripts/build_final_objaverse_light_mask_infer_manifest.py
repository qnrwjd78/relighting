#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data/final_objaverse_light_mask_unseen_480_png"
DEFAULT_OUTPUT = ROOT / "data_train/final_objaverse_light_mask_unseen_480_infer/metadata.jsonl"
DEFAULT_PROMPT = "photorealistic object relighting, preserve geometry and materials"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a sample-level TokenLight inference JSONL from a final Objaverse light-mask dataset."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dataset-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--tasks",
        default="position,color,power",
        help="Comma-separated sample tasks to include.",
    )
    parser.add_argument(
        "--no-check-files",
        action="store_true",
        help="Do not verify that every referenced image exists.",
    )
    return parser.parse_args()


def finite_float(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number, got {value!r}")
    return result


def triple(value: Any, *, field: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field} must contain exactly three numbers, got {value!r}")
    return [finite_float(item, field=f"{field}[{index}]") for index, item in enumerate(value)]


def nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}") from exc
    if result < 0 or result != float(value):
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}")
    return result


def integer_triple(value: Any, *, field: str, positive: bool = False) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field} must contain exactly three integers, got {value!r}")
    result = [nonnegative_int(item, field=f"{field}[{index}]") for index, item in enumerate(value)]
    if positive and any(item == 0 for item in result):
        raise ValueError(f"{field} entries must be positive, got {value!r}")
    return result


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def dataset_relative(scene_dir: Path, value: Any, *, data_root: Path, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path, got {value!r}")
    path = scene_dir / value
    try:
        return path.relative_to(data_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"{field} resolves outside the dataset root: {path}") from exc


def position_samples(samples: list[dict[str, Any]]) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    by_index: dict[int, dict[str, Any]] = {}
    by_light_id: dict[int, dict[str, Any]] = {}
    position_index = 0
    for sample in samples:
        if sample.get("task") != "position":
            continue
        by_index[position_index] = sample
        position_index += 1
        light = sample.get("light")
        if isinstance(light, dict) and light.get("id") is not None:
            by_light_id[int(light["id"])] = sample
    return by_index, by_light_id


def anchor_sample(
    sample: dict[str, Any],
    *,
    by_index: dict[int, dict[str, Any]],
    by_light_id: dict[int, dict[str, Any]],
    context: str,
) -> dict[str, Any]:
    index = sample.get("anchor_position_index")
    if index is not None and int(index) in by_index:
        return by_index[int(index)]
    light_id = sample.get("anchor_position_light_id")
    if light_id is not None and int(light_id) in by_light_id:
        return by_light_id[int(light_id)]
    raise ValueError(f"Could not resolve the anchor position for {context}")


def attrs_for_sample(
    sample: dict[str, Any],
    *,
    ambient_strength: float,
    default_power_scale: float,
    by_index: dict[int, dict[str, Any]],
    by_light_id: dict[int, dict[str, Any]],
    context: str,
) -> tuple[dict[str, Any], int | None]:
    task = str(sample.get("task"))
    if task == "position":
        position_sample = sample
        light = sample.get("light", {})
        power_scale = light.get("power_scale", light.get("default_power_scale", default_power_scale))
        color_value = sample.get("light", {}).get("component_color", [1.0, 1.0, 1.0])
    elif task in {"color", "power"}:
        position_sample = anchor_sample(
            sample,
            by_index=by_index,
            by_light_id=by_light_id,
            context=context,
        )
        power_scale = sample.get("power_scale", default_power_scale)
        color_value = sample.get("color", [1.0, 1.0, 1.0]) if task == "color" else [1.0, 1.0, 1.0]
    else:
        raise ValueError(f"Unsupported sample task {task!r} in {context}")

    light = position_sample.get("light")
    if not isinstance(light, dict):
        raise ValueError(f"Missing position light metadata in {context}")
    x, y, z = triple(light.get("canonical_position"), field=f"{context}.canonical_position")
    r, g, b = triple(color_value, field=f"{context}.color")
    light_attrs = {
        "x": x,
        "y": y,
        "z": z,
        "r": r,
        "g": g,
        "b": b,
        "lambda": finite_float(power_scale, field=f"{context}.power_scale"),
        "d": finite_float(light.get("canonical_radius"), field=f"{context}.canonical_radius"),
    }
    attrs = {"a": ambient_strength, "dg": 0.0, "lights": [light_attrs], "t": 1.0}
    original_light_id = int(light["id"]) if light.get("id") is not None else None
    return attrs, original_light_id


def require_files(rows: list[dict[str, Any]], data_root: Path) -> None:
    path_keys = (
        "input_image",
        "video",
        "mask",
        "inf_mask",
        "shadow_mask",
        "shadow_mask_pad16",
        "pbr_albedo_image",
        "pbr_depth_image",
        "pbr_normal_image",
        "pbr_roughness_image",
    )
    missing: list[str] = []
    for row in rows:
        for key in path_keys:
            value = row.get(key)
            if value and not (data_root / str(value)).is_file():
                missing.append(f"row {row['manifest_index']} {key}: {value}")
                if len(missing) >= 20:
                    break
        if len(missing) >= 20:
            break
    if missing:
        details = "\n".join(missing)
        raise FileNotFoundError(f"Referenced dataset files are missing (first {len(missing)}):\n{details}")


def build_rows(data_root: Path, dataset_manifest: Path, prompt: str, tasks: set[str]) -> list[dict[str, Any]]:
    if dataset_manifest.is_file():
        manifest = load_json(dataset_manifest)
        scene_entries = manifest.get("scenes")
        if not isinstance(scene_entries, list):
            raise ValueError(f"Missing scenes list in {dataset_manifest}")
    else:
        scene_entries = [
            {"scene_id": path.parent.name, "meta": path.relative_to(data_root).as_posix()}
            for path in sorted((data_root / "scenes").glob("*/meta.json"))
        ]
        if not scene_entries:
            raise FileNotFoundError(
                f"Missing {dataset_manifest} and no scene metadata found under {data_root / 'scenes'}"
            )

    rows: list[dict[str, Any]] = []
    for scene_entry in scene_entries:
        if not isinstance(scene_entry, dict):
            raise ValueError(f"Invalid scene entry in {dataset_manifest}: {scene_entry!r}")
        meta_value = scene_entry.get("meta")
        if not isinstance(meta_value, str) or not meta_value:
            raise ValueError(f"Scene entry is missing meta: {scene_entry!r}")
        meta_path = data_root / meta_value
        meta = load_json(meta_path)
        scene_dir = meta_path.parent
        scene_id = str(meta.get("scene_id") or scene_entry.get("scene_id") or scene_dir.name)
        samples = meta.get("samples")
        if not isinstance(samples, list):
            raise ValueError(f"Missing samples list in {meta_path}")
        if not all(isinstance(sample, dict) for sample in samples):
            raise ValueError(f"Every sample must be an object in {meta_path}")

        common = meta.get("common", {})
        pbr_maps = common.get("pbr_maps", {})
        source = meta.get("source", {})
        ambient_source = source.get("ambient_source", {})
        sampling = meta.get("sampling", {})
        ambient_strength = finite_float(
            ambient_source.get("strength"), field=f"{scene_id}.source.ambient_source.strength"
        )
        power_values = sampling.get("power_values")
        fallback_power = power_values[0] if isinstance(power_values, list) and power_values else 1.0
        default_power_scale = finite_float(
            sampling.get("default_power_scale", fallback_power),
            field=f"{scene_id}.sampling.default_power_scale",
        )
        source_path = source.get("ambient_only", {}).get("render", "source.png")
        object_mask = common.get("object_mask")
        positions_by_index, positions_by_light_id = position_samples(samples)

        for sample_index, sample in enumerate(samples):
            task = str(sample.get("task"))
            if task not in tasks:
                continue
            context = f"{scene_id}.{sample.get('name', sample_index)}"
            attrs, original_light_id = attrs_for_sample(
                sample,
                ambient_strength=ambient_strength,
                default_power_scale=default_power_scale,
                by_index=positions_by_index,
                by_light_id=positions_by_light_id,
                context=context,
            )
            masks = sample.get("masks", {})
            row = {
                "attrs_json": json.dumps(attrs, ensure_ascii=False, separators=(",", ":")),
                "input_image": dataset_relative(
                    scene_dir, source_path, data_root=data_root, field=f"{context}.input_image"
                ),
                "light_id": sample_index,
                "manifest_index": len(rows),
                "mask": dataset_relative(
                    scene_dir, object_mask, data_root=data_root, field=f"{context}.mask"
                ),
                "original_light_id": original_light_id,
                "pbr_albedo_image": dataset_relative(
                    scene_dir, pbr_maps.get("albedo"), data_root=data_root, field=f"{context}.pbr_albedo"
                ),
                "pbr_depth_image": dataset_relative(
                    scene_dir, pbr_maps.get("depth"), data_root=data_root, field=f"{context}.pbr_depth"
                ),
                "pbr_normal_image": dataset_relative(
                    scene_dir, pbr_maps.get("normal"), data_root=data_root, field=f"{context}.pbr_normal"
                ),
                "pbr_roughness_image": dataset_relative(
                    scene_dir, pbr_maps.get("roughness"), data_root=data_root, field=f"{context}.pbr_roughness"
                ),
                "prompt": prompt,
                "sample_name": str(sample.get("name") or f"sample_{sample_index:03d}"),
                "scene_folder": scene_dir.name,
                "scene_id": scene_id,
                "task": task,
                "valid": True,
                "video": dataset_relative(
                    scene_dir, sample.get("image"), data_root=data_root, field=f"{context}.video"
                ),
            }

            # Preserve optional grid/variant metadata used by fixed-grid
            # inference visualizations.  Older datasets do not contain these
            # fields and retain their previous manifest schema unchanged.
            position_sample = sample
            if task in {"color", "power"}:
                position_sample = anchor_sample(
                    sample,
                    by_index=positions_by_index,
                    by_light_id=positions_by_light_id,
                    context=context,
                )
            position_light = position_sample.get("light", {})
            if isinstance(position_light, dict):
                variant_value = position_light.get("light_variant_id")
                grid_cell_value = position_light.get("grid_cell")
                grid_resolution_value = position_light.get("grid_resolution")
                if variant_value is not None:
                    row["light_variant_id"] = nonnegative_int(
                        variant_value, field=f"{context}.light_variant_id"
                    )
                if grid_cell_value is not None:
                    row["grid_cell"] = integer_triple(grid_cell_value, field=f"{context}.grid_cell")
                if grid_resolution_value is not None:
                    row["grid_resolution"] = integer_triple(
                        grid_resolution_value,
                        field=f"{context}.grid_resolution",
                        positive=True,
                    )
                if grid_cell_value is not None and grid_resolution_value is not None:
                    if any(cell >= resolution for cell, resolution in zip(row["grid_cell"], row["grid_resolution"])):
                        raise ValueError(
                            f"{context}.grid_cell {row['grid_cell']} falls outside "
                            f"grid_resolution {row['grid_resolution']}"
                        )
                candidate_source = position_light.get("candidate_source")
                if isinstance(candidate_source, str) and candidate_source:
                    row["light_candidate_source"] = candidate_source
            row["power_scale"] = float(attrs["lights"][0]["lambda"])
            direct_lit = masks.get("object_direct_lit_clean")
            shadow = masks.get("object_shadow_clean")
            shadow_pad = masks.get("object_shadow_clean_pad16")
            if direct_lit:
                row["inf_mask"] = dataset_relative(
                    scene_dir, direct_lit, data_root=data_root, field=f"{context}.inf_mask"
                )
            if shadow:
                row["shadow_mask"] = dataset_relative(
                    scene_dir, shadow, data_root=data_root, field=f"{context}.shadow_mask"
                )
            if shadow_pad:
                row["shadow_mask_pad16"] = dataset_relative(
                    scene_dir, shadow_pad, data_root=data_root, field=f"{context}.shadow_mask_pad16"
                )
            rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    dataset_manifest = (
        args.dataset_manifest.expanduser().resolve()
        if args.dataset_manifest is not None
        else data_root / "dataset_manifest.json"
    )
    output = args.output.expanduser().resolve()
    tasks = {item.strip() for item in str(args.tasks).split(",") if item.strip()}
    supported_tasks = {"position", "color", "power"}
    if not tasks or not tasks <= supported_tasks:
        raise ValueError(f"--tasks must be a non-empty subset of {sorted(supported_tasks)}, got {sorted(tasks)}")

    rows = build_rows(data_root, dataset_manifest, args.prompt, tasks)
    if not args.no_check_files:
        require_files(rows, data_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

    task_counts = {task: sum(row["task"] == task for row in rows) for task in sorted(tasks)}
    print(f"Wrote {len(rows)} rows to {output}")
    print(f"Data root: {data_root}")
    print(f"Task counts: {json.dumps(task_counts, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
