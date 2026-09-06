#!/usr/bin/env python3
"""Read-only preflight validation for the shadow C2F relighting pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_shadowadapter_cache import (  # noqa: E402
    CacheError,
    official_asset_paths,
    validate_official_assets,
)
from utils.shadow_c2f_dataset import (  # noqa: E402
    ADAPTER_DELTA_CACHE_SCHEMA,
    ADAPTER_IMAGE_CACHE_SCHEMA,
    GEOMETRY_CACHE_SCHEMA,
    PHYSICS_CACHE_SCHEMA,
)
from utils.shadow_pipeline_io import read_jsonl, resolve_path, sample_key  # noqa: E402


REPORT_SCHEMA = "tokenlight_shadow_c2f_preflight_v1"
REQUIRED_ROW_KEYS = (
    "source_image",
    "baseline_image",
    "object_mask",
    "receiver_mask",
    "gt_shadow_mask",
    "geometry_cache",
)
OPTIONAL_ROW_KEYS = (
    "point_map",
    "adapter_source_cache",
    "adapter_target_cache",
    "adapter_delta_cache",
    "physics_cache",
    "predicted_shadow_mask",
)
BANNED_KEYS = ("shadow_mask", "shadow_mask_pad16")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--adaptershadow-root", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path)
    parser.add_argument("--efficientnet-checkpoint", type=Path)
    parser.add_argument("--c2f-checkpoint", type=Path)
    parser.add_argument("--wan-checkpoint", type=Path)
    parser.add_argument("--base-path", type=Path, default=ROOT)
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser


def _scalar_text(value: Any) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"Expected scalar metadata, got {array.shape}")
    return str(array.item())


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_existing_row_path(
    row: dict[str, Any],
    key: str,
    *,
    base_path: Path,
) -> Path | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return resolve_path(str(value), base_path=base_path).resolve()


def _record_error(report: dict[str, Any], message: str) -> None:
    report["errors"].append(message)


def _record_warning(report: dict[str, Any], message: str) -> None:
    report["warnings"].append(message)


def _inspect_light_position(value: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"ok": False}
    if not isinstance(value, (list, tuple)):
        result["error"] = f"light_position must be a 3-vector, got {type(value).__name__}"
        return result
    if len(value) != 3:
        result["error"] = f"light_position must have length 3, got {len(value)}"
        return result
    try:
        components = [float(item) for item in value]
    except (TypeError, ValueError) as error:
        result["error"] = f"light_position is not numeric: {error}"
        return result
    if not np.isfinite(np.asarray(components, dtype=np.float64)).all():
        result["error"] = "light_position contains NaN or infinity"
        return result
    result["value"] = components
    result["ok"] = True
    return result


def _inspect_image_checkpoint(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.resolve().as_posix(),
        "ok": False,
    }
    if not _nonempty_file(path):
        result["error"] = f"Missing checkpoint file: {path}"
        return result
    result["size_bytes"] = int(path.stat().st_size)
    if path.suffix == ".safetensors":
        try:
            from safetensors import safe_open
        except Exception as error:
            result["warning"] = f"safetensors metadata unavailable: {error}"
            result["ok"] = True
            result["kind"] = "safetensors"
            return result
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
        result["kind"] = "safetensors"
        result["tensor_count"] = len(keys)
        result["example_tensors"] = keys[:8]
        result["ok"] = len(keys) > 0
        if not result["ok"]:
            result["error"] = f"No tensors found in {path}"
        return result
    payload = torch.load(path, map_location="cpu", weights_only=False)
    result["kind"] = path.suffix.lstrip(".") or "torch"
    if isinstance(payload, dict):
        result["top_level_keys"] = sorted(str(key) for key in payload.keys())[:16]
        tensors = 0
        stack = [payload]
        seen_ids: set[int] = set()
        while stack:
            current = stack.pop()
            if id(current) in seen_ids:
                continue
            seen_ids.add(id(current))
            if isinstance(current, dict):
                stack.extend(current.values())
            elif isinstance(current, (list, tuple)):
                stack.extend(current)
            elif torch.is_tensor(current):
                tensors += 1
        result["tensor_count"] = tensors
        result["ok"] = tensors > 0 or bool(payload)
        if not result["ok"]:
            result["error"] = f"Checkpoint dict is empty: {path}"
        return result
    result["top_level_type"] = type(payload).__name__
    result["ok"] = True
    return result


def _inspect_c2f_checkpoint(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.resolve().as_posix(),
        "ok": False,
    }
    if not _nonempty_file(path):
        result["error"] = f"Missing C2F checkpoint file: {path}"
        return result
    payload = torch.load(path, map_location="cpu", weights_only=False)
    result["size_bytes"] = int(path.stat().st_size)
    if not isinstance(payload, dict):
        result["error"] = f"C2F checkpoint is not a dict: {path}"
        return result
    result["schema"] = payload.get("schema")
    if payload.get("schema") != "tokenlight_shadow_c2f_checkpoint_v1":
        result["error"] = (
            "Unexpected C2F checkpoint schema: "
            f"{payload.get('schema')!r} (expected tokenlight_shadow_c2f_checkpoint_v1)"
        )
        return result
    model = payload.get("model")
    if not isinstance(model, dict) or not model:
        result["error"] = f"C2F checkpoint has no model weights: {path}"
        return result
    coarse_prior_mode = payload.get("coarse_prior_mode")
    if coarse_prior_mode not in {"adapter_delta", "adapter_target", "physics"}:
        result["error"] = (
            f"C2F checkpoint has invalid coarse_prior_mode: {coarse_prior_mode!r}"
        )
        return result
    result["coarse_prior_mode"] = coarse_prior_mode
    result["epoch"] = int(payload.get("epoch", 0))
    result["tensor_count"] = len(model)
    result["model_config_keys"] = sorted(
        str(key) for key in (payload.get("model_config") or {}).keys()
    )
    result["ok"] = True
    return result


def _inspect_adaptershadow(args: argparse.Namespace) -> dict[str, Any]:
    asset_args = argparse.Namespace(
        adaptershadow_root=args.adaptershadow_root,
        checkpoint=args.adapter_checkpoint,
        sam_checkpoint=args.sam_checkpoint,
        efficientnet_checkpoint=args.efficientnet_checkpoint,
    )
    assets = official_asset_paths(asset_args)
    result = {
        "root": assets.root.as_posix(),
        "checkpoint": assets.checkpoint.as_posix(),
        "sam_checkpoint": assets.sam_checkpoint.as_posix(),
        "efficientnet_checkpoint": assets.efficientnet_checkpoint.as_posix(),
        "ok": False,
    }
    try:
        validate_official_assets(assets)
    except CacheError as error:
        result["error"] = str(error)
        return result
    result["ok"] = True
    for key, path in (
        ("checkpoint_size_bytes", assets.checkpoint),
        ("sam_checkpoint_size_bytes", assets.sam_checkpoint),
        ("efficientnet_checkpoint_size_bytes", assets.efficientnet_checkpoint),
    ):
        result[key] = int(path.stat().st_size)
    return result


def _inspect_npz(
    path: Path,
    *,
    expected_schema: str,
    expected_fields: tuple[str, ...],
    exact_output_size: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.as_posix(),
        "ok": False,
    }
    if not _nonempty_file(path):
        result["error"] = f"Missing cache: {path}"
        return result
    try:
        with np.load(path, allow_pickle=False) as archive:
            fields = set(archive.files)
            missing_fields = [field for field in expected_fields if field not in fields]
            if missing_fields:
                result["error"] = f"{path} missing fields: {missing_fields}"
                return result
            try:
                schema = _scalar_text(archive["schema"])
            except Exception as error:
                result["error"] = f"{path} invalid schema metadata: {error}"
                return result
            result["schema"] = schema
            if schema != expected_schema:
                result["error"] = (
                    f"{path} schema {schema!r} does not match expected {expected_schema!r}"
                )
                return result
            if exact_output_size is not None:
                output_size = int(np.asarray(archive["output_size"]).item())
                result["output_size"] = output_size
                if output_size != exact_output_size:
                    result["error"] = (
                        f"{path} output_size={output_size} does not match expected "
                        f"{exact_output_size}"
                    )
                    return result
            result["ok"] = True
    except (OSError, ValueError) as error:
        result["error"] = f"{path} is not a readable NPZ cache: {error}"
    return result


def _inspect_geometry_cache(path: Path) -> dict[str, Any]:
    result = _inspect_npz(
        path,
        expected_schema=GEOMETRY_CACHE_SCHEMA,
        expected_fields=("schema", "cache_id", "point_features", "point_valid"),
    )
    if not result["ok"]:
        return result
    with np.load(path, allow_pickle=False) as archive:
        point_features = np.asarray(archive["point_features"])
        point_valid = np.asarray(archive["point_valid"])
        if point_features.ndim != 3 or point_features.shape[0] != 3:
            result["ok"] = False
            result["error"] = (
                f"{path} point_features must be [3,H,W], got {point_features.shape}"
            )
            return result
        if point_valid.shape != (1, *point_features.shape[-2:]):
            result["ok"] = False
            result["error"] = (
                f"{path} point_valid must be [1,H,W], got {point_valid.shape}"
            )
            return result
        result["cache_id"] = _scalar_text(archive["cache_id"])
        result["shape"] = list(point_features.shape)
    return result


def _inspect_physics_cache(path: Path) -> dict[str, Any]:
    result = _inspect_npz(
        path,
        expected_schema=PHYSICS_CACHE_SCHEMA,
        expected_fields=(
            "schema",
            "cache_id",
            "geometry_cache_id",
            "coarse_size",
            "physics_prior",
            "light_direction",
        ),
    )
    if not result["ok"]:
        return result
    with np.load(path, allow_pickle=False) as archive:
        prior = np.asarray(archive["physics_prior"])
        coarse = np.asarray(archive["coarse_size"])
        direction = np.asarray(archive["light_direction"])
        if prior.ndim != 2:
            result["ok"] = False
            result["error"] = f"{path} physics_prior must be [H,W], got {prior.shape}"
            return result
        if coarse.shape != (2,) or tuple(int(value) for value in coarse) != prior.shape:
            result["ok"] = False
            result["error"] = (
                f"{path} coarse_size={coarse.tolist()} does not match prior={prior.shape}"
            )
            return result
        if direction.shape != (3,):
            result["ok"] = False
            result["error"] = (
                f"{path} light_direction must be [3], got {direction.shape}"
            )
            return result
        result["cache_id"] = _scalar_text(archive["cache_id"])
        result["geometry_cache_id"] = _scalar_text(archive["geometry_cache_id"])
        result["shape"] = list(prior.shape)
    return result


def _inspect_point_map(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.as_posix(),
        "ok": False,
    }
    if not _nonempty_file(path):
        result["error"] = f"Missing point map: {path}"
        return result
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 3 or array.shape[-1] != 3:
        result["error"] = f"{path} point map must be [H,W,3], got {array.shape}"
        return result
    result["shape"] = list(array.shape)
    result["ok"] = True
    return result


def _inspect_predicted_mask(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.as_posix(),
        "ok": False,
    }
    if not _nonempty_file(path):
        result["error"] = f"Missing predicted mask: {path}"
        return result
    with Image.open(path) as image:
        gray = image.convert("L")
        if gray.size != (480, 480):
            result["error"] = f"{path} size is {gray.size}, expected (480, 480)"
            return result
        array = np.asarray(gray, dtype=np.uint8)
    values = np.unique(array)
    if not np.isin(values, (0, 255)).all():
        result["error"] = (
            f"{path} must be hard 0/255, found values {values[:8].tolist()}"
        )
        return result
    result["empty"] = bool(np.count_nonzero(array) == 0)
    result["ok"] = True
    return result


def _init_coverage() -> dict[str, dict[str, Any]]:
    return {
        key: {
            "rows_with_key": 0,
            "existing": 0,
            "missing": 0,
            "invalid": 0,
            "missing_examples": [],
            "invalid_examples": [],
        }
        for key in REQUIRED_ROW_KEYS + OPTIONAL_ROW_KEYS
    }


def _append_example(bucket: list[str], value: str, *, limit: int = 10) -> None:
    if len(bucket) < limit:
        bucket.append(value)


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    base_path = args.base_path.resolve()
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "allow_incomplete": bool(args.allow_incomplete),
        "manifests": [],
        "totals": {
            "row_count": 0,
            "unique_sample_count": 0,
            "scene_count": 0,
            "split_counts": {},
        },
        "scene_split_disjoint": {"ok": True, "conflicts": []},
        "duplicate_samples": {"identical": 0, "conflicts": []},
        "coverage": _init_coverage(),
        "masks": {
            "predicted_shadow_mask": {
                "validated": 0,
                "valid_binary_480": 0,
                "empty": 0,
            }
        },
        "light_position": {
            "rows_with_key": 0,
            "valid": 0,
            "invalid": 0,
            "invalid_examples": [],
        },
        "checkpoints": {},
        "errors": [],
        "warnings": [],
    }

    rows_by_sample: dict[str, dict[str, Any]] = {}
    scenes_to_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    all_rows: list[tuple[dict[str, Any], Path]] = []
    path_cache: dict[tuple[str, str], dict[str, Any]] = {}

    for manifest_path in args.manifest:
        resolved_manifest = manifest_path.resolve()
        rows = read_jsonl(resolved_manifest)
        split_counter = Counter(str(row.get("shadow_c2f_split", "unspecified")) for row in rows)
        scenes = sorted({str(row.get("scene_id", "")) for row in rows if row.get("scene_id")})
        report["manifests"].append(
            {
                "path": resolved_manifest.as_posix(),
                "row_count": len(rows),
                "scene_count": len(scenes),
                "split_counts": dict(sorted(split_counter.items())),
            }
        )
        for row in rows:
            all_rows.append((row, resolved_manifest))

    report["totals"]["row_count"] = len(all_rows)
    if not all_rows:
        _record_error(report, "No rows found across the provided manifests")

    for row, manifest_path in all_rows:
        split = str(row.get("shadow_c2f_split", "unspecified"))
        scene = str(row.get("scene_id", ""))
        sample_id = str(row.get("shadow_c2f_sample_id") or sample_key(row))
        split_counts[split] += 1
        if scene:
            scenes_to_splits[scene].add(split)

        shadow_like = sorted(
            key
            for key in row
            if (
                "shadow_mask" in key
                and key not in {"gt_shadow_mask", "predicted_shadow_mask"}
            )
            or key in BANNED_KEYS
        )
        if shadow_like:
            _record_error(
                report,
                f"{manifest_path}:{sample_id} leaks GT shadow fields: {shadow_like}",
            )
        if "gt_shadow_mask" in row and "predicted_shadow_mask" in row:
            gt_path = _resolve_existing_row_path(row, "gt_shadow_mask", base_path=base_path)
            pred_path = _resolve_existing_row_path(row, "predicted_shadow_mask", base_path=base_path)
            if gt_path is not None and pred_path is not None and gt_path == pred_path:
                _record_error(
                    report,
                    f"{manifest_path}:{sample_id} predicted_shadow_mask aliases gt_shadow_mask",
                )

        normalized = {
            "split": split,
            "scene_id": scene,
            "light_position": row.get("light_position"),
            "source_image": row.get("source_image"),
            "baseline_image": row.get("baseline_image"),
            "object_mask": row.get("object_mask"),
            "receiver_mask": row.get("receiver_mask"),
            "gt_shadow_mask": row.get("gt_shadow_mask"),
            "predicted_shadow_mask": row.get("predicted_shadow_mask"),
        }
        previous = rows_by_sample.get(sample_id)
        if previous is None:
            rows_by_sample[sample_id] = normalized
        elif previous == normalized:
            report["duplicate_samples"]["identical"] += 1
        else:
            report["duplicate_samples"]["conflicts"].append(
                {"sample_id": sample_id, "first": previous, "second": normalized}
            )

        light_position_result = _inspect_light_position(row.get("light_position"))
        report["light_position"]["rows_with_key"] += int(row.get("light_position") is not None)
        if light_position_result["ok"]:
            report["light_position"]["valid"] += 1
        else:
            report["light_position"]["invalid"] += 1
            _append_example(
                report["light_position"]["invalid_examples"],
                f"{sample_id}:{light_position_result['error']}",
            )
            _record_error(
                report,
                f"{manifest_path}:{sample_id} invalid light_position: "
                f"{light_position_result['error']}",
            )

        for key in REQUIRED_ROW_KEYS + OPTIONAL_ROW_KEYS:
            coverage = report["coverage"][key]
            resolved = _resolve_existing_row_path(row, key, base_path=base_path)
            if resolved is None:
                continue
            coverage["rows_with_key"] += 1
            if not _nonempty_file(resolved):
                coverage["missing"] += 1
                _append_example(
                    coverage["missing_examples"],
                    f"{sample_id}:{resolved.as_posix()}",
                )
                continue
            coverage["existing"] += 1

            cache_key = (key, resolved.as_posix())
            if cache_key in path_cache:
                result = path_cache[cache_key]
            elif key == "predicted_shadow_mask":
                result = _inspect_predicted_mask(resolved)
            elif key == "point_map":
                result = _inspect_point_map(resolved)
            elif key in ("adapter_source_cache", "adapter_target_cache"):
                result = _inspect_npz(
                    resolved,
                    expected_schema=ADAPTER_IMAGE_CACHE_SCHEMA,
                    expected_fields=(
                        "schema",
                        "cache_id",
                        "output_size",
                        "final_logit",
                        "final_prob",
                        "coarse_logit",
                        "coarse_prob",
                    ),
                    exact_output_size=480,
                )
            elif key == "adapter_delta_cache":
                result = _inspect_npz(
                    resolved,
                    expected_schema=ADAPTER_DELTA_CACHE_SCHEMA,
                    expected_fields=(
                        "schema",
                        "cache_id",
                        "output_size",
                        "positive_delta_final_prob",
                        "positive_delta_coarse_prob",
                    ),
                    exact_output_size=480,
                )
            elif key == "geometry_cache":
                result = _inspect_geometry_cache(resolved)
            elif key == "physics_cache":
                result = _inspect_physics_cache(resolved)
            else:
                result = {"path": resolved.as_posix(), "ok": True}
            path_cache[cache_key] = result

            if not result["ok"]:
                coverage["invalid"] += 1
                _append_example(
                    coverage["invalid_examples"],
                    f"{sample_id}:{result.get('error', resolved.as_posix())}",
                )
            elif key == "predicted_shadow_mask":
                report["masks"]["predicted_shadow_mask"]["validated"] += 1
                report["masks"]["predicted_shadow_mask"]["valid_binary_480"] += 1
                report["masks"]["predicted_shadow_mask"]["empty"] += int(
                    bool(result.get("empty"))
                )

        geometry_path = _resolve_existing_row_path(row, "geometry_cache", base_path=base_path)
        physics_path = _resolve_existing_row_path(row, "physics_cache", base_path=base_path)
        if geometry_path and physics_path and _nonempty_file(geometry_path) and _nonempty_file(physics_path):
            geometry_result = path_cache.get(("geometry_cache", geometry_path.as_posix()))
            physics_result = path_cache.get(("physics_cache", physics_path.as_posix()))
            if geometry_result and physics_result and geometry_result.get("ok") and physics_result.get("ok"):
                if geometry_result.get("cache_id") != physics_result.get("geometry_cache_id"):
                    _record_error(
                        report,
                        f"{manifest_path}:{sample_id} physics_cache geometry_cache_id "
                        f"{physics_result.get('geometry_cache_id')!r} does not match "
                        f"geometry_cache cache_id {geometry_result.get('cache_id')!r}",
                    )

    report["totals"]["unique_sample_count"] = len(rows_by_sample)
    report["totals"]["scene_count"] = len(scenes_to_splits)
    report["totals"]["split_counts"] = dict(sorted(split_counts.items()))

    for scene, splits in sorted(scenes_to_splits.items()):
        if len(splits) > 1:
            report["scene_split_disjoint"]["ok"] = False
            report["scene_split_disjoint"]["conflicts"].append(
                {"scene_id": scene, "splits": sorted(splits)}
            )

    if report["duplicate_samples"]["conflicts"]:
        _record_error(
            report,
            f"Found {len(report['duplicate_samples']['conflicts'])} conflicting duplicate sample rows",
        )
    if not report["scene_split_disjoint"]["ok"]:
        _record_error(
            report,
            f"Found {len(report['scene_split_disjoint']['conflicts'])} scene/split conflicts",
        )

    for key in REQUIRED_ROW_KEYS:
        coverage = report["coverage"][key]
        if coverage["rows_with_key"] != len(all_rows):
            _record_error(
                report,
                f"Required field {key} is absent from {len(all_rows) - coverage['rows_with_key']} row(s)",
            )
        if coverage["missing"] or coverage["invalid"]:
            _record_error(
                report,
                f"Required field {key} has missing={coverage['missing']} invalid={coverage['invalid']}",
            )

    for key in OPTIONAL_ROW_KEYS:
        coverage = report["coverage"][key]
        if coverage["rows_with_key"] == 0:
            _record_warning(report, f"Optional field {key} is absent from all rows")
            continue
        if coverage["missing"] or coverage["invalid"]:
            _record_error(
                report,
                f"Optional field {key} has missing={coverage['missing']} invalid={coverage['invalid']}",
            )

    if report["light_position"]["rows_with_key"] != len(all_rows):
        _record_error(
            report,
            f"Required field light_position is absent from {len(all_rows) - report['light_position']['rows_with_key']} row(s)",
        )
    if report["light_position"]["invalid"]:
        _record_error(
            report,
            f"Required field light_position has invalid={report['light_position']['invalid']}",
        )

    report["checkpoints"]["baseline_checkpoint"] = _inspect_image_checkpoint(
        args.baseline_checkpoint.resolve()
    )
    report["checkpoints"]["baseline_dir"] = {
        "path": args.baseline_dir.resolve().as_posix(),
        "ok": args.baseline_dir.resolve().is_dir(),
    }
    if not report["checkpoints"]["baseline_dir"]["ok"]:
        report["checkpoints"]["baseline_dir"]["error"] = (
            f"Missing baseline prediction directory: {args.baseline_dir.resolve()}"
        )

    report["checkpoints"]["adaptershadow"] = _inspect_adaptershadow(args)
    if args.c2f_checkpoint:
        report["checkpoints"]["c2f_checkpoint"] = _inspect_c2f_checkpoint(
            args.c2f_checkpoint.resolve()
        )
    if args.wan_checkpoint:
        report["checkpoints"]["wan_checkpoint"] = _inspect_image_checkpoint(
            args.wan_checkpoint.resolve()
        )

    for name, value in report["checkpoints"].items():
        if not value.get("ok", False):
            _record_error(report, f"{name} check failed: {value.get('error', 'unknown error')}")

    report["ok"] = not report["errors"]
    return report


def exit_code_for_report(report: dict[str, Any], *, allow_incomplete: bool) -> int:
    if report["ok"] or allow_incomplete:
        return 0
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = run_preflight(args)
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output_report:
        _write_report(args.output_report.resolve(), report)
    return exit_code_for_report(report, allow_incomplete=bool(args.allow_incomplete))


if __name__ == "__main__":
    raise SystemExit(main())
