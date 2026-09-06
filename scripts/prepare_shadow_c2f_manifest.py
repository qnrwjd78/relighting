#!/usr/bin/env python3
"""Build leakage-safe manifests for ShadowAdapter + physics + C2F training.

The output never contains a ``shadow_mask`` key.  Renderer labels are renamed
to ``gt_shadow_mask`` and the mask intended for Wan is named
``predicted_shadow_mask``.  This makes accidental GT conditioning much harder.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.shadow_pipeline_io import (  # noqa: E402
    adaptershadow_cache_paths,
    atomic_write_jsonl,
    baseline_prediction_name,
    light_position,
    read_jsonl,
    resolve_path,
    sample_key,
    sample_name,
    scene_id,
    stable_row_rank,
    stable_scene_unit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-path", type=Path)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--adapter-cache-root", type=Path, required=True)
    parser.add_argument("--refined-mask-root", type=Path, required=True)
    parser.add_argument(
        "--decoded-rgb-root",
        type=Path,
        help="For cache-only training data: scenes/<scene>/source.png and samples/...",
    )
    parser.add_argument("--source-key", default="input_image")
    parser.add_argument("--target-key", default="video")
    parser.add_argument("--object-mask-key", default="mask")
    parser.add_argument("--gt-shadow-key", default="shadow_mask")
    parser.add_argument("--point-map-key", default="point_map")
    parser.add_argument(
        "--point-map-root",
        type=Path,
        help="Optional scene point maps laid out as <root>/<scene>/source.npy.",
    )
    parser.add_argument("--split-name", default="eval")
    parser.add_argument("--assign-splits", action="store_true")
    parser.add_argument("--split-ratios", default="0.8,0.1,0.1")
    parser.add_argument("--split-seed", type=int, default=260831)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--max-per-scene", type=int, default=0)
    parser.add_argument(
        "--require-assets",
        action="store_true",
        help="Require source, target, baseline, masks and geometry now.",
    )
    return parser.parse_args()


def _split_ratios(value: str) -> tuple[float, float, float]:
    ratios = tuple(float(part.strip()) for part in value.split(","))
    if len(ratios) != 3 or any(not math.isfinite(x) or x < 0 for x in ratios):
        raise ValueError("--split-ratios must be three finite non-negative numbers")
    total = sum(ratios)
    if total <= 0:
        raise ValueError("--split-ratios must have a positive sum")
    return tuple(x / total for x in ratios)  # type: ignore[return-value]


def _assign_split(scene: str, *, seed: int, ratios: tuple[float, float, float]) -> str:
    unit = stable_scene_unit(scene, seed=seed)
    if unit < ratios[0]:
        return "train"
    if unit < ratios[0] + ratios[1]:
        return "val"
    return "test"


def _select_rows(
    rows: list[dict[str, Any]], *, max_scenes: int, max_per_scene: int, seed: int
) -> list[dict[str, Any]]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scene[scene_id(row)].append(row)
    # Keep subset selection statistically independent from the split hash.
    # Reusing ``seed`` here and in ``_assign_split`` would select the scenes
    # with the smallest split values; a compact subset could consequently land
    # entirely in ``train`` (the first 0.8 interval).
    selection_seed = int(seed) ^ 0x5E1EC7
    scenes = sorted(
        by_scene,
        key=lambda name: (stable_scene_unit(name, seed=selection_seed), name),
    )
    if max_scenes > 0:
        scenes = scenes[:max_scenes]

    selected: list[dict[str, Any]] = []
    for scene in scenes:
        candidates = by_scene[scene]
        if max_per_scene <= 0 or len(candidates) <= max_per_scene:
            selected.extend(sorted(candidates, key=lambda row: stable_row_rank(row, seed=seed)))
            continue

        # Round-robin over light heights prevents the common low-z rows from
        # dominating a compact proof-of-concept subset.
        by_height: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            z = round(light_position(row)[2], 6)
            by_height[z].append(row)
        for height_rows in by_height.values():
            height_rows.sort(key=lambda row: stable_row_rank(row, seed=seed))
        chosen: list[dict[str, Any]] = []
        heights = sorted(by_height)
        cursor = 0
        while len(chosen) < max_per_scene and heights:
            height = heights[cursor % len(heights)]
            bucket = by_height[height]
            if bucket:
                chosen.append(bucket.pop())
            if not bucket:
                heights.remove(height)
                cursor = 0
            else:
                cursor += 1
        selected.extend(chosen)
    return selected


def _existing_candidate(candidates: Iterable[Path]) -> Path:
    values = list(candidates)
    for path in values:
        if path.is_file():
            return path
    return values[0]


def _scene_root_from_gt(gt_shadow: Path, scene: str) -> Path | None:
    parts = gt_shadow.parts
    try:
        index = parts.index(scene)
    except ValueError:
        return None
    return Path(*parts[: index + 1])


def _resolved_row(row: dict[str, Any], args: argparse.Namespace, split: str) -> dict[str, Any]:
    scene = scene_id(row)
    sample = sample_name(row)
    base_path = args.base_path.resolve() if args.base_path else None

    gt_value = row.get(args.gt_shadow_key)
    if not gt_value:
        raise KeyError(f"{sample_key(row)} has no {args.gt_shadow_key!r}")
    gt_shadow = resolve_path(str(gt_value), base_path=base_path)
    shadow_scene_root = _scene_root_from_gt(gt_shadow, scene)
    base_scene_root = base_path / "scenes" / scene if base_path else None

    if args.decoded_rgb_root:
        decoded_scene = args.decoded_rgb_root.resolve() / "scenes" / scene
        source = decoded_scene / "source.png"
        target = decoded_scene / "samples" / "position" / f"{sample}.png"
    else:
        source_value = row.get(args.source_key)
        target_value = row.get(args.target_key)
        if not source_value or not target_value:
            raise KeyError(f"{sample_key(row)} is missing source or target")
        source = resolve_path(str(source_value), base_path=base_path)
        target = resolve_path(str(target_value), base_path=base_path)

    object_candidates: list[Path] = []
    object_value = row.get(args.object_mask_key)
    if object_value:
        object_candidates.append(resolve_path(str(object_value), base_path=base_path))
    if base_scene_root is not None:
        object_candidates.append(base_scene_root / "masks" / "object_mask.png")
    if shadow_scene_root is not None:
        object_candidates.append(shadow_scene_root / "masks" / "object_mask.png")
    if not object_candidates:
        raise KeyError(f"Cannot derive object mask for {sample_key(row)}")
    object_mask = _existing_candidate(object_candidates)

    receiver_candidates: list[Path] = []
    if base_scene_root is not None:
        receiver_candidates.append(base_scene_root / "masks" / "receiver_mask.png")
    if shadow_scene_root is not None:
        receiver_candidates.append(shadow_scene_root / "masks" / "receiver_mask.png")
    if not receiver_candidates:
        raise KeyError(f"Cannot derive receiver mask for {sample_key(row)}")
    receiver_mask = _existing_candidate(receiver_candidates)

    pbr_root_candidates: list[Path] = []
    if base_scene_root is not None:
        pbr_root_candidates.append(base_scene_root)
    if shadow_scene_root is not None:
        pbr_root_candidates.append(shadow_scene_root)
    pbr_scene = next(
        (root for root in pbr_root_candidates if (root / "pbr" / "depth.png").is_file()),
        pbr_root_candidates[0] if pbr_root_candidates else Path("."),
    )
    meta_path = next(
        (root / "meta.json" for root in pbr_root_candidates if (root / "meta.json").is_file()),
        pbr_scene / "meta.json",
    )

    point_map: Path | None = None
    point_value = row.get(args.point_map_key)
    if point_value:
        point_map = resolve_path(str(point_value), base_path=base_path)
    elif args.point_map_root:
        point_map = args.point_map_root.resolve() / scene / "source.npy"

    result = {
        key: value
        for key, value in row.items()
        if key not in {"shadow_mask", "shadow_mask_pad16", "predicted_shadow_mask"}
        and not key.startswith("_manifest_")
    }
    adapter_paths = adaptershadow_cache_paths(
        args.adapter_cache_root.resolve(), row, source_path=source
    )
    result.update(
        {
            "shadow_c2f_schema": "tokenlight_shadow_c2f_manifest_v1",
            "shadow_c2f_split": split,
            "shadow_c2f_sample_id": sample_key(row),
            "source_image": source.resolve().as_posix(),
            "target_image": target.resolve().as_posix(),
            "baseline_image": (
                args.baseline_dir.resolve() / baseline_prediction_name(row)
            ).as_posix(),
            "object_mask": object_mask.resolve().as_posix(),
            "receiver_mask": receiver_mask.resolve().as_posix(),
            "gt_shadow_mask": gt_shadow.resolve().as_posix(),
            "pbr_depth": (pbr_scene / "pbr" / "depth.png").resolve().as_posix(),
            "pbr_normal": (pbr_scene / "pbr" / "normal.png").resolve().as_posix(),
            "scene_meta": meta_path.resolve().as_posix(),
            **{
                key: path.resolve().as_posix()
                for key, path in adapter_paths.items()
            },
            "predicted_shadow_mask": (
                args.refined_mask_root.resolve() / scene / f"{sample}.png"
            ).as_posix(),
            "light_position": light_position(row),
        }
    )
    if point_map is not None:
        result["point_map"] = point_map.resolve().as_posix()
    oracle_available = Path(result["scene_meta"]).is_file() and Path(result["pbr_depth"]).is_file()
    moge_available = point_map is not None and point_map.is_file()
    result["geometry_mode_available"] = (
        "oracle+moge" if oracle_available and moge_available else "oracle" if oracle_available else "moge" if moge_available else "none"
    )
    return result


def _validate_assets(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in (
        "source_image",
        "target_image",
        "baseline_image",
        "object_mask",
        "receiver_mask",
        "gt_shadow_mask",
    ):
        if not Path(row[key]).is_file():
            errors.append(f"{key}={row[key]}")
    if row["geometry_mode_available"] == "none":
        errors.append("no oracle or MoGe geometry")
    return errors


def main() -> int:
    args = parse_args()
    ratios = _split_ratios(args.split_ratios)
    input_rows = read_jsonl(args.input.resolve())
    selected = _select_rows(
        input_rows,
        max_scenes=max(0, args.max_scenes),
        max_per_scene=max(0, args.max_per_scene),
        seed=args.split_seed,
    )

    output_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for row in selected:
        split = (
            _assign_split(scene_id(row), seed=args.split_seed, ratios=ratios)
            if args.assign_splits
            else args.split_name
        )
        try:
            prepared = _resolved_row(row, args, split)
            if args.require_assets:
                missing = _validate_assets(prepared)
                if missing:
                    raise FileNotFoundError(", ".join(missing))
            output_rows.append(prepared)
        except Exception as error:
            failures.append(f"{sample_key(row)}: {type(error).__name__}: {error}")

    if failures:
        preview = "\n".join(failures[:20])
        suffix = f"\n... {len(failures) - 20} more" if len(failures) > 20 else ""
        raise RuntimeError(f"Failed to prepare {len(failures)} rows:\n{preview}{suffix}")
    if not output_rows:
        raise RuntimeError("No rows selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(args.output_dir / "all.jsonl", output_rows)
    split_counts = Counter(row["shadow_c2f_split"] for row in output_rows)
    for split in sorted(split_counts):
        atomic_write_jsonl(
            args.output_dir / f"{split}.jsonl",
            (row for row in output_rows if row["shadow_c2f_split"] == split),
        )

    summary = {
        "schema": "tokenlight_shadow_c2f_manifest_summary_v1",
        "source_manifest": args.input.resolve().as_posix(),
        "input_rows": len(input_rows),
        "output_rows": len(output_rows),
        "scene_count": len({row["scene_id"] for row in output_rows}),
        "split_counts": dict(sorted(split_counts.items())),
        "geometry_counts": dict(
            sorted(Counter(row["geometry_mode_available"] for row in output_rows).items())
        ),
        "gt_isolation": {
            "renderer_key": "gt_shadow_mask",
            "wan_key": "predicted_shadow_mask",
            "shadow_mask_key_present": False,
        },
        "require_assets": bool(args.require_assets),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
