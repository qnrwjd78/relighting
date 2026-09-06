#!/usr/bin/env python3
"""Build scene-disjoint CoShadow metadata from the fixed32 PNG dataset.

The input rows are preserved verbatim (including ``attrs_json``).  This script
only validates their image contract and appends derived shadow-layout fields.
All rows from one scene are assigned to the same split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image


DEFAULT_METADATA = "data_train/objaverse_fixed32_480/metadata.jsonl"
DEFAULT_DATA_ROOT = "data/objaverse_fixed32_png"
DEFAULT_OUTPUT_DIR = "data_train/objaverse_fixed32_coshadow_480"
DEFAULT_REJECT_METADATA = "data/objaverse_fixed32_png/reject_metadata.txt"
REQUIRED_PATH_KEYS = ("video", "input_image", "mask", "shadow_mask")


def _finite_ratio(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative, got {value}")
    return value


def parse_split_ratios(value: str) -> dict[str, float]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("--split-ratios must contain train,val,test ratios")
    ratios = {
        name: _finite_ratio(float(part), name)
        for name, part in zip(("train", "val", "test"), parts, strict=True)
    }
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("At least one split ratio must be positive")
    return {name: ratio / total for name, ratio in ratios.items()}


def scene_split(scene_id: str, *, seed: int, ratios: Mapping[str, float]) -> str:
    """Assign a complete scene using a stable hash, independent of row order."""

    digest = hashlib.sha256(f"{int(seed)}:{scene_id}".encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
    train_boundary = float(ratios["train"])
    val_boundary = train_boundary + float(ratios["val"])
    if unit < train_boundary:
        return "train"
    if unit < val_boundary:
        return "val"
    return "test"


def resolve_under_root(root: Path, value: str, *, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing non-empty path key {key!r}")
    raw = Path(value)
    path = raw if raw.is_absolute() else root / raw
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Path for {key!r} escapes data root: {value!r}") from exc
    if path.suffix.lower() != ".png":
        raise ValueError(f"Expected PNG for {key!r}, got {value!r}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {key!r} PNG: {resolved}")
    return resolved


def decode_png(path: Path) -> Image.Image:
    """Verify PNG structure, then force a second full pixel decode."""

    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError(f"File does not decode as PNG: {path}")
        image.verify()
    with Image.open(path) as image:
        image.load()
        return image.copy()


def foreground_bbox_from_mask(
    image: Image.Image,
    *,
    threshold: int = 127,
) -> tuple[list[float], bool, int]:
    """Return normalized half-open XYXY bounds, validity, and foreground area."""

    if not 0 <= int(threshold) <= 255:
        raise ValueError(f"threshold must be in [0,255], got {threshold}")
    gray = image.convert("L")
    width, height = gray.size
    histogram = gray.histogram()
    area = int(sum(histogram[int(threshold) + 1 :]))
    if area == 0:
        return [0.0, 0.0, 0.0, 0.0], False, 0
    binary = gray.point(lambda value: 255 if value > threshold else 0, mode="1")
    bounds = binary.getbbox()
    if bounds is None:
        raise AssertionError("Mask histogram is non-empty but Pillow returned no bounding box")
    min_x, min_y, max_x_exclusive, max_y_exclusive = bounds
    # x2/y2 are exclusive so a mask touching the final pixel maps exactly to 1.
    bbox = [
        min_x / float(width),
        min_y / float(height),
        max_x_exclusive / float(width),
        max_y_exclusive / float(height),
    ]
    return bbox, True, area


def quantize_bbox(bbox: Iterable[float], *, bins: int = 16) -> list[int]:
    if int(bins) < 2:
        raise ValueError("bins must be at least 2")
    values = list(bbox)
    if len(values) != 4:
        raise ValueError(f"Expected four bbox coordinates, got {values}")
    return [
        max(0, min(int(bins) - 1, int(math.floor(float(value) * (int(bins) - 1) + 0.5))))
        for value in values
    ]


def _scene_id(row: Mapping[str, Any]) -> str:
    for key in ("scene_id", "scene_folder"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    video = row.get("video")
    if isinstance(video, str):
        parts = Path(video).parts
        if "scenes" in parts:
            index = parts.index("scenes")
            if index + 1 < len(parts):
                return parts[index + 1]
    raise ValueError("Metadata row has no resolvable scene id")


def enrich_row(
    row: Mapping[str, Any],
    *,
    data_root: Path,
    expected_size: tuple[int, int] | None,
    bins: int,
    threshold: int,
    split: str,
) -> dict[str, Any]:
    result = dict(row)
    if result.get("valid") is False:
        raise ValueError("Input metadata row is marked valid=false")
    attrs = result.get("attrs_json")
    if not isinstance(attrs, (str, Mapping)):
        raise ValueError("Missing attrs_json; fixed32 light conditioning would be lost")
    parsed_attrs = json.loads(attrs) if isinstance(attrs, str) else dict(attrs)
    if not isinstance(parsed_attrs, Mapping) or not parsed_attrs:
        raise ValueError("attrs_json must decode to a non-empty object")
    lights = parsed_attrs.get("lights")
    if not isinstance(lights, list) or not lights or not isinstance(lights[0], Mapping):
        raise ValueError("attrs_json must contain a non-empty lights array for fixed32")
    for name in ("x", "y", "z", "r", "g", "b", "lambda", "d"):
        value = lights[0].get(name)
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"attrs_json lights[0].{name} must be numeric, got {value!r}") from exc
        if not math.isfinite(number):
            raise ValueError(f"attrs_json lights[0].{name} must be finite, got {value!r}")

    decoded: dict[str, Image.Image] = {}
    for key in REQUIRED_PATH_KEYS:
        decoded[key] = decode_png(resolve_under_root(data_root, result.get(key), key=key))
    sizes = {key: image.size for key, image in decoded.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"Image-size mismatch: {sizes}")
    size = next(iter(sizes.values()))
    if expected_size is not None and size != expected_size:
        raise ValueError(f"Expected image size {expected_size}, got {size}")

    bbox, bbox_valid, area = foreground_bbox_from_mask(decoded["shadow_mask"], threshold=threshold)
    bins_xyxy = quantize_bbox(bbox, bins=bins) if bbox_valid else [0, 0, 0, 0]
    result.update(
        {
            "coshadow_split": split,
            "coshadow_bbox_xyxy": bbox,
            "coshadow_bbox_valid": bbox_valid,
            "coshadow_bbox_bins": bins_xyxy,
            "coshadow_bbox_num_bins": int(bins),
            "coshadow_shadow_area_pixels": int(area),
            "coshadow_image_width": int(size[0]),
            "coshadow_image_height": int(size[1]),
        }
    )
    return result


def _atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return count


def load_rejected_scenes(value: str | Path | None) -> set[str]:
    if value in (None, "", "None", "none", "null"):
        return set()
    path = Path(value).resolve(strict=True)
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    metadata_path = Path(args.metadata).resolve(strict=True)
    data_root = Path(args.data_root).resolve(strict=True)
    output_dir = Path(args.output_dir)
    ratios = parse_split_ratios(args.split_ratios)
    reject_disabled = args.reject_metadata in (None, "", "None", "none", "null")
    rejected_scenes = set() if args.include_rejected or reject_disabled else load_rejected_scenes(args.reject_metadata)
    expected_size = None if args.expected_width <= 0 or args.expected_height <= 0 else (
        int(args.expected_width),
        int(args.expected_height),
    )

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scene_to_split: dict[str, str] = {}
    seen_samples: set[tuple[str, str]] = set()
    failures: list[dict[str, Any]] = []
    input_rows = 0
    scanned_rows = 0
    excluded_rows = 0
    excluded_scenes: set[str] = set()
    empty_masks = 0
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if args.limit > 0 and input_rows >= args.limit:
                break
            scanned_rows += 1
            try:
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise TypeError("JSONL row must be an object")
                scene_id = _scene_id(row)
                if scene_id in rejected_scenes:
                    excluded_rows += 1
                    excluded_scenes.add(scene_id)
                    continue
                input_rows += 1
                split = scene_split(scene_id, seed=args.seed, ratios=ratios)
                previous = scene_to_split.setdefault(scene_id, split)
                if previous != split:
                    raise AssertionError(f"Scene split changed for {scene_id}: {previous} -> {split}")
                identity = (scene_id, str(row.get("video", "")))
                if identity in seen_samples:
                    raise ValueError(f"Duplicate sample path within scene: {identity}")
                seen_samples.add(identity)
                enriched = enrich_row(
                    row,
                    data_root=data_root,
                    expected_size=expected_size,
                    bins=args.bins,
                    threshold=args.mask_threshold,
                    split=split,
                )
                empty_masks += int(not enriched["coshadow_bbox_valid"])
                by_split[split].append(enriched)
            except Exception as exc:
                failure = {
                    "line": line_number,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failures.append(failure)
                if args.strict:
                    raise RuntimeError(
                        f"Strict validation failed at {metadata_path}:{line_number}: {failure['error']}"
                    ) from exc

    split_scenes = {
        name: {scene for scene, assigned in scene_to_split.items() if assigned == name}
        for name in ("train", "val", "test")
    }
    if split_scenes["train"] & split_scenes["val"] or split_scenes["train"] & split_scenes["test"] or split_scenes["val"] & split_scenes["test"]:
        raise AssertionError("Scene leakage detected across output splits")

    summary = {
        "schema": "fixed32_coshadow_metadata_v1",
        "source_metadata": str(metadata_path),
        "data_root": str(data_root),
        "bins": int(args.bins),
        "mask_threshold": int(args.mask_threshold),
        "split_seed": int(args.seed),
        "split_ratios": ratios,
        "reject_metadata": (
            None
            if args.include_rejected or reject_disabled
            else str(Path(args.reject_metadata).resolve())
        ),
        "include_rejected": bool(args.include_rejected),
        "configured_rejected_scenes": len(rejected_scenes),
        "scanned_rows": scanned_rows,
        "input_rows": input_rows,
        "excluded_rejected_rows": excluded_rows,
        "excluded_rejected_scenes": len(excluded_scenes),
        "output_rows": sum(len(rows) for rows in by_split.values()),
        "empty_shadow_masks": empty_masks,
        "failures": len(failures),
        "rows_by_split": {name: len(by_split[name]) for name in ("train", "val", "test")},
        "scenes_by_split": {name: len(split_scenes[name]) for name in ("train", "val", "test")},
        "attrs_preserved": True,
        "bbox_convention": "normalized_xyxy_half_open",
        "quantization": "round(coord*(bins-1))",
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        preview = next((rows[0] for rows in by_split.values() if rows), None)
        if preview is not None:
            summary["preview"] = {
                key: preview[key]
                for key in (
                    "scene_id",
                    "video",
                    "coshadow_split",
                    "coshadow_bbox_xyxy",
                    "coshadow_bbox_valid",
                    "coshadow_bbox_bins",
                )
                if key in preview
            }
        return summary

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        _atomic_write_jsonl(output_dir / f"{split}.jsonl", by_split[split])
    _atomic_write_jsonl(output_dir / "metadata.jsonl", (row for split in ("train", "val", "test") for row in by_split[split]))
    _atomic_write_jsonl(output_dir / "rejected.jsonl", failures)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--metadata", default=DEFAULT_METADATA)
    result.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    result.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    result.add_argument("--reject-metadata", default=DEFAULT_REJECT_METADATA)
    result.add_argument(
        "--include-rejected",
        action="store_true",
        help="Opt in to scenes excluded by the canonical fixed32 reject list.",
    )
    result.add_argument("--bins", type=int, default=16)
    result.add_argument("--mask-threshold", type=int, default=127)
    result.add_argument("--expected-width", type=int, default=480)
    result.add_argument("--expected-height", type=int, default=480)
    result.add_argument("--split-ratios", default="0.90,0.05,0.05")
    result.add_argument("--seed", type=int, default=260302743)
    result.add_argument("--limit", type=int, default=0, help="Validate only the first N non-empty rows; 0 means all.")
    result.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--dry-run", action="store_true", help="Validate and summarize without writing outputs.")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.bins < 2:
        raise SystemExit("--bins must be at least 2")
    if not 0 <= args.mask_threshold <= 255:
        raise SystemExit("--mask-threshold must be in [0,255]")
    summary = build(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
