#!/usr/bin/env python3
"""Aggregate TokenLight metrics by selected light height and scene-height."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


GROUPS = ("full_image", "object_only", "background_only")
METRICS = ("psnr", "ssim", "lpips")


def average(records: list[dict]) -> dict:
    result = {}
    for group in GROUPS:
        result[group] = {}
        for metric in METRICS:
            values = []
            for row in records:
                group_values = row.get(group)
                if group_values is None and group == "full_image":
                    group_values = row.get("metrics")
                if not isinstance(group_values, dict) or group_values.get(metric) is None:
                    continue
                value = float(group_values[metric])
                if math.isfinite(value):
                    values.append(value)
            result[group][metric] = sum(values) / len(values) if values else None
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_height: dict[str, list[dict]] = defaultdict(list)
    by_scene_height: dict[str, list[dict]] = defaultdict(list)
    for record in payload["records"]:
        index = int(record.get("index", record.get("manifest_index")))
        row = rows[index]
        height = f"{float(row['light_height']):.6f}".rstrip("0").rstrip(".")
        by_height[height].append(record)
        by_scene_height[f"{row['scene_id']}|{height}"].append(record)
    output = {
        "metrics": args.metrics.resolve().as_posix(),
        "manifest": args.manifest.resolve().as_posix(),
        "height_averages": {
            key: {"count": len(items), **average(items)} for key, items in sorted(by_height.items(), key=lambda x: float(x[0]))
        },
        "scene_height_averages": {
            key: {"count": len(items), **average(items)} for key, items in sorted(by_scene_height.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
