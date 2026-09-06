#!/usr/bin/env python3
"""Select reproducible random light heights per scene from an eval manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selection-output", required=True, type=Path)
    parser.add_argument("--heights-per-scene", type=int, default=3)
    parser.add_argument("--seed", type=int, default=24525002999)
    parser.add_argument("--inference-seed", type=int, default=0)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_scene: dict[str, dict[float, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("valid") is False:
            continue
        scene = str(row["scene_id"])
        height = round(float(row["light_height"]), 6)
        by_scene[scene][height].append(row)

    selected_ids: set[int] = set()
    selection: dict[str, dict] = {}
    for scene, by_height in sorted(by_scene.items()):
        heights = sorted(by_height)
        if len(heights) < args.heights_per_scene:
            raise ValueError(f"{scene} has only {len(heights)} heights")
        digest = hashlib.sha256(f"{args.seed}:{scene}".encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        chosen = sorted(rng.sample(heights, args.heights_per_scene))
        counts = {f"{height:g}": len(by_height[height]) for height in chosen}
        selection[scene] = {
            "heights": chosen,
            "position_counts": counts,
            "all_selected_heights_complete_7x7": all(count == 49 for count in counts.values()),
        }
        for height in chosen:
            selected_ids.update(id(row) for row in by_height[height])

    selected_rows = []
    for row in rows:
        if id(row) not in selected_ids:
            continue
        current = dict(row)
        current["inference_seed"] = int(args.inference_seed)
        selected_rows.append(current)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected_rows:
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
    payload = {
        "schema": "objaverse245_random_scene_heights_v1",
        "source_manifest": args.input.resolve().as_posix(),
        "output_manifest": args.output.resolve().as_posix(),
        "selection_seed": args.seed,
        "inference_seed": args.inference_seed,
        "heights_per_scene": args.heights_per_scene,
        "scene_count": len(selection),
        "row_count": len(selected_rows),
        "complete_7x7_scene_count": sum(
            int(item["all_selected_heights_complete_7x7"]) for item in selection.values()
        ),
        "scenes": selection,
    }
    args.selection_output.parent.mkdir(parents=True, exist_ok=True)
    args.selection_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("scene_count", "row_count", "complete_7x7_scene_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
