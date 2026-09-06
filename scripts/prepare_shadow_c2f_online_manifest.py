#!/usr/bin/env python3
"""Select C2F rows whose online-training spatial assets are complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    rows = []
    missing = []
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            scene = str(row["scene_id"])
            scene_root = args.mask_root.resolve() / "scenes" / scene
            row["object_mask"] = str(scene_root / "masks" / "object_mask.png")
            row["receiver_mask"] = str(scene_root / "masks" / "receiver_mask.png")
            row["pbr_depth"] = str(scene_root / "pbr" / "depth.png")
            row["pbr_normal"] = str(scene_root / "pbr" / "normal.png")
            row["scene_meta"] = str(scene_root / "meta.json")
            required = [
                Path(row["source_image"]), Path(row["gt_shadow_mask"]),
                Path(row["object_mask"]), Path(row["receiver_mask"]),
                Path(row["point_map"]),
            ]
            absent = [str(path) for path in required if not path.is_file()]
            if absent:
                missing.append({"scene_id": scene, "missing": absent})
                continue
            rows.append(row)
            if args.limit > 0 and len(rows) >= args.limit:
                break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
    temporary.replace(args.output)
    print(json.dumps({"input": str(args.input), "output": str(args.output), "kept": len(rows), "dropped": len(missing), "missing_examples": missing[:3]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
