#!/usr/bin/env python3
from __future__ import annotations

"""Run only the clean-condition shadow-mask head from a joint checkpoint."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.joint_mask import LightConditionedMaskPredictor  # noqa: E402


def resolve(path: str | Path, base: Path = ROOT) -> Path:
    value = Path(path)
    return value if value.is_absolute() else base / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-path", default="")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for manifest_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("valid") is False:
                continue
            row["_manifest_index"] = manifest_index
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def load_rgb(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as opened:
        return opened.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def label(image: Image.Image, text: str) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, result.width, 28), fill="black")
    draw.text((8, 7), text, fill="white")
    return result


def main() -> None:
    args = parse_args()
    checkpoint = resolve(args.checkpoint)
    manifest = resolve(args.manifest)
    cache_root = resolve(args.cache_root)
    output_dir = resolve(args.output_dir)
    base_path = resolve(args.base_path) if args.base_path else ROOT
    output_dir.mkdir(parents=True, exist_ok=True)

    full_state = load_file(str(checkpoint), device="cpu")
    prefix = "joint_mask_predictor."
    state = {key.removeprefix(prefix): value for key, value in full_state.items() if key.startswith(prefix)}
    if not state:
        raise KeyError(f"No {prefix} weights in {checkpoint}")
    predictor = LightConditionedMaskPredictor().to(args.device)
    predictor.load_state_dict(state, strict=True)
    predictor.eval()

    scene_cache: dict[str, dict] = {}
    records = []
    for output_index, row in enumerate(load_rows(manifest, args.limit)):
        relative_cache = str(row["_scene_cache_file"])
        if relative_cache not in scene_cache:
            scene_cache[relative_cache] = torch.load(
                cache_root / relative_cache, map_location="cpu", weights_only=False, mmap=True
            )
        source = scene_cache[relative_cache]["source_latent"]
        if source.ndim == 4:
            source = source.unsqueeze(0)
        source = source.to(args.device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            logits, _ = predictor(source, row.get("attrs_json"), output_size=(480, 480))
            probability = logits[0, 0, 0].float().sigmoid().cpu().numpy()

        stem = f"{output_index:03d}_{row.get('scene_id', 'scene')}_{row.get('sample_name', 'sample')}"
        probability_image = Image.fromarray(np.round(probability * 255).astype(np.uint8), mode="L")
        binary_image = Image.fromarray((probability >= args.threshold).astype(np.uint8) * 255, mode="L")
        probability_image.save(output_dir / f"{stem}_probability.png")
        binary_image.save(output_dir / f"{stem}_binary.png")

        panels = []
        source_path = resolve(row["input_image"], base_path)
        if source_path.is_file():
            panels.append(label(load_rgb(source_path, (480, 480)), "source"))
        panels.append(label(probability_image.convert("RGB"), "prediction probability"))
        panels.append(label(binary_image.convert("RGB"), f"prediction >= {args.threshold:g}"))
        gt_value = row.get("shadow_mask")
        gt_path = resolve(gt_value, base_path) if gt_value else None
        if gt_path is not None and gt_path.is_file():
            panels.append(label(load_rgb(gt_path, (480, 480)), "GT shadow mask"))
        comparison = Image.new("RGB", (480 * len(panels), 480))
        for panel_index, panel in enumerate(panels):
            comparison.paste(panel, (480 * panel_index, 0))
        comparison.save(output_dir / f"{stem}_comparison.png")
        records.append({
            "manifest_index": row["_manifest_index"],
            "scene_id": row.get("scene_id"),
            "sample_name": row.get("sample_name"),
            "foreground_probability_mean": float(probability.mean()),
            "binary_foreground_fraction": float((probability >= args.threshold).mean()),
            "probability": f"{stem}_probability.png",
            "binary": f"{stem}_binary.png",
            "comparison": f"{stem}_comparison.png",
        })
        print(f"[{output_index + 1}/{len(load_rows(manifest, args.limit))}] {stem}", flush=True)

    summary = {
        "checkpoint": str(checkpoint),
        "manifest": str(manifest),
        "threshold": args.threshold,
        "count": len(records),
        "records": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
