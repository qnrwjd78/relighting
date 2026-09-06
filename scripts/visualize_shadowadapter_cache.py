#!/usr/bin/env python3
"""Render AdapterShadow source/target/delta caches as an auditable contact sheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    return parser.parse_args()


def resolve(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_rgb(path: Path, size: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def probability_image(array: np.ndarray, size: int) -> Image.Image:
    values = np.asarray(array, dtype=np.float32).squeeze()
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    values = np.clip(values, 0.0, 1.0)
    image = Image.fromarray(np.round(values * 255.0).astype(np.uint8), mode="L")
    return image.resize((size, size), Image.Resampling.BILINEAR).convert("RGB")


def add_label(image: Image.Image, label: str, font: ImageFont.ImageFont) -> Image.Image:
    result = Image.new("RGB", (image.width, image.height + 28), "white")
    result.paste(image, (0, 28))
    draw = ImageDraw.Draw(result)
    draw.text((8, 7), label, fill="black", font=font)
    return result


def main() -> int:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    manifest = args.manifest.resolve()
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    summaries = []
    for row in rows[: args.limit]:
        source_cache = resolve(row["adapter_source_cache"], manifest.parent)
        target_cache = resolve(row["adapter_target_cache"], manifest.parent)
        delta_cache = resolve(row["adapter_delta_cache"], manifest.parent)
        with np.load(source_cache, allow_pickle=False) as source_npz, \
                np.load(target_cache, allow_pickle=False) as target_npz, \
                np.load(delta_cache, allow_pickle=False) as delta_npz:
            cache_ids = {
                str(source_npz["cache_id"]), str(target_npz["cache_id"]), str(delta_npz["cache_id"])
            }
            if len(cache_ids) != 1:
                raise ValueError(f"Adapter cache ID mismatch: {sorted(cache_ids)}")
            cache_id = next(iter(cache_ids))
            size = 320
            panels = [
                add_label(load_rgb(resolve(row["source_image"], manifest.parent), size), "Source image", font),
                add_label(load_rgb(resolve(row["baseline_image"], manifest.parent), size), "Baseline epoch-10", font),
                add_label(probability_image(source_npz["final_prob"], size), "Adapter source probability", font),
                add_label(probability_image(target_npz["final_prob"], size), "Adapter baseline probability", font),
                add_label(probability_image(delta_npz["positive_delta_coarse_prob"], size), "Positive delta (coarse)", font),
                add_label(probability_image(delta_npz["positive_delta_final_prob"], size), "Positive delta (final)", font),
            ]
            if row.get("gt_shadow_mask"):
                panels.append(add_label(load_rgb(resolve(row["gt_shadow_mask"], manifest.parent), size), "GT shadow mask", font))
        banner = 44
        sheet = Image.new("RGB", (size * len(panels), panels[0].height + banner), "white")
        draw = ImageDraw.Draw(sheet)
        warning = "MOCK BACKEND - PIPELINE CHECK ONLY" if cache_id.startswith("mock-") else "AdapterShadow result"
        draw.rectangle((0, 0, sheet.width, banner), fill=(150, 20, 20) if cache_id.startswith("mock-") else (25, 80, 145))
        draw.text((10, 9), f"{warning} | cache_id={cache_id}", fill="white", font=font)
        for index, panel in enumerate(panels):
            sheet.paste(panel, (index * size, banner))
        sample_id = row.get("shadow_c2f_sample_id") or f"{row.get('scene_id', 'sample')}_{row.get('light_id', 0)}"
        output = output_dir / f"{sample_id}.jpg"
        sheet.save(output, quality=94)
        summaries.append({"sample_id": sample_id, "cache_id": cache_id, "output": output.as_posix()})
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"visualized": len(summaries), "output_dir": output_dir.as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
