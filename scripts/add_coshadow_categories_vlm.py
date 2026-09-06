#!/usr/bin/env python3
"""Add one short object category per scene to CoShadow JSONL metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageChops
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL; must differ from input")
    parser.add_argument("--dataset-root", default="data/objaverse_fixed32_png")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image-key", default="input_image")
    parser.add_argument("--mask-key", default="mask")
    parser.add_argument("--scene-key", default="scene_id")
    parser.add_argument("--category-key", default="coshadow_category")
    parser.add_argument("--fallback", default="object")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser.parse_args()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def object_crop(image_path: Path, mask_path: Path) -> Image.Image:
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    with Image.open(mask_path) as opened:
        mask = opened.convert("L").resize(image.size, Image.Resampling.NEAREST)
    bbox = ImageChops.difference(mask, Image.new("L", mask.size)).getbbox()
    if bbox is None:
        return image
    left, top, right, bottom = bbox
    pad = max(8, int(max(right - left, bottom - top) * 0.15))
    bbox = (max(0, left - pad), max(0, top - pad), min(image.width, right + pad), min(image.height, bottom + pad))
    return image.crop(bbox)


def clean_category(text: str, fallback: str) -> str:
    text = text.strip().splitlines()[0].strip(" .,:;\"'").lower()
    for prefix in ("the object is ", "this is ", "a photo of ", "an image of "):
        if text.startswith(prefix):
            text = text[len(prefix):]
    words = [word for word in text.split() if word]
    return " ".join(words[:5]) or fallback


def main() -> None:
    args = parse_args()
    input_path, output_path = Path(args.input), Path(args.output)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Refusing in-place overwrite; use a distinct --output path")
    root = Path(args.dataset_root)
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    processor = AutoProcessor.from_pretrained(args.model)
    dtype = torch.bfloat16 if str(args.device).startswith("cuda") else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    prompt = "Identify the main foreground object. Reply with only a short category noun phrase, no sentence."
    scene_categories: dict[str, str] = {}
    generated = 0
    for row in rows:
        scene = str(row.get(args.scene_key) or row.get("scene_folder") or len(scene_categories))
        existing = str(row.get(args.category_key) or "").strip()
        if existing and not args.overwrite_existing:
            scene_categories.setdefault(scene, existing)
            continue
        if scene not in scene_categories:
            if args.limit and generated >= args.limit:
                scene_categories[scene] = args.fallback
            else:
                crop = object_crop(resolve(root, row[args.image_key]), resolve(root, row[args.mask_key]))
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
                text = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(text=text, images=[crop], return_tensors="pt").to(args.device)
                with torch.inference_mode():
                    ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                decoded = processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                scene_categories[scene] = clean_category(decoded, args.fallback)
                generated += 1
        row[args.category_key] = scene_categories[scene]
        row["coshadow_category_model"] = args.model
        row["prompt"] = f"{scene_categories[scene]} casting shadow"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "unique_scenes": len(scene_categories), "generated": generated, "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
