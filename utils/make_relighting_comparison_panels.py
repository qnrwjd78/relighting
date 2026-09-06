#!/usr/bin/env python3
"""Build Source/Baseline/Mask loss/GT shadow/LGI/CoShadow/GT panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


def repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="data_train/objaverse_fixed32_part2000_2099_infer/metadata.jsonl",
    )
    parser.add_argument("--base-path", default="data/objaverse_fixed32_part2000_2099_png")
    parser.add_argument(
        "--baseline-dir",
        default="outputs/infer_tokenlight_480_rgb_step5690_objaverse_fixed32_part2000_2099_png",
    )
    parser.add_argument(
        "--maskloss-dir",
        default="outputs/infer/maskloss_074746_epoch7_part2000_2099",
    )
    parser.add_argument(
        "--gt-shadow-dir",
        default="outputs/infer/maskloss_074452_epoch9_part2000_2099",
    )
    parser.add_argument("--lgi-dir", default="outputs/infer/part2000_2099/lgi_step8910")
    parser.add_argument("--coshadow-dir", default="outputs/infer/part2000_2099/coshadow_step8000")
    parser.add_argument("--output-dir", default="outputs/panels/part2000_2099_7panel")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("valid") is not False:
                rows.append(row)
    return rows


def image_path(base_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_path / path


def prediction_name(row: dict) -> str:
    return f"{row['scene_id']}_light_{int(row['light_id']):03d}.png"


def load_rgb(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize(size, Image.Resampling.BICUBIC)


def font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def compose(items: list[tuple[str, Path]], output: Path) -> None:
    with Image.open(items[0][1]) as first:
        size = first.size
    images = [load_rgb(path, size) for _, path in items]
    caption_height = max(38, size[1] // 12)
    canvas = Image.new("RGB", (size[0] * len(items), size[1] + caption_height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    label_font = font(max(16, caption_height // 2))
    for index, ((label, _), image) in enumerate(zip(items, images)):
        x = index * size[0]
        canvas.paste(image, (x, 0))
        box = draw.textbbox((0, 0), label, font=label_font)
        text_width = box[2] - box[0]
        draw.text(
            (x + (size[0] - text_width) // 2, size[1] + (caption_height - (box[3] - box[1])) // 2),
            label,
            fill=(240, 240, 240),
            font=label_font,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, compress_level=1)


def main() -> int:
    args = parse_args()
    manifest = repo_path(args.manifest)
    base_path = repo_path(args.base_path)
    output_dir = repo_path(args.output_dir)
    model_dirs = [
        ("Baseline", repo_path(args.baseline_dir)),
        ("Mask loss", repo_path(args.maskloss_dir)),
        ("GT shadow", repo_path(args.gt_shadow_dir)),
        ("LGI", repo_path(args.lgi_dir)),
        ("CoShadow", repo_path(args.coshadow_dir)),
    ]

    created = skipped = missing = 0
    missing_examples: list[str] = []
    for row in load_rows(manifest):
        name = prediction_name(row)
        output = output_dir / name
        if output.exists() and not args.overwrite:
            skipped += 1
            continue
        source_value = row.get("input_image")
        target_value = row.get("target_image") or row.get("video")
        items = [("Source", image_path(base_path, source_value))]
        items.extend((label, directory / name) for label, directory in model_dirs)
        items.append(("GT", image_path(base_path, target_value)))
        absent = [str(path) for _, path in items if not path.is_file()]
        if absent:
            missing += 1
            if len(missing_examples) < 10:
                missing_examples.extend(absent[: 10 - len(missing_examples)])
            continue
        compose(items, output)
        created += 1

    print(f"created={created} skipped={skipped} missing={missing} output_dir={output_dir}")
    for path in missing_examples:
        print(f"missing: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
