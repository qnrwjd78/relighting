#!/usr/bin/env python3
"""Decode selected RGB source/target frames from Wan scene-latent caches.

The Objaverse-245 training PNGs are not retained locally, while the RGB Wan
VAE scene caches are.  C2F/ShadowAdapter need pixels, so this utility restores
only the compact, scene-balanced subset listed in a prepared manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline  # noqa: E402
from utils.shadow_pipeline_io import atomic_write_jsonl, read_jsonl, sample_name, scene_id  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vae", type=Path, default=Path("weights/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--sources-only",
        action="store_true",
        help="Decode one ambient source per scene and leave selected GT RGB frames cached.",
    )
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _image_from_decode(decoded: torch.Tensor) -> Image.Image:
    if decoded.ndim != 4 or decoded.shape[0] < 3:
        raise ValueError(f"Expected decoded [C,T,H,W], got {tuple(decoded.shape)}")
    array = (
        ((decoded[:3, 0].float() + 1.0) * 0.5)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _save_decoded_batch(pipe, latents: torch.Tensor, paths: list[Path]) -> None:
    decoded = pipe.vae.decode(
        latents.to(device=pipe.device, dtype=pipe.torch_dtype),
        device=pipe.device,
        tiled=False,
    )
    if int(decoded.shape[0]) != len(paths):
        raise AssertionError("Wan VAE decode batch-size mismatch")
    for value, path in zip(decoded, paths, strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp.png")
        _image_from_decode(value).save(temporary, format="PNG")
        temporary.replace(path)


def _cache_path(cache_root: Path, row: dict[str, Any]) -> Path:
    value = row.get("_scene_cache_file")
    if not value:
        raise KeyError(f"{scene_id(row)}/{sample_name(row)} has no _scene_cache_file")
    path = Path(str(value))
    return path if path.is_absolute() else cache_root / path


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    rows = read_jsonl(_absolute(args.manifest))
    if args.limit > 0:
        rows = rows[: args.limit]
    by_cache: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    cache_root = _absolute(args.scene_cache_root).resolve()
    output_root = _absolute(args.output_root).resolve()
    for row in rows:
        by_cache[_cache_path(cache_root, row).resolve()].append(row)

    vae_path = _absolute(args.vae).resolve()
    if not vae_path.is_file():
        raise FileNotFoundError(vae_path)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[ModelConfig(vae_path.as_posix())],
    )
    pipe.load_models_to_device(["vae"])

    records: list[dict[str, Any]] = []
    decoded_targets = 0
    decoded_sources = 0
    with torch.inference_mode():
        for cache_path, scene_rows in sorted(by_cache.items(), key=lambda item: item[0].as_posix()):
            cache = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)
            if cache.get("schema") != "wan_vae_scene_latent_cache_v2":
                raise ValueError(f"Unexpected scene cache schema: {cache_path}")
            if cache.get("image_transform") != "rgb":
                raise ValueError(f"C2F needs RGB cache, got {cache.get('image_transform')!r}: {cache_path}")
            cache_scene = str(cache.get("scene_id"))
            if {scene_id(row) for row in scene_rows} != {cache_scene}:
                raise ValueError(f"Scene/cache mismatch in {cache_path}")
            source_path = output_root / "scenes" / cache_scene / "source.png"
            if not args.skip_existing or not source_path.is_file():
                source = cache.get("source_latent")
                if not isinstance(source, torch.Tensor):
                    raise ValueError(f"Missing source_latent: {cache_path}")
                _save_decoded_batch(pipe, source.unsqueeze(0), [source_path])
                decoded_sources += 1

            samples = cache.get("sample_latents")
            if not isinstance(samples, torch.Tensor) or samples.ndim != 5:
                raise ValueError(f"Malformed sample_latents: {cache_path}")
            pending: list[tuple[dict[str, Any], int, Path]] = []
            for row in scene_rows:
                index = int(row.get("_scene_cache_sample_index", -1))
                if index < 0 or index >= int(samples.shape[0]):
                    raise IndexError(f"Bad sample index {index} in {cache_path}")
                target_path = (
                    output_root
                    / "scenes"
                    / cache_scene
                    / "samples"
                    / "position"
                    / f"{sample_name(row)}.png"
                )
                if not args.sources_only and (
                    not args.skip_existing or not target_path.is_file()
                ):
                    pending.append((row, index, target_path))
                records.append(
                    {
                        "scene_id": cache_scene,
                        "sample_name": sample_name(row),
                        "source_image": source_path.as_posix(),
                        "target_image": target_path.as_posix(),
                        "target_decoded": bool(not args.sources_only),
                        "scene_cache": cache_path.as_posix(),
                        "scene_cache_sample_index": index,
                    }
                )
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start : start + args.batch_size]
                batch_latents = torch.stack([samples[index] for _, index, _ in batch])
                _save_decoded_batch(pipe, batch_latents, [path for _, _, path in batch])
                decoded_targets += len(batch)
            print(
                f"scene={cache_scene} selected={len(scene_rows)} "
                f"decoded_targets={decoded_targets}",
                flush=True,
            )

    atomic_write_jsonl(output_root / "index.jsonl", records)
    summary = {
        "schema": "tokenlight_shadow_c2f_decoded_rgb_v1",
        "manifest": _absolute(args.manifest).resolve().as_posix(),
        "scene_cache_root": cache_root.as_posix(),
        "output_root": output_root.as_posix(),
        "selected_rows": len(rows),
        "scene_count": len(by_cache),
        "decoded_sources_this_run": decoded_sources,
        "decoded_targets_this_run": decoded_targets,
        "sources_only": bool(args.sources_only),
        "vae": vae_path.as_posix(),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
