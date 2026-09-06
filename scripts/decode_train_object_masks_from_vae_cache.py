#!/usr/bin/env python3
"""Decode cached object-mask latents into compact token-grid binary masks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline  # noqa: E402
from model import train_tokenlight as base  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vae", default="weights/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--token-size", type=int, default=15)
    parser.add_argument("--token-threshold", type=float, default=0.1)
    args = parser.parse_args()
    cache_dir = (ROOT / args.cache_dir).resolve() if not Path(args.cache_dir).is_absolute() else Path(args.cache_dir)
    output_dir = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    vae_path = (ROOT / args.vae).resolve() if not Path(args.vae).is_absolute() else Path(args.vae)
    if args.batch_size <= 0 or args.token_size <= 0 or not 0.0 < args.token_threshold < 1.0:
        raise ValueError("batch-size and token-size must be positive")

    records = []
    with (cache_dir / "index.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            scene_id = next((part for part in Path(row["path"]).parts if part.startswith("scene_")), None)
            if scene_id is not None:
                records.append((scene_id, row["path"]))
    records = sorted(dict(records).items())
    if not records:
        raise ValueError("No scene masks in latent cache index")

    output_dir.mkdir(parents=True, exist_ok=True)
    store = base.VaeLatentCacheStore(cache_dir, shard_lru=4)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[ModelConfig(str(vae_path))],
    )
    pipe.load_models_to_device(["vae"])
    completed = 0
    center_fallbacks = 0
    with torch.no_grad():
        for start in range(0, len(records), args.batch_size):
            batch_records = records[start : start + args.batch_size]
            latents = torch.stack([store.get(path) for _, path in batch_records]).to(
                device=pipe.device, dtype=pipe.torch_dtype
            )
            decoded = pipe.vae.decode(latents, device=pipe.device, tiled=False)
            unit = ((decoded.float() + 1.0) * 0.5).mean(dim=1)
            token_masks = F.interpolate(
                unit,
                size=(args.token_size, args.token_size),
                mode="area",
            )[:, 0] >= float(args.token_threshold)
            for (scene_id, _), mask in zip(batch_records, token_masks):
                if not bool(mask.any()):
                    # A few extremely small cached masks vanish through the
                    # VAE reconstruction.  Objects are centered by the render
                    # contract, so retain a minimal 3x3 object region rather
                    # than silently disabling object-side dropout.
                    center = args.token_size // 2
                    mask[max(0, center - 1) : center + 2, max(0, center - 1) : center + 2] = True
                    center_fallbacks += 1
                scene_dir = output_dir / scene_id
                scene_dir.mkdir(parents=True, exist_ok=True)
                np.save(scene_dir / "object_mask_15.npy", mask.cpu().numpy().astype(np.uint8))
                completed += 1
            print(f"decoded={completed}/{len(records)}", flush=True)

    summary = {
        "schema": "tokenlight_object_mask_token_grid_v1",
        "source_cache": cache_dir.as_posix(),
        "scene_count": completed,
        "token_size": args.token_size,
        "threshold": args.token_threshold,
        "center_fallback_count": center_fallbacks,
        "vae": vae_path.as_posix(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
