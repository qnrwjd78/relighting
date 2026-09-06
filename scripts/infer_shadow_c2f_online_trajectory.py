#!/usr/bin/env python3
"""Run frozen Wan→FOCUS and a trained C2F checkpoint on an ordered manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from model.shadow_c2f import ShadowC2FConfig, ShadowCoarseToFine
from utils.shadow_c2f_dataset import ShadowC2FDataset
from utils.shadow_c2f_online import FrozenOnlineShadowConditioner


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--baseline-checkpoint", type=Path, required=True)
    p.add_argument("--wan-weights", type=Path, required=True)
    p.add_argument("--focus-config", type=Path, required=True)
    p.add_argument("--focus-checkpoint", type=Path, required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=2.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = torch.device(args.device)
    dataset = ShadowC2FDataset(
        args.manifest, adapter_mode="zero", coarse_prior_mode="adapter_delta",
        require_target=False, augment_adapter=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config_values = dict(checkpoint["model_config"])
    config_values["coarse_size"] = tuple(config_values["coarse_size"])
    model = ShadowCoarseToFine(ShadowC2FConfig(**config_values)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    conditioner = FrozenOnlineShadowConditioner(
        device=device, baseline_checkpoint=args.baseline_checkpoint,
        wan_weights=args.wan_weights, focus_config=args.focus_config,
        focus_checkpoint=args.focus_checkpoint, steps=args.steps,
        cfg_scale=args.cfg_scale,
    )
    masks = args.output_dir / "masks"
    probabilities = args.output_dir / "probabilities"
    masks.mkdir(parents=True, exist_ok=True)
    probabilities.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            features, prior = conditioner(batch)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    batch["image"], batch["object_mask"], batch["point_map"],
                    batch["light_position"], prior,
                    receiver_mask=batch["receiver_mask"], adapter_features=features,
                )
            probability = output["receiver_masked_mask"][0, 0].float().cpu().numpy()
            binary = (probability >= args.threshold).astype(np.uint8) * 255
            Image.fromarray(binary, mode="L").save(masks / f"{index:03d}.png")
            np.save(probabilities / f"{index:03d}.npy", probability.astype(np.float16))
            records.append({"frame_index": index, "sample_id": batch["sample_id"][0], "foreground_ratio": float(binary.mean() / 255)})
            print(f"frame={index + 1}/{len(dataset)} foreground={records[-1]['foreground_ratio']:.4f}", flush=True)
    (args.output_dir / "inference_summary.json").write_text(json.dumps({
        "schema": "shadow_c2f_online_trajectory_v1", "rows": len(records),
        "checkpoint": str(args.checkpoint.resolve()), "threshold": args.threshold,
        "records": records,
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
