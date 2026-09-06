#!/usr/bin/env python3
"""Offline structural and model-load verification for the local GenLit multi setup."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from genlit.models import (
    ControlNetSDVModel,
    UNetSpatioTemporalConditionControlNetModel,
)


BASE = Path("/workspace/weights/genlit/stable-video-diffusion-img2vid-xt")
CHECKPOINT = Path("/workspace/weights/genlit/checkpoint/multi_object")


def tensor_summary(path: Path) -> dict[str, int]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        parameter_count = sum(math.prod(handle.get_slice(key).get_shape()) for key in keys)
    return {"tensor_count": len(keys), "parameter_count": parameter_count}


def main() -> None:
    with (CHECKPOINT / "controlnet/config.json").open() as handle:
        control_config = json.load(handle)

    control_file = CHECKPOINT / "controlnet/diffusion_pytorch_model.safetensors"
    unet_file = BASE / "unet/diffusion_pytorch_model.fp16.safetensors"

    control_summary = tensor_summary(control_file)
    unet_summary = tensor_summary(unet_file)

    controlnet = ControlNetSDVModel.from_pretrained(
        CHECKPOINT, subfolder="controlnet", local_files_only=True
    )
    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(
        BASE, subfolder="unet", local_files_only=True, torch_dtype=torch.float16
    )

    report = {
        "conditioning_channels": int(control_config["conditioning_channels"]),
        "controlnet_class": type(controlnet).__name__,
        "controlnet_parameters": sum(p.numel() for p in controlnet.parameters()),
        "controlnet_safetensors": control_summary,
        "unet_class": type(unet).__name__,
        "unet_parameters": sum(p.numel() for p in unet.parameters()),
        "unet_safetensors": unet_summary,
        "unet_dtype": str(next(unet.parameters()).dtype),
    }
    if report["conditioning_channels"] != 5:
        raise RuntimeError(f"Expected 5 conditioning channels, got {report['conditioning_channels']}")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
