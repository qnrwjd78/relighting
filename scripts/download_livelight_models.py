#!/usr/bin/env python3
"""Download third-party LiveLight base models into the requested weights root."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = {
    "sd-image-variations-diffusers": (
        "lambdalabs/sd-image-variations-diffusers",
        "42bc0ee1726b141d49f519a6ea02ccfbf073db2e",
        [
            "unet/config.json",
            "unet/diffusion_pytorch_model.bin",
            "image_encoder/config.json",
            "image_encoder/pytorch_model.bin",
        ],
    ),
    "sd-vae-ft-mse": (
        "stabilityai/sd-vae-ft-mse",
        "31f26fdeee1355a5c34592e401dd41e45d25a493",
        ["config.json", "diffusion_pytorch_model.bin"],
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-root", type=Path, required=True)
    args = parser.parse_args()
    args.weights_root.mkdir(parents=True, exist_ok=True)

    for local_name, (repo_id, revision, allow_patterns) in MODELS.items():
        output = args.weights_root / local_name
        print(f"Downloading {repo_id}@{revision} -> {output}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=allow_patterns,
            local_dir=output,
            local_dir_use_symlinks=False,
            resume_download=True,
        )


if __name__ == "__main__":
    main()
