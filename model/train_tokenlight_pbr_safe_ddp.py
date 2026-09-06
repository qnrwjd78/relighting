#!/usr/bin/env python3
"""Two-GPU DDP entrypoint for the isolated PBR trainer (no DeepSpeed)."""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from model.train_tokenlight_pbr_safe import main


def _prepare() -> None:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != 2:
        raise RuntimeError("Launch this entrypoint with exactly two DDP processes")
    if os.environ.get("ACCELERATE_USE_DEEPSPEED", "").lower() in {"1", "true", "yes"}:
        raise RuntimeError("This entrypoint does not allow DeepSpeed")
    if "--train_mode" not in sys.argv:
        sys.argv += ["--train_mode", "single"]
    if "--gradient_accumulation_steps" not in sys.argv:
        sys.argv += ["--gradient_accumulation_steps", "10"]
    if int(os.environ.get("RANK", "0")) == 0:
        print("DDP contract: micro_batch=2, world_size=2, accumulation=10, global_batch=40")


if __name__ == "__main__":
    _prepare()
    main()
