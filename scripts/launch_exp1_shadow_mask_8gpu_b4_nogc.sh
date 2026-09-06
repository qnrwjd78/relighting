#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/lib/training_launch.sh"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

launch_exp1_shadow_mask configs/train_480/exp1_7x7x5_power06_rgb_shadow_mask_vae_scene64_15ep_b4x8_ga1_gb32_nogc.json --expected-global-batch 32
