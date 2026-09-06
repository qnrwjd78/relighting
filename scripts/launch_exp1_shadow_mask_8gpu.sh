#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/lib/training_launch.sh"

launch_exp1_shadow_mask configs/train_480/exp1_7x7x5_power06_rgb_shadow_mask_vae_scene64_15ep_b5x8_ga1_gb40.json
