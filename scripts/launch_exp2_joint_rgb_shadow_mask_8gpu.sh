#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/lib/training_launch.sh"

launch_exp2_joint_mask \
  configs/train_480/exp2_7x7x5_power06_rgb_joint_shadow_mask_clean_20ep_b5x8_ga1_gb40.json \
  configs/accelerate_8gpu_ddp.yaml 0,1,2,3,4,5,6,7
