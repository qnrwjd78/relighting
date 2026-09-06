#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/lib/training_launch.sh"

launch_exp2_joint_mask \
  configs/train_480/exp2_7x7x5_power06_rgb_joint_shadow_mask_clean_resume_e3_to20_b5x4_ga2_gb40.json \
  configs/accelerate_4gpu_ddp.yaml 4,5,6,7 \
  outputs/train/exp_2/rgb_joint_shadow_mask_cleancond_7x7x5_power06_scene64_fresh_20ep_b5x8_ga1_gb40_20260831_161445/epoch-3.safetensors
