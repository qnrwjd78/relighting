#!/usr/bin/env bash
set -euo pipefail

cd /workspace

export CUDA_VISIBLE_DEVICES=4,5,6,7
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

/usr/bin/python3 model/train_tokenlight_joint_mask.py \
  --config configs/train_480/exp2_7x7x5_power06_rgb_joint_shadow_mask_clean_20ep_b5x4_ga2_gb40.json \
  --preflight \
  --preflight_max_samples 128

exec /usr/local/bin/accelerate launch \
  --config_file configs/accelerate_4gpu_ddp.yaml \
  model/train_tokenlight_joint_mask.py \
  --config configs/train_480/exp2_7x7x5_power06_rgb_joint_shadow_mask_clean_20ep_b5x4_ga2_gb40.json
