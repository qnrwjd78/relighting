#!/usr/bin/env bash
set -euo pipefail

cd /workspace

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/usr/bin/python scripts/preflight_exp1_shadow_mask_training.py \
  --config configs/train_480/exp1_7x7x5_power06_rgb_shadow_mask_vae_scene64_15ep_b4x8_ga1_gb32_nogc.json \
  --expected-global-batch 32

exec accelerate launch \
  --config_file configs/accelerate_8gpu_ddp.yaml \
  model/train_tokenlight_scene_cache_shadow_safe_retained.py \
  --config configs/train_480/exp1_7x7x5_power06_rgb_shadow_mask_vae_scene64_15ep_b4x8_ga1_gb32_nogc.json
