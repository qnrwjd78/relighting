#!/usr/bin/env bash
set -euo pipefail

cd /workspace

/usr/bin/python scripts/prepare_exp1_shadow_mask_metadata.py

/usr/bin/python utils/build_latent_cache.py \
  --mode custom \
  --image-keys shadow_mask \
  --transform none \
  --data-root /workspace \
  --metadata-path data_train/objaverse_fixed_7x7x5_power06_rgb_shadow_mask_480/metadata.jsonl \
  --output-dir data/vae_cache_exp1_7x7x5_power06_rgb_shadow_mask_480 \
  --weights-dir weights/Wan2.2-TI2V-5B \
  --height 480 \
  --width 480 \
  --gpu-devices 0,1,2,3,4,5,6,7 \
  --batch-size 16 \
  --num-workers 4 \
  --shard-size 512 \
  --vae-dtype bf16 \
  --save-dtype bf16

/usr/bin/python scripts/preflight_exp1_shadow_mask_training.py
