#!/usr/bin/env bash
set -euo pipefail

cd /workspace

readonly CONFIG="configs/train_480/exp2_7x7x5_power06_rgb_joint_shadow_mask_clean_resume_e3_to20_b5x4_ga2_gb40.json"
readonly CHECKPOINT="outputs/train/exp_2/rgb_joint_shadow_mask_cleancond_7x7x5_power06_scene64_fresh_20ep_b5x8_ga1_gb40_20260831_161445/epoch-3.safetensors"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing continuation checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=4,5,6,7
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

/usr/bin/python3 model/train_tokenlight_joint_mask.py \
  --config "${CONFIG}" \
  --preflight \
  --preflight_max_samples 128

exec /usr/local/bin/accelerate launch \
  --config_file configs/accelerate_4gpu_ddp.yaml \
  model/train_tokenlight_joint_mask.py \
  --config "${CONFIG}"
