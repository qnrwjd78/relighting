#!/usr/bin/env bash
# Shared launch implementation. Source this file from a launch_exp*.sh wrapper.

TRAINING_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

launch_exp1_shadow_mask() {
  local experiment_config="$1"
  shift
  cd "$TRAINING_WORKSPACE"
  "${PYTHON_BIN:-python}" scripts/preflight_exp1_shadow_mask_training.py \
    --config "$experiment_config" "$@"
  exec "${ACCELERATE_BIN:-accelerate}" launch \
    --config_file configs/accelerate_8gpu_ddp.yaml \
    model/train_scene_cache_shadow_safe_retained.py \
    --config "$experiment_config"
}

launch_exp2_joint_mask() {
  local experiment_config="$1"
  local accelerate_config="$2"
  local gpu_devices="$3"
  local resume_checkpoint="${4:-}"
  cd "$TRAINING_WORKSPACE"
  if [[ -n "$resume_checkpoint" && ! -f "$resume_checkpoint" ]]; then
    echo "Missing continuation checkpoint: $resume_checkpoint" >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="$gpu_devices"
  export OMP_NUM_THREADS=1
  export PYTHONUNBUFFERED=1
  "${PYTHON_BIN:-python}" model/train_joint_mask.py \
    --config "$experiment_config" --preflight --preflight_max_samples 128
  exec "${ACCELERATE_BIN:-accelerate}" launch \
    --config_file "$accelerate_config" \
    model/train_joint_mask.py --config "$experiment_config"
}
