#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: run_shadow_c2f_pipeline.sh [--stage NAMES] [--limit N] [--execute]

Stages (comma-separated):
  baseline-eval, prepare-eval, adapter-eval, geometry-eval, c2f-infer,
  prepare-train, decode-train, baseline-train, adapter-train,
  geometry-train, train, wan-eval, all

Default: --stage all --dry-run. The script is intentionally non-executing
until --execute is supplied. Configure paths with the environment variables
documented in docs/SHADOW_C2F_PIPELINE.md.

Options:
  --stage NAMES   One stage or a comma-separated ordered selection.
  --limit N       Eval smoke subset; 0 means the full manifest.
  --execute       Execute selected stages.
  --dry-run       Print exact commands and write nothing (default).
  -h, --help
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

print_command() {
  printf '[dry-run]'
  printf ' %q' "$@"
  printf '\n'
}

run_command() {
  if (( DRY_RUN )); then
    print_command "$@"
  else
    "$@"
  fi
}

selected() {
  local name="$1"
  [[ "$STAGES" == "all" || ",$STAGES," == *",${name},"* ]]
}

absolute() {
  local value="$1"
  if [[ "$value" = /* ]]; then
    realpath -m -- "$value"
  else
    realpath -m -- "${WORKSPACE}/${value}"
  fi
}

STAGES="${SHADOW_C2F_STAGES:-all}"
LIMIT="${LIMIT:-0}"
DRY_RUN=1
while (( $# )); do
  case "$1" in
    --stage) STAGES="${2:?missing value for --stage}"; shift 2 ;;
    --limit) LIMIT="${2:?missing value for --limit}"; shift 2 ;;
    --execute) DRY_RUN=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ "$LIMIT" =~ ^[0-9]+$ ]] || die "--limit must be a non-negative integer"

IFS=',' read -r -a REQUESTED_STAGES <<< "$STAGES"
VALID_STAGES=(
  baseline-eval prepare-eval adapter-eval geometry-eval c2f-infer
  prepare-train decode-train baseline-train adapter-train geometry-train
  train wan-eval all
)
for requested in "${REQUESTED_STAGES[@]}"; do
  valid=0
  for candidate in "${VALID_STAGES[@]}"; do
    if [[ "$requested" == "$candidate" ]]; then
      valid=1
      break
    fi
  done
  (( valid )) || die "unknown stage: $requested"
done

PYTHON_BIN="${PYTHON_BIN:-python}"
ADAPTER_PYTHON_BIN="${ADAPTER_PYTHON_BIN:-$PYTHON_BIN}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
WAN_WEIGHTS_DIR="$(absolute "${WAN_WEIGHTS_DIR:-weights/Wan2.2-TI2V-5B}")"

# Frozen checkpoints selected for this experiment.
BASELINE_CHECKPOINT="$(absolute "${BASELINE_CHECKPOINT:-outputs/train/exp_1/rgb_baseline_7x7x5_power06_rgb_scene64_fresh_15ep_b8_ga5_gb40_20260824_095204/epoch-10.safetensors}")"
BASELINE_DEVICE="${BASELINE_DEVICE:-cuda}"
BASELINE_GPU_DEVICES="${BASELINE_GPU_DEVICES:-}"

# Evaluation branch (unseen scenes 2500-2999, random3 subset).
EVAL_VARIANT="eval_2500_2999_rgb_random3"
if (( LIMIT > 0 )); then
  EVAL_VARIANT="${EVAL_VARIANT}_limit${LIMIT}"
fi
RAW_EVAL_MANIFEST="$(absolute "${RAW_EVAL_MANIFEST:-data_train/objaverse_245_eval_2500_2999_rgb_random3/metadata_random3_heights.jsonl}")"
EVAL_DATA_ROOT="$(absolute "${EVAL_DATA_ROOT:-data/objaverse_245_eval_2500_2999/unseen_7x7x5_power06_png_exclude_requested}")"
EVAL_RUN_ROOT="$(absolute "${EVAL_RUN_ROOT:-outputs/shadow_c2f/${EVAL_VARIANT}}")"
BASELINE_EVAL_ROOT="$(absolute "${BASELINE_EVAL_ROOT:-outputs/infer/exp_1/objaverse245_eval_2500_2999_rgb_random3/rgb_baseline_epoch10}")"
BASELINE_EVAL_DIR="$(absolute "${BASELINE_EVAL_DIR:-${BASELINE_EVAL_ROOT}/predictions}")"
PREP_EVAL_DIR="$(absolute "${PREP_EVAL_DIR:-data_train/shadow_c2f_${EVAL_VARIANT}}")"
ADAPTER_EVAL_ROOT="$(absolute "${ADAPTER_EVAL_ROOT:-${EVAL_RUN_ROOT}/adaptershadow}")"
GEOMETRY_EVAL_ROOT="$(absolute "${GEOMETRY_EVAL_ROOT:-${EVAL_RUN_ROOT}/geometry}")"
EVAL_GEOMETRY_MANIFEST="$(absolute "${EVAL_GEOMETRY_MANIFEST:-${PREP_EVAL_DIR}/eval_geometry.jsonl}")"
C2F_INFER_ROOT="$(absolute "${C2F_INFER_ROOT:-${EVAL_RUN_ROOT}/c2f_infer}")"
C2F_WAN_MANIFEST="$(absolute "${C2F_WAN_MANIFEST:-${C2F_INFER_ROOT}/wan_manifest.jsonl}")"
C2F_EVAL_MANIFEST="$(absolute "${C2F_EVAL_MANIFEST:-${C2F_INFER_ROOT}/evaluation_manifest.jsonl}")"
WAN_EVAL_ROOT="$(absolute "${WAN_EVAL_ROOT:-${EVAL_RUN_ROOT}/wan_refined}")"

# AdapterShadow assets. sbu.ckpt is intentionally not auto-downloaded.
ADAPTER_ROOT="$(absolute "${ADAPTER_ROOT:-external/AdapterShadow}")"
ADAPTER_CHECKPOINT="$(absolute "${ADAPTER_CHECKPOINT:-external/AdapterShadow/checkpoint/sbu.ckpt}")"
SAM_CHECKPOINT="$(absolute "${SAM_CHECKPOINT:-external/AdapterShadow/checkpoint/sam/sam_vit_b_01ec64.pth}")"
EFFICIENTNET_CHECKPOINT="$(absolute "${EFFICIENTNET_CHECKPOINT:-external/AdapterShadow/efficientnet/hub/checkpoints/tf_efficientnet_b1_ap-44ef0a3d.pth}")"
ADAPTER_DEVICE="${ADAPTER_DEVICE:-cuda:0}"
ADAPTER_BATCH_SIZE="${ADAPTER_BATCH_SIZE:-1}"
GEOMETRY_DEVICE="${GEOMETRY_DEVICE:-cpu}"
GEOMETRY_MODE="${GEOMETRY_MODE:-moge}"

# Training branch. Defaults define a compact scene-disjoint C2F subset, not
# the unseen evaluation set. Generation is expensive and remains dry-run.
RAW_TRAIN_MANIFEST="$(absolute "${RAW_TRAIN_MANIFEST:-data_train/objaverse_fixed_7x7x5_power06_rgb_shadow_mask_480/metadata.jsonl}")"
TRAIN_RUN_ROOT="$(absolute "${TRAIN_RUN_ROOT:-outputs/shadow_c2f/train_0000_1999_subset}")"
PREP_TRAIN_DIR="$(absolute "${PREP_TRAIN_DIR:-data_train/shadow_c2f_train_0000_1999_subset}")"
TRAIN_DECODED_ROOT="$(absolute "${TRAIN_DECODED_ROOT:-${TRAIN_RUN_ROOT}/decoded_rgb}")"
TRAIN_SCENE_CACHE_ROOT="$(absolute "${TRAIN_SCENE_CACHE_ROOT:-data/objaverse_245_train_rgb_cache_0000_1999}")"
TRAIN_POINT_MAP_ROOT="$(absolute "${TRAIN_POINT_MAP_ROOT:-data/objaverse_245_train_point_map_0000_1999/objaverse_fixed_7x7x5_power06_source}")"
BASELINE_TRAIN_DIR="$(absolute "${BASELINE_TRAIN_DIR:-${TRAIN_RUN_ROOT}/baseline/predictions}")"
ADAPTER_TRAIN_ROOT="$(absolute "${ADAPTER_TRAIN_ROOT:-${TRAIN_RUN_ROOT}/adaptershadow}")"
GEOMETRY_TRAIN_ROOT="$(absolute "${GEOMETRY_TRAIN_ROOT:-${TRAIN_RUN_ROOT}/geometry}")"
TRAIN_GEOMETRY_MANIFEST="$(absolute "${TRAIN_GEOMETRY_MANIFEST:-${PREP_TRAIN_DIR}/train_geometry.jsonl}")"
VAL_GEOMETRY_MANIFEST="$(absolute "${VAL_GEOMETRY_MANIFEST:-${PREP_TRAIN_DIR}/val_geometry.jsonl}")"
C2F_TRAIN_ROOT="$(absolute "${C2F_TRAIN_ROOT:-outputs/train/shadow_c2f/moge_adapter_subset}")"
C2F_CHECKPOINT="$(absolute "${C2F_CHECKPOINT:-${C2F_TRAIN_ROOT}/best.pt}")"
TRAIN_MAX_SCENES="${TRAIN_MAX_SCENES:-256}"
TRAIN_MAX_PER_SCENE="${TRAIN_MAX_PER_SCENE:-16}"
C2F_EPOCHS="${C2F_EPOCHS:-30}"
C2F_BATCH_SIZE="${C2F_BATCH_SIZE:-4}"
C2F_NPROC="${C2F_NPROC:-1}"
C2F_DEVICE="${C2F_DEVICE:-cuda}"

WAN_CHECKPOINT="$(absolute "${WAN_CHECKPOINT:-outputs/train/exp_1/rgb_shadow_mask_vae_7x7x5_power06_rgb_scene64_15ep_b5x8_ga1_gb40_20260830_163108/epoch-9.safetensors}")"
WAN_DEVICE="${WAN_DEVICE:-cuda}"
WAN_GPU_DEVICES="${WAN_GPU_DEVICES:-}"

LIMIT_ARGS=()
if (( LIMIT > 0 )); then
  LIMIT_ARGS=(--limit "$LIMIT")
fi
BASELINE_GPU_ARGS=()
if [[ -n "$BASELINE_GPU_DEVICES" ]]; then
  BASELINE_GPU_ARGS=(--gpu-devices "$BASELINE_GPU_DEVICES")
fi

EFFECTIVE_RAW_EVAL="$RAW_EVAL_MANIFEST"
eval_stage_requested=0
for name in baseline-eval prepare-eval adapter-eval geometry-eval c2f-infer wan-eval; do
  if selected "$name"; then
    eval_stage_requested=1
    break
  fi
done
if (( LIMIT > 0 && eval_stage_requested )); then
  EFFECTIVE_RAW_EVAL="${EVAL_RUN_ROOT}/manifests/raw_eval_limit_${LIMIT}.jsonl"
  if (( DRY_RUN )); then
    printf '[dry-run] select first %d valid rows: %q -> %q\n' \
      "$LIMIT" "$RAW_EVAL_MANIFEST" "$EFFECTIVE_RAW_EVAL"
  else
    "$PYTHON_BIN" - "$RAW_EVAL_MANIFEST" "$EFFECTIVE_RAW_EVAL" "$LIMIT" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path

source, output, limit = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
rows = []
with source.open("r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("valid", True) is False:
            continue
        rows.append(row)
        if len(rows) >= limit:
            break
if len(rows) != limit:
    raise RuntimeError(f"Requested {limit} rows but found {len(rows)}")
output.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, output)
PY
  fi
fi

if (( ! DRY_RUN )); then
  printf 'WARNING: execution mode mutates caches/outputs and may allocate GPUs.\n' >&2
  printf 'Do not execute GPU stages while the current 8-GPU Wan training job is active.\n' >&2
fi

if selected baseline-eval; then
  run_command \
    "$PYTHON_BIN" "${WORKSPACE}/scripts/infer_manifest.py" \
    --manifest "$EFFECTIVE_RAW_EVAL" \
    --base-path "$EVAL_DATA_ROOT" \
    --output-dir "$BASELINE_EVAL_DIR" \
    --checkpoint "$BASELINE_CHECKPOINT" \
    --weights_dir "$WAN_WEIGHTS_DIR" \
    --source-key input_image \
    --target-key target_image \
    --target-fallback-key video \
    --height 480 --width 480 --num_frames 1 \
    --num_inference_steps 50 --cfg_scale 2.0 \
    --seed-key inference_seed --tokenlight_max_lights 2 \
    --no-tokenlight_mask_tokens --no-use-mask-input --no-with-gt \
    --device "$BASELINE_DEVICE" --skip-existing \
    "${BASELINE_GPU_ARGS[@]}" "${LIMIT_ARGS[@]}"
fi

if selected prepare-eval; then
  run_command \
    "$PYTHON_BIN" "${WORKSPACE}/scripts/prepare_shadow_c2f_manifest.py" \
    --input "$EFFECTIVE_RAW_EVAL" \
    --output-dir "$PREP_EVAL_DIR" \
    --base-path "$EVAL_DATA_ROOT" \
    --baseline-dir "$BASELINE_EVAL_DIR" \
    --adapter-cache-root "$ADAPTER_EVAL_ROOT" \
    --refined-mask-root "${C2F_INFER_ROOT}/refined_binary" \
    --split-name eval --require-assets
fi

if selected adapter-eval; then
  run_command \
    "$ADAPTER_PYTHON_BIN" "${WORKSPACE}/scripts/run_shadowadapter_cache.py" \
    --manifest "${PREP_EVAL_DIR}/eval.jsonl" \
    --source-key source_image --baseline-key baseline_image \
    --output-dir "$ADAPTER_EVAL_ROOT" \
    --backend official --adaptershadow-root "$ADAPTER_ROOT" \
    --checkpoint "$ADAPTER_CHECKPOINT" \
    --sam-checkpoint "$SAM_CHECKPOINT" \
    --efficientnet-checkpoint "$EFFICIENTNET_CHECKPOINT" \
    --device "$ADAPTER_DEVICE" --batch-size "$ADAPTER_BATCH_SIZE" \
    --skip-existing "${LIMIT_ARGS[@]}"
fi

if selected geometry-eval; then
  run_command \
    "$PYTHON_BIN" "${WORKSPACE}/scripts/build_shadow_physics_cache.py" \
    --manifest "${PREP_EVAL_DIR}/eval.jsonl" \
    --output-root "$GEOMETRY_EVAL_ROOT" \
    --output-manifest "$EVAL_GEOMETRY_MANIFEST" \
    --geometry-mode "$GEOMETRY_MODE" --geometry-only \
    --device "$GEOMETRY_DEVICE" --skip-existing "${LIMIT_ARGS[@]}"
fi

if selected prepare-train; then
  run_command \
    "$PYTHON_BIN" "${WORKSPACE}/scripts/prepare_shadow_c2f_manifest.py" \
    --input "$RAW_TRAIN_MANIFEST" \
    --output-dir "$PREP_TRAIN_DIR" \
    --base-path "$WORKSPACE" \
    --baseline-dir "$BASELINE_TRAIN_DIR" \
    --adapter-cache-root "$ADAPTER_TRAIN_ROOT" \
    --refined-mask-root "${TRAIN_RUN_ROOT}/refined_binary" \
    --decoded-rgb-root "$TRAIN_DECODED_ROOT" \
    --point-map-root "$TRAIN_POINT_MAP_ROOT" \
    --assign-splits --split-ratios 0.8,0.1,0.1 --split-seed 260831 \
    --max-scenes "$TRAIN_MAX_SCENES" --max-per-scene "$TRAIN_MAX_PER_SCENE"
fi

if selected decode-train; then
  run_command \
    "$PYTHON_BIN" "${WORKSPACE}/scripts/decode_shadow_c2f_train_rgb.py" \
    --manifest "${PREP_TRAIN_DIR}/all.jsonl" \
    --scene-cache-root "$TRAIN_SCENE_CACHE_ROOT" \
    --output-root "$TRAIN_DECODED_ROOT" \
    --vae "${WAN_WEIGHTS_DIR}/Wan2.2_VAE.pth" \
    --device "$C2F_DEVICE" --batch-size "$C2F_BATCH_SIZE" \
    --sources-only --skip-existing
fi

if selected baseline-train; then
  run_command \
    "$PYTHON_BIN" "${WORKSPACE}/scripts/infer_manifest_scene_cache_v2.py" \
    --manifest "${PREP_TRAIN_DIR}/all.jsonl" \
    --scene-cache-root "$TRAIN_SCENE_CACHE_ROOT" \
    --scene-cache-transform rgb \
    --base-path "$WORKSPACE" \
    --output-dir "$BASELINE_TRAIN_DIR" \
    --checkpoint "$BASELINE_CHECKPOINT" \
    --weights_dir "$WAN_WEIGHTS_DIR" \
    --source-key source_image --target-key target_image --target-fallback-key "" \
    --height 480 --width 480 --num_frames 1 \
    --num_inference_steps 50 --cfg_scale 2.0 \
    --tokenlight_max_lights 2 \
    --no-tokenlight_mask_tokens --no-use-mask-input --no-with-gt \
    --device "$BASELINE_DEVICE" --skip-existing "${BASELINE_GPU_ARGS[@]}"
fi

if selected adapter-train; then
  run_command \
    "$ADAPTER_PYTHON_BIN" "${WORKSPACE}/scripts/run_shadowadapter_cache.py" \
    --manifest "${PREP_TRAIN_DIR}/all.jsonl" \
    --source-key source_image --baseline-key baseline_image \
    --output-dir "$ADAPTER_TRAIN_ROOT" \
    --backend official --adaptershadow-root "$ADAPTER_ROOT" \
    --checkpoint "$ADAPTER_CHECKPOINT" \
    --sam-checkpoint "$SAM_CHECKPOINT" \
    --efficientnet-checkpoint "$EFFICIENTNET_CHECKPOINT" \
    --device "$ADAPTER_DEVICE" --batch-size "$ADAPTER_BATCH_SIZE" --skip-existing
fi

if selected geometry-train; then
  run_command \
    "$PYTHON_BIN" "${WORKSPACE}/scripts/build_shadow_physics_cache.py" \
    --manifest "${PREP_TRAIN_DIR}/train.jsonl" \
    --output-root "$GEOMETRY_TRAIN_ROOT" \
    --output-manifest "$TRAIN_GEOMETRY_MANIFEST" \
    --geometry-mode moge --geometry-only \
    --device "$GEOMETRY_DEVICE" --skip-existing
  run_command \
    "$PYTHON_BIN" "${WORKSPACE}/scripts/build_shadow_physics_cache.py" \
    --manifest "${PREP_TRAIN_DIR}/val.jsonl" \
    --output-root "$GEOMETRY_TRAIN_ROOT" \
    --output-manifest "$VAL_GEOMETRY_MANIFEST" \
    --geometry-mode moge --geometry-only \
    --device "$GEOMETRY_DEVICE" --skip-existing
fi

if selected train; then
  TRAIN_COMMAND=(
    "${WORKSPACE}/scripts/train_shadow_c2f.py"
    --train-manifest "$TRAIN_GEOMETRY_MANIFEST"
    --val-manifest "$VAL_GEOMETRY_MANIFEST"
    --output-dir "$C2F_TRAIN_ROOT"
    --epochs "$C2F_EPOCHS"
    --batch-size "$C2F_BATCH_SIZE"
    --adapter-mode required
    --coarse-prior-mode adapter_delta
    --precision bf16
    --device "$C2F_DEVICE"
  )
  if (( C2F_NPROC > 1 )); then
    run_command "$TORCHRUN_BIN" --standalone --nproc_per_node "$C2F_NPROC" "${TRAIN_COMMAND[@]}"
  else
    run_command "$PYTHON_BIN" "${TRAIN_COMMAND[@]}"
  fi
fi

if selected c2f-infer; then
  run_command \
    "$PYTHON_BIN" "${WORKSPACE}/scripts/infer_shadow_c2f.py" \
    --manifest "$EVAL_GEOMETRY_MANIFEST" \
    --checkpoint "$C2F_CHECKPOINT" \
    --output-root "$C2F_INFER_ROOT" \
    --output-manifest "$C2F_WAN_MANIFEST" \
    --evaluation-manifest "$C2F_EVAL_MANIFEST" \
    --adapter-mode required --coarse-prior-mode adapter_delta --fallback none \
    --device "$C2F_DEVICE" --batch-size "$C2F_BATCH_SIZE" \
    --skip-existing "${LIMIT_ARGS[@]}"
fi

if selected wan-eval; then
  WAN_WRAPPER=(
    bash "${WORKSPACE}/scripts/run_shadow_c2f_wan_eval.sh"
    --manifest "$C2F_EVAL_MANIFEST"
    --checkpoint "$WAN_CHECKPOINT"
    --baseline-dir "$BASELINE_EVAL_DIR"
    --output-root "$WAN_EVAL_ROOT"
    --base-path "$EVAL_DATA_ROOT"
    --weights-dir "$WAN_WEIGHTS_DIR"
    --python "$PYTHON_BIN"
    --device "$WAN_DEVICE"
    --metric-device "${METRIC_DEVICE:-cpu}"
    --limit "$LIMIT"
  )
  if [[ -n "$WAN_GPU_DEVICES" ]]; then
    WAN_WRAPPER+=(--gpu-devices "$WAN_GPU_DEVICES")
  fi
  if [[ "${WAN_EVAL_LPIPS:-0}" == "1" ]]; then
    WAN_WRAPPER+=(--lpips)
  fi
  if (( DRY_RUN )); then
    "${WAN_WRAPPER[@]}" --dry-run
  else
    "${WAN_WRAPPER[@]}" --execute
  fi
fi

if (( DRY_RUN )); then
  printf 'Dry-run complete. No files or jobs were changed.\n'
else
  printf 'Selected stages completed: %s\n' "$STAGES"
fi
