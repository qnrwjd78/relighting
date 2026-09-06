#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  run_shadow_c2f_wan_eval.sh --manifest EVAL_WITH_GT.jsonl \
    --checkpoint WAN_CHECKPOINT --baseline-dir BASELINE_PREDICTIONS \
    --output-root OUTPUT_ROOT [options]

This wrapper creates a minimal, GT-free Wan manifest, runs strict mask-
conditioned Wan inference, then evaluates masks and RGB using the original
evaluation-only manifest.

Required:
  --manifest PATH          C2F inference manifest; retained for evaluation only.
  --checkpoint PATH        Finished shadow-mask-conditioned Wan checkpoint.
  --baseline-dir PATH      Baseline RGB prediction directory.
  --output-root PATH       Wan predictions, strict manifest, and metrics root.

Options:
  --base-path PATH         Root for legacy relative input_image paths.
                           Default: /workspace
  --weights-dir PATH       Default: weights/Wan2.2-TI2V-5B
  --python PATH            Default: python
  --device DEVICE          Default: cuda
  --gpu-devices IDS        Optional infer_manifest worker list, e.g. 0 or 0,1.
  --metric-device DEVICE   Default: cpu
  --steps N                Wan denoising steps. Default: 50
  --cfg-scale FLOAT        Default: 2.0
  --seed N                 Used when inference_seed is absent. Default: 0
  --limit N                First N valid rows; 0 means all. Default: 0
  --oracle-pred-dir PATH   Optional third RGB method for an oracle-mask Wan run.
  --lpips                  Enable LPIPS in regional RGB evaluation.
  --skip-wan               Rebuild/validate strict manifest but reuse RGB outputs.
  --skip-mask-eval         Do not run mask evaluation.
  --skip-rgb-eval          Do not run RGB evaluation.
  --execute                Execute commands. Default is dry-run.
  --dry-run                Print commands and write nothing (default).
  -h, --help

The Wan invocation is always pinned to:
  --source-key source_image
  --mask-key predicted_shadow_mask --mask-fallback-key ''
  --tokenlight_mask_tokens --use-mask-input --require-mask-input
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

MANIFEST="${SHADOW_C2F_EVAL_MANIFEST:-}"
WAN_CHECKPOINT="${WAN_CHECKPOINT:-}"
BASELINE_DIR="${BASELINE_DIR:-}"
OUTPUT_ROOT="${WAN_EVAL_OUTPUT_ROOT:-}"
BASE_PATH="${DATA_BASE_PATH:-${WORKSPACE}}"
WEIGHTS_DIR="${WAN_WEIGHTS_DIR:-weights/Wan2.2-TI2V-5B}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${WAN_DEVICE:-cuda}"
GPU_DEVICES="${WAN_GPU_DEVICES:-}"
METRIC_DEVICE="${METRIC_DEVICE:-cpu}"
STEPS="${WAN_STEPS:-50}"
CFG_SCALE="${WAN_CFG_SCALE:-2.0}"
DEFAULT_SEED="${WAN_SEED:-0}"
LIMIT="${LIMIT:-0}"
ORACLE_PRED_DIR="${ORACLE_PRED_DIR:-}"
ENABLE_LPIPS=0
SKIP_WAN=0
SKIP_MASK_EVAL=0
SKIP_RGB_EVAL=0
DRY_RUN=1

while (( $# )); do
  case "$1" in
    --manifest) MANIFEST="${2:?missing value for --manifest}"; shift 2 ;;
    --checkpoint) WAN_CHECKPOINT="${2:?missing value for --checkpoint}"; shift 2 ;;
    --baseline-dir) BASELINE_DIR="${2:?missing value for --baseline-dir}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:?missing value for --output-root}"; shift 2 ;;
    --base-path) BASE_PATH="${2:?missing value for --base-path}"; shift 2 ;;
    --weights-dir) WEIGHTS_DIR="${2:?missing value for --weights-dir}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?missing value for --python}"; shift 2 ;;
    --device) DEVICE="${2:?missing value for --device}"; shift 2 ;;
    --gpu-devices) GPU_DEVICES="${2:?missing value for --gpu-devices}"; shift 2 ;;
    --metric-device) METRIC_DEVICE="${2:?missing value for --metric-device}"; shift 2 ;;
    --steps) STEPS="${2:?missing value for --steps}"; shift 2 ;;
    --cfg-scale) CFG_SCALE="${2:?missing value for --cfg-scale}"; shift 2 ;;
    --seed) DEFAULT_SEED="${2:?missing value for --seed}"; shift 2 ;;
    --limit) LIMIT="${2:?missing value for --limit}"; shift 2 ;;
    --oracle-pred-dir) ORACLE_PRED_DIR="${2:?missing value for --oracle-pred-dir}"; shift 2 ;;
    --lpips) ENABLE_LPIPS=1; shift ;;
    --skip-wan) SKIP_WAN=1; shift ;;
    --skip-mask-eval) SKIP_MASK_EVAL=1; shift ;;
    --skip-rgb-eval) SKIP_RGB_EVAL=1; shift ;;
    --execute) DRY_RUN=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$MANIFEST" ]] || die "--manifest is required"
[[ -n "$WAN_CHECKPOINT" ]] || die "--checkpoint is required"
[[ -n "$BASELINE_DIR" ]] || die "--baseline-dir is required"
[[ -n "$OUTPUT_ROOT" ]] || die "--output-root is required"
[[ "$LIMIT" =~ ^[0-9]+$ ]] || die "--limit must be a non-negative integer"
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || die "--steps must be a positive integer"

MANIFEST="$(realpath -m -- "$MANIFEST")"
WAN_CHECKPOINT="$(realpath -m -- "$WAN_CHECKPOINT")"
BASELINE_DIR="$(realpath -m -- "$BASELINE_DIR")"
OUTPUT_ROOT="$(realpath -m -- "$OUTPUT_ROOT")"
BASE_PATH="$(realpath -m -- "$BASE_PATH")"
if [[ "$WEIGHTS_DIR" = /* ]]; then
  WEIGHTS_DIR="$(realpath -m -- "$WEIGHTS_DIR")"
else
  WEIGHTS_DIR="$(realpath -m -- "${WORKSPACE}/${WEIGHTS_DIR}")"
fi
if [[ -n "$ORACLE_PRED_DIR" ]]; then
  ORACLE_PRED_DIR="$(realpath -m -- "$ORACLE_PRED_DIR")"
fi

STRICT_MANIFEST="${OUTPUT_ROOT}/manifests/wan_strict_no_gt.jsonl"
SANITIZE_SUMMARY="${OUTPUT_ROOT}/manifests/wan_strict_no_gt.summary.json"
PREDICTIONS="${OUTPUT_ROOT}/predictions"
METRICS_ROOT="${OUTPUT_ROOT}/metrics"

if (( ! DRY_RUN )); then
  [[ -f "$MANIFEST" ]] || die "evaluation manifest not found: $MANIFEST"
  [[ -f "$WAN_CHECKPOINT" ]] || die "finished Wan checkpoint not found: $WAN_CHECKPOINT"
  [[ -d "$BASELINE_DIR" ]] || die "baseline prediction directory not found: $BASELINE_DIR"
  [[ -e "$WEIGHTS_DIR" ]] || die "Wan base weights not found: $WEIGHTS_DIR"
  mkdir -p "$(dirname -- "$STRICT_MANIFEST")" "$PREDICTIONS" "$METRICS_ROOT"
fi

sanitize_manifest() {
  if (( DRY_RUN )); then
    printf '[dry-run] sanitize %q -> %q (allowlist, ambient-source check, no GT keys)\n' \
      "$MANIFEST" "$STRICT_MANIFEST"
    return
  fi
  "$PYTHON_BIN" - "$MANIFEST" "$STRICT_MANIFEST" "$SANITIZE_SUMMARY" \
    "$BASE_PATH" "$DEFAULT_SEED" "$LIMIT" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path

from PIL import Image

source_manifest, output_manifest, summary_path, base_path, default_seed, limit = sys.argv[1:]
source_manifest = Path(source_manifest).resolve()
output_manifest = Path(output_manifest).resolve()
summary_path = Path(summary_path).resolve()
base_path = Path(base_path).resolve()
limit = int(limit)

allowed = {
    "scene_id",
    "scene_folder",
    "task",
    "sample_name",
    "light_id",
    "prompt",
    "attrs_json",
    "light_position",
    "light_height",
    "grid_cell",
    "grid_resolution",
    "valid",
    "inference_seed",
    "source_image",
    "predicted_shadow_mask",
}
banned = {"shadow_mask", "shadow_mask_pad16", "gt_shadow_mask", "inf_mask"}


def resolved(value):
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base_path / path).resolve()


rows = []
empty_masks = 0
with source_manifest.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("valid", True) is False:
            continue
        identity = f"{row.get('scene_id', '?')}/{row.get('light_id', '?')}"
        missing = [key for key in ("source_image", "predicted_shadow_mask", "attrs_json") if row.get(key) in (None, "")]
        if missing:
            raise ValueError(f"{source_manifest}:{line_number} {identity}: missing {missing}")

        source = resolved(row["source_image"])
        predicted = resolved(row["predicted_shadow_mask"])
        if not source.is_file():
            raise FileNotFoundError(f"{identity}: ambient source does not exist: {source}")
        if not predicted.is_file():
            raise FileNotFoundError(f"{identity}: predicted mask does not exist: {predicted}")

        original_value = row.get("input_image")
        if original_value not in (None, ""):
            original = resolved(original_value)
            if not original.is_file():
                raise FileNotFoundError(
                    f"{identity}: cannot verify original input_image under --base-path: {original}"
                )
            if original != source:
                raise ValueError(
                    f"{identity}: source_image is not original input_image: {source} != {original}"
                )
        for comparison_key in ("baseline_image", "target_image", "video"):
            value = row.get(comparison_key)
            if value not in (None, "") and resolved(value) == source:
                raise ValueError(f"{identity}: source_image aliases {comparison_key}")
        gt_value = row.get("gt_shadow_mask") or row.get("shadow_mask")
        if gt_value not in (None, "") and resolved(gt_value) == predicted:
            raise ValueError(f"{identity}: predicted_shadow_mask aliases renderer GT")

        with Image.open(predicted) as image:
            gray = image.convert("L")
            if gray.size != (480, 480):
                raise ValueError(f"{identity}: predicted mask is {gray.size}, expected (480, 480)")
            histogram = gray.histogram()
        unexpected = [index for index, count in enumerate(histogram) if count and index not in (0, 255)]
        if unexpected:
            raise ValueError(f"{identity}: predicted mask is not hard 0/255: {unexpected[:8]}")
        empty_masks += int(histogram[255] == 0)

        clean = {key: row[key] for key in allowed if key in row}
        clean["source_image"] = source.as_posix()
        clean["predicted_shadow_mask"] = predicted.as_posix()
        clean.setdefault("inference_seed", int(default_seed))
        leaked = banned.intersection(clean)
        unexpected_shadow_keys = {
            key for key in clean if "shadow_mask" in key and key != "predicted_shadow_mask"
        }
        if leaked or unexpected_shadow_keys:
            raise AssertionError(f"{identity}: GT field leaked: {sorted(leaked | unexpected_shadow_keys)}")
        rows.append(clean)
        if limit > 0 and len(rows) >= limit:
            break

if not rows:
    raise RuntimeError(f"No valid rows selected from {source_manifest}")

output_manifest.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{output_manifest.name}.", suffix=".tmp", dir=output_manifest.parent
)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary_name, output_manifest)
except Exception:
    try:
        os.unlink(temporary_name)
    except FileNotFoundError:
        pass
    raise

summary = {
    "schema": "tokenlight_wan_strict_no_gt_manifest_v1",
    "source_manifest": source_manifest.as_posix(),
    "output_manifest": output_manifest.as_posix(),
    "rows": len(rows),
    "empty_predicted_masks": empty_masks,
    "source_key": "source_image",
    "source_contract": "original ambient input_image",
    "mask_key": "predicted_shadow_mask",
    "mask_fallback": None,
    "removed_evaluation_fields": sorted(banned | {"target_image", "video"}),
    "contains_gt_shadow_mask": False,
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

sanitize_manifest

LIMIT_ARGS=()
if (( LIMIT > 0 )); then
  LIMIT_ARGS=(--limit "$LIMIT")
fi
GPU_ARGS=()
if [[ -n "$GPU_DEVICES" ]]; then
  GPU_ARGS=(--gpu-devices "$GPU_DEVICES")
fi

if (( ! SKIP_WAN )); then
  WAN_COMMAND=(
    "$PYTHON_BIN" "${WORKSPACE}/scripts/infer_manifest.py"
    --manifest "$STRICT_MANIFEST"
    --base-path "$WORKSPACE"
    --output-dir "$PREDICTIONS"
    --checkpoint "$WAN_CHECKPOINT"
    --weights_dir "$WEIGHTS_DIR"
    --source-key source_image
    --target-key __disabled_target__
    --target-fallback-key ""
    --mask-key predicted_shadow_mask
    --mask-fallback-key ""
    --tokenlight_mask_tokens
    --use-mask-input
    --require-mask-input
    --no-with-gt
    --height 480
    --width 480
    --num_frames 1
    --num_inference_steps "$STEPS"
    --cfg_scale "$CFG_SCALE"
    --seed-key inference_seed
    --tokenlight_max_lights 2
    --device "$DEVICE"
    --skip-existing
    "${GPU_ARGS[@]}"
    "${LIMIT_ARGS[@]}"
  )
  run_command "${WAN_COMMAND[@]}"
fi

if (( ! SKIP_MASK_EVAL )); then
  MASK_EVAL_COMMAND=(
    "$PYTHON_BIN" "${WORKSPACE}/utils/evaluate_shadow_c2f.py"
    --manifest "$MANIFEST"
    --method "adapter_target=adapter_target_cache#final_prob"
    --method "adapter_delta=adapter_delta_cache#positive_delta_final_prob"
    --method "refined_probability=refined_probability"
    --method "refined_binary=predicted_shadow_mask"
    --threshold refined_binary=0.5
    --output "${METRICS_ROOT}/mask_metrics.json"
    "${LIMIT_ARGS[@]}"
  )
  run_command "${MASK_EVAL_COMMAND[@]}"
fi

if (( ! SKIP_RGB_EVAL )); then
  RGB_EVAL_COMMAND=(
    "$PYTHON_BIN" "${WORKSPACE}/utils/evaluate_shadow_conditioned_rgb.py"
    --manifest "$MANIFEST"
    --prediction "baseline=${BASELINE_DIR}"
    --prediction "refined=${PREDICTIONS}"
    --output "${METRICS_ROOT}/rgb_metrics.json"
    --device "$METRIC_DEVICE"
    "${LIMIT_ARGS[@]}"
  )
  if [[ -n "$ORACLE_PRED_DIR" ]]; then
    RGB_EVAL_COMMAND+=(--prediction "oracle=${ORACLE_PRED_DIR}")
  fi
  if (( ENABLE_LPIPS )); then
    RGB_EVAL_COMMAND+=(--lpips)
  fi
  run_command "${RGB_EVAL_COMMAND[@]}"
fi

if (( DRY_RUN )); then
  printf 'Dry-run only. Add --execute after checkpoints are complete and training GPUs are free.\n'
else
  printf 'Completed strict Wan + evaluation pipeline: %s\n' "$OUTPUT_ROOT"
fi
