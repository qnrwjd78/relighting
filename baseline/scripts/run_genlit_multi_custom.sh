#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN=/workspace/miniconda3/bin/conda
ENV_PREFIX=/workspace/conda_envs/genlit
REPO_DIR=/workspace/baseline/repos/genlit
WEIGHT_ROOT=/workspace/weights/genlit
BASE_MODEL_DIR="${WEIGHT_ROOT}/stable-video-diffusion-img2vid-xt"
CHECKPOINT_DIR="${WEIGHT_ROOT}/checkpoint/multi_object"
INTENSITY_HELPER=/workspace/baseline/scripts/make_genlit_multi_intensity_variant.py

usage() {
  cat <<'EOF'
Usage:
  run_genlit_multi_custom.sh \
    --input-json FILE --sequences-file FILE --output-dir DIR \
    [--gpu 0] [--trajectory-indices 0] [--point-intensities 25,50,75] \
    [--steps 50] [--num-workers 0]

Without --point-intensities, the trajectory's original per-frame intensities are used.
With one or more comma-separated candidates, an intensity-specific manifest and output
subdirectory are created for every candidate.
EOF
}

input_json=
sequences_file=
output_dir=
gpu=0
trajectory_indices=0
point_intensities=
steps=50
num_workers=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-json) input_json=$2; shift 2 ;;
    --sequences-file) sequences_file=$2; shift 2 ;;
    --output-dir) output_dir=$2; shift 2 ;;
    --gpu) gpu=$2; shift 2 ;;
    --trajectory-indices) trajectory_indices=$2; shift 2 ;;
    --point-intensities) point_intensities=$2; shift 2 ;;
    --steps) steps=$2; shift 2 ;;
    --num-workers) num_workers=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${input_json}" ]] || { echo "--input-json is required" >&2; exit 2; }
[[ -n "${sequences_file}" ]] || { echo "--sequences-file is required" >&2; exit 2; }
[[ -n "${output_dir}" ]] || { echo "--output-dir is required" >&2; exit 2; }
[[ -f "${input_json}" ]] || { echo "Missing input JSON: ${input_json}" >&2; exit 1; }
[[ -f "${sequences_file}" ]] || { echo "Missing trajectory: ${sequences_file}" >&2; exit 1; }
[[ -f "${BASE_MODEL_DIR}/model_index.json" ]] || { echo "Missing SVD-XT model" >&2; exit 1; }
[[ -f "${CHECKPOINT_DIR}/controlnet/diffusion_pytorch_model.safetensors" ]] || \
  { echo "Missing GenLit multi checkpoint" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${gpu}"
export CONDA_PKGS_DIRS=/workspace/.conda_pkgs
export PIP_CACHE_DIR=/workspace/.pip_cache
export HF_HOME="${WEIGHT_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="${WEIGHT_ROOT}/torch"
export XDG_CACHE_HOME="${WEIGHT_ROOT}/cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${output_dir}" "${HF_HUB_CACHE}" "${HF_XET_CACHE}" "${TORCH_HOME}"

run_one() {
  local active_sequences=$1
  local active_output=$2
  mkdir -p "${active_output}"
  cd "${REPO_DIR}"
  "${CONDA_BIN}" run --no-capture-output --prefix "${ENV_PREFIX}" \
    python -m genlit.inference \
      --mode multi \
      --img_json "${input_json}" \
      --sequences_file "${active_sequences}" \
      --trajectory_indices "${trajectory_indices}" \
      --checkpoint_dir "${CHECKPOINT_DIR}" \
      --pretrained_model_name_or_path "${BASE_MODEL_DIR}" \
      --num_frames 25 \
      --width 640 \
      --height 448 \
      --conditioning_channels 5 \
      --num_inference_steps "${steps}" \
      --num_workers "${num_workers}" \
      --output_dir "${active_output}"
}

if [[ -z "${point_intensities}" ]]; then
  run_one "${sequences_file}" "${output_dir}"
else
  IFS=',' read -r -a candidates <<< "${point_intensities}"
  for raw in "${candidates[@]}"; do
    intensity=$(echo "${raw}" | xargs)
    [[ -n "${intensity}" ]] || continue
    slug=${intensity//./p}
    slug=${slug//-/_neg_}
    variant="${output_dir}/manifests/trajectory_point_${slug}.npy"
    candidate_output="${output_dir}/point_${slug}"
    "${CONDA_BIN}" run --no-capture-output --prefix "${ENV_PREFIX}" \
      python "${INTENSITY_HELPER}" \
        --input "${sequences_file}" \
        --output "${variant}" \
        --point-intensity "${intensity}"
    run_one "${variant}" "${candidate_output}"
  done
fi
