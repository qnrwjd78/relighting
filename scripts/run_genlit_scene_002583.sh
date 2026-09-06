#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN=/workspace/miniconda3/bin/conda
ENV_PREFIX=/workspace/conda_envs/genlit
REPO_DIR=/workspace/repos/genlit
WEIGHT_ROOT=/workspace/weights/genlit
BASE_MODEL_DIR="${WEIGHT_ROOT}/stable-video-diffusion-img2vid"
CHECKPOINT_DIR="${WEIGHT_ROOT}/checkpoint/single_object"
INPUT_JSON=/workspace/scripts/genlit_scene_002583.json
OUTPUT_DIR=/workspace/outputs/infer/relighting_external/scene_002583/genlit
FRAME_DIR="${OUTPUT_DIR}/validation_images_single/checkpoint/videos/scene_002583/0_images"
MP4_PATH="${OUTPUT_DIR}/scene_002583_trajectory_0_7fps.mp4"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CONDA_PKGS_DIRS=/workspace/.conda_pkgs
export PIP_CACHE_DIR=/workspace/.pip_cache
export HF_HOME="${WEIGHT_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"
export HF_HUB_DISABLE_XET=1
export TORCH_HOME="${WEIGHT_ROOT}/torch"
export XDG_CACHE_HOME="${WEIGHT_ROOT}/cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT_DIR}" "${HF_HUB_CACHE}" "${HF_XET_CACHE}" "${TORCH_HOME}"
if [[ ! -f "${BASE_MODEL_DIR}/model_index.json" ]]; then
  echo "Missing SVD base model: ${BASE_MODEL_DIR}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT_DIR}/controlnet/diffusion_pytorch_model.safetensors" ]]; then
  echo "Missing gated GenLit checkpoint: ${CHECKPOINT_DIR}/controlnet" >&2
  exit 1
fi
cd "${REPO_DIR}"

"${CONDA_BIN}" run --no-capture-output --prefix "${ENV_PREFIX}" \
  python -m genlit.inference \
    --mode single \
    --img_json "${INPUT_JSON}" \
    --trajectory_indices 0 \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --pretrained_model_name_or_path "${BASE_MODEL_DIR}" \
    --num_frames 14 \
    --num_inference_steps 50 \
    --num_workers 0 \
    --output_dir "${OUTPUT_DIR}"

if [[ ! -f "${FRAME_DIR}/014.png" ]]; then
  echo "Expected 14 generated frames plus input frame under ${FRAME_DIR}" >&2
  exit 1
fi
png_count=$(find "${FRAME_DIR}" -maxdepth 1 -type f -name '*.png' | wc -l)
if [[ "${png_count}" -ne 15 ]]; then
  echo "Expected 15 PNG files (source + 14 generated), found ${png_count}" >&2
  exit 1
fi

# Frame 000 is the original source image. Encode generated frames 001..014 only.
ffmpeg -y -loglevel error \
  -framerate 7 -start_number 1 -i "${FRAME_DIR}/%03d.png" \
  -frames:v 14 -c:v libx264 -pix_fmt yuv420p -crf 18 "${MP4_PATH}"

video_probe=$(ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=width,height,nb_read_frames \
  -of csv=p=0 "${MP4_PATH}")
if [[ "${video_probe}" != "512,512,14" ]]; then
  echo "Unexpected MP4 stream metadata: ${video_probe}" >&2
  exit 1
fi

echo "frames=${FRAME_DIR}"
echo "png_count=${png_count}"
echo "video=${MP4_PATH}"
echo "video_stream=${video_probe}"
