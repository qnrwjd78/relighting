#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN=/workspace/miniconda3/bin/conda
ENV_PREFIX=/workspace/conda_envs/genlit
WEIGHT_ROOT=/workspace/weights/genlit
BASE_MODEL_DIR="${WEIGHT_ROOT}/stable-video-diffusion-img2vid"
CHECKPOINT_ROOT="${WEIGHT_ROOT}/checkpoint"

export HF_HOME="${WEIGHT_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"
# Direct Hub downloads are markedly faster than the Xet client on this host.
export HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME="${WEIGHT_ROOT}/cache"

mkdir -p "${BASE_MODEL_DIR}" "${CHECKPOINT_ROOT}" "${HF_HUB_CACHE}" "${HF_XET_CACHE}"

verify_file() {
  local rel_path=$1
  local expected_size=$2
  local expected_sha=$3
  local full_path="${BASE_MODEL_DIR}/${rel_path}"
  [[ -f "${full_path}" ]] || return 1
  [[ "$(stat -c %s "${full_path}")" == "${expected_size}" ]] || return 1
  [[ "$(sha256sum "${full_path}" | awk '{print $1}')" == "${expected_sha}" ]]
}

svd_ready=true
verify_file image_encoder/model.fp16.safetensors \
  1264217240 ae616c24393dd1854372b0639e5541666f7521cbe219669255e865cb7f89466a || svd_ready=false
verify_file unet/diffusion_pytorch_model.fp16.safetensors \
  3049435868 cb552818963e736506c6693ffc279d59df423d63aced38902ea4373ab1fd2932 || svd_ready=false
verify_file vae/diffusion_pytorch_model.fp16.safetensors \
  195531910 af602cd0eb4ad6086ec94fbf1438dfb1be5ec9ac03fd0215640854e90d6463a3 || svd_ready=false

for config_file in \
  model_index.json \
  feature_extractor/preprocessor_config.json \
  image_encoder/config.json \
  scheduler/scheduler_config.json \
  unet/config.json \
  vae/config.json; do
  [[ -f "${BASE_MODEL_DIR}/${config_file}" ]] || svd_ready=false
done

if [[ "${svd_ready}" != true ]]; then
  "${CONDA_BIN}" run --no-capture-output --prefix "${ENV_PREFIX}" \
    hf download stabilityai/stable-video-diffusion-img2vid \
      model_index.json \
      feature_extractor/preprocessor_config.json \
      image_encoder/config.json \
      image_encoder/model.fp16.safetensors \
      scheduler/scheduler_config.json \
      unet/config.json \
      unet/diffusion_pytorch_model.fp16.safetensors \
      vae/config.json \
      vae/diffusion_pytorch_model.fp16.safetensors \
      --local-dir "${BASE_MODEL_DIR}"
fi

verify_file image_encoder/model.fp16.safetensors \
  1264217240 ae616c24393dd1854372b0639e5541666f7521cbe219669255e865cb7f89466a
verify_file unet/diffusion_pytorch_model.fp16.safetensors \
  3049435868 cb552818963e736506c6693ffc279d59df423d63aced38902ea4373ab1fd2932
verify_file vae/diffusion_pytorch_model.fp16.safetensors \
  195531910 af602cd0eb4ad6086ec94fbf1438dfb1be5ec9ac03fd0215640854e90d6463a3

# The upstream GenLit loader requests the default filenames while casting to fp16.
# Point those names at the official fp16 SVD variants to avoid downloading duplicate
# fp32 tensors and to keep the local model self-contained.
[[ -e "${BASE_MODEL_DIR}/image_encoder/model.safetensors" ]] || \
  ln -s model.fp16.safetensors "${BASE_MODEL_DIR}/image_encoder/model.safetensors"
[[ -e "${BASE_MODEL_DIR}/unet/diffusion_pytorch_model.safetensors" ]] || \
  ln -s diffusion_pytorch_model.fp16.safetensors \
    "${BASE_MODEL_DIR}/unet/diffusion_pytorch_model.safetensors"
[[ -e "${BASE_MODEL_DIR}/vae/diffusion_pytorch_model.safetensors" ]] || \
  ln -s diffusion_pytorch_model.fp16.safetensors \
    "${BASE_MODEL_DIR}/vae/diffusion_pytorch_model.safetensors"

"${CONDA_BIN}" run --no-capture-output --prefix "${ENV_PREFIX}" \
  hf download sbharadwaj/genlit \
    single_object/controlnet/config.json \
    single_object/controlnet/diffusion_pytorch_model.safetensors \
    --local-dir "${CHECKPOINT_ROOT}"
