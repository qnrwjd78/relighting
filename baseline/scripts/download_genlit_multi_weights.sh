#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN=/workspace/miniconda3/bin/conda
ENV_PREFIX=/workspace/conda_envs/genlit
WEIGHT_ROOT=/workspace/weights/genlit
BASE_MODEL_DIR="${WEIGHT_ROOT}/stable-video-diffusion-img2vid-xt"
CHECKPOINT_ROOT="${WEIGHT_ROOT}/checkpoint"

# Pin both official repositories so local verification is reproducible.
GENLIT_REVISION=066945c322286b06320d76b9094d60d6e368bbaf
SVD_XT_REVISION=9e43909513c6714f1bc78bcb44d96e733cd242aa

export HF_HOME="${WEIGHT_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"
export HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME="${WEIGHT_ROOT}/cache"

mkdir -p "${BASE_MODEL_DIR}" "${CHECKPOINT_ROOT}" "${HF_HUB_CACHE}" "${HF_XET_CACHE}"

verify_file() {
  local root=$1
  local rel_path=$2
  local expected_size=$3
  local expected_sha=$4
  local full_path="${root}/${rel_path}"
  [[ -f "${full_path}" ]]
  [[ "$(stat -c %s "${full_path}")" == "${expected_size}" ]]
  [[ "$(sha256sum "${full_path}" | awk '{print $1}')" == "${expected_sha}" ]]
}

svd_ready=true
verify_file "${BASE_MODEL_DIR}" image_encoder/model.fp16.safetensors \
  1264217240 ae616c24393dd1854372b0639e5541666f7521cbe219669255e865cb7f89466a || svd_ready=false
verify_file "${BASE_MODEL_DIR}" unet/diffusion_pytorch_model.fp16.safetensors \
  3049435868 9fbc02e90f37d422f5e3a4aeaee95f6629dc8c45ca211b951626e930daf2bddf || svd_ready=false
verify_file "${BASE_MODEL_DIR}" vae/diffusion_pytorch_model.fp16.safetensors \
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
    hf download stabilityai/stable-video-diffusion-img2vid-xt \
      model_index.json \
      feature_extractor/preprocessor_config.json \
      image_encoder/config.json \
      image_encoder/model.fp16.safetensors \
      scheduler/scheduler_config.json \
      unet/config.json \
      unet/diffusion_pytorch_model.fp16.safetensors \
      vae/config.json \
      vae/diffusion_pytorch_model.fp16.safetensors \
      --revision "${SVD_XT_REVISION}" \
      --local-dir "${BASE_MODEL_DIR}"
fi

verify_file "${BASE_MODEL_DIR}" image_encoder/model.fp16.safetensors \
  1264217240 ae616c24393dd1854372b0639e5541666f7521cbe219669255e865cb7f89466a
verify_file "${BASE_MODEL_DIR}" unet/diffusion_pytorch_model.fp16.safetensors \
  3049435868 9fbc02e90f37d422f5e3a4aeaee95f6629dc8c45ca211b951626e930daf2bddf
verify_file "${BASE_MODEL_DIR}" vae/diffusion_pytorch_model.fp16.safetensors \
  195531910 af602cd0eb4ad6086ec94fbf1438dfb1be5ec9ac03fd0215640854e90d6463a3

# The upstream loader does not pass variant="fp16". Keep the official fp16 files and
# expose the default filenames as local symlinks instead of downloading fp32 copies.
[[ -e "${BASE_MODEL_DIR}/image_encoder/model.safetensors" ]] || \
  ln -s model.fp16.safetensors "${BASE_MODEL_DIR}/image_encoder/model.safetensors"
[[ -e "${BASE_MODEL_DIR}/unet/diffusion_pytorch_model.safetensors" ]] || \
  ln -s diffusion_pytorch_model.fp16.safetensors \
    "${BASE_MODEL_DIR}/unet/diffusion_pytorch_model.safetensors"
[[ -e "${BASE_MODEL_DIR}/vae/diffusion_pytorch_model.safetensors" ]] || \
  ln -s diffusion_pytorch_model.fp16.safetensors \
    "${BASE_MODEL_DIR}/vae/diffusion_pytorch_model.safetensors"

multi_ready=true
verify_file "${CHECKPOINT_ROOT}" multi_object/controlnet/diffusion_pytorch_model.safetensors \
  2728175364 779b0d8a1a4b90d0be6d10cb0c3ca22e26d3b5a93bdbd4f4f28f23405090e9b4 || multi_ready=false
[[ -f "${CHECKPOINT_ROOT}/multi_object/controlnet/config.json" ]] || multi_ready=false

if [[ "${multi_ready}" != true ]]; then
  "${CONDA_BIN}" run --no-capture-output --prefix "${ENV_PREFIX}" \
    hf download sbharadwaj/genlit \
      multi_object/controlnet/config.json \
      multi_object/controlnet/diffusion_pytorch_model.safetensors \
      --revision "${GENLIT_REVISION}" \
      --local-dir "${CHECKPOINT_ROOT}"
fi

verify_file "${CHECKPOINT_ROOT}" multi_object/controlnet/diffusion_pytorch_model.safetensors \
  2728175364 779b0d8a1a4b90d0be6d10cb0c3ca22e26d3b5a93bdbd4f4f28f23405090e9b4
[[ -f "${CHECKPOINT_ROOT}/multi_object/controlnet/config.json" ]]

echo "genlit_revision=${GENLIT_REVISION}"
echo "svd_xt_revision=${SVD_XT_REVISION}"
echo "base_model=${BASE_MODEL_DIR}"
echo "checkpoint=${CHECKPOINT_ROOT}/multi_object"
echo "status=verified"
