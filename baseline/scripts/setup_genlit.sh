#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN=/workspace/miniconda3/bin/conda
ENV_PREFIX=/workspace/conda_envs/genlit
REPO_DIR=/workspace/baseline/repos/genlit
WEIGHT_ROOT=/workspace/weights/genlit

export CONDA_PKGS_DIRS=/workspace/.conda_pkgs
export PIP_CACHE_DIR=/workspace/.pip_cache
export HF_HOME="${WEIGHT_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"
export TORCH_HOME="${WEIGHT_ROOT}/torch"
export XDG_CACHE_HOME="${WEIGHT_ROOT}/cache"

mkdir -p \
  /workspace/conda_envs \
  "${WEIGHT_ROOT}" \
  "${HF_HUB_CACHE}" \
  "${HF_XET_CACHE}" \
  "${TORCH_HOME}" \
  "${XDG_CACHE_HOME}"

if [[ ! -f "${REPO_DIR}/pyproject.toml" ]]; then
  echo "Missing repository: ${REPO_DIR}" >&2
  exit 1
fi

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  "${CONDA_BIN}" create --yes --override-channels --channel conda-forge \
    --prefix "${ENV_PREFIX}" python=3.10 pip
fi

"${CONDA_BIN}" run --prefix "${ENV_PREFIX}" \
  python -m pip install --upgrade pip setuptools wheel
"${CONDA_BIN}" run --prefix "${ENV_PREFIX}" \
  python -m pip install --editable "${REPO_DIR}"

"${CONDA_BIN}" run --no-capture-output --prefix "${ENV_PREFIX}" python - <<'PY'
import torch
import diffusers
import transformers
import huggingface_hub

print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
print(f"diffusers={diffusers.__version__}")
print(f"transformers={transformers.__version__}")
print(f"huggingface_hub={huggingface_hub.__version__}")
PY
