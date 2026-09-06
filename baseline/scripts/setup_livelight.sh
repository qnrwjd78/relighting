#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace/baseline/repos/LiveLight
ENV_PREFIX=/workspace/conda_envs/livelight
WEIGHTS=/workspace/weights/livelight
CONDA=/workspace/miniconda3/bin/conda
export CONDA_PKGS_DIRS=/workspace/.conda_pkgs
export PIP_CACHE_DIR=/workspace/.pip_cache
export HF_HOME="$WEIGHTS/.hf_home"
export HF_HUB_CACHE="$WEIGHTS/.hf_cache"
export MODELSCOPE_CACHE="$WEIGHTS/.modelscope_cache"

mkdir -p /workspace/baseline/repos /workspace/conda_envs "$WEIGHTS"
if [[ ! -f "$REPO/requirements.txt" ]]; then
  echo "Missing baseline source: $REPO" >&2
  exit 1
fi
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$CONDA" create -y -p "$ENV_PREFIX" --override-channels -c conda-forge python=3.10 pip
fi

# The upstream requirements leave Gradio and MLflow unbounded. Pin versions that
# remain compatible with its Pillow 11.3 and Hugging Face Hub 0.25.1 pins.
"$ENV_PREFIX/bin/python" -m pip install \
  --find-links="$WEIGHTS/wheelhouse" \
  -r "$REPO/requirements.txt" \
  modelscope==1.31.0 gradio==5.14.0 mlflow==3.6.0
"$ENV_PREFIX/bin/python" -m pip check
"$ENV_PREFIX/bin/python" "$REPO/download_weights.py" \
  --output-dir "$WEIGHTS/LiveLight"
"$ENV_PREFIX/bin/python" /workspace/baseline/scripts/download_livelight_models.py \
  --weights-root "$WEIGHTS"
"$ENV_PREFIX/bin/python" /workspace/baseline/scripts/verify_livelight_weights.py \
  --weights-root "$WEIGHTS" \
  --report "$WEIGHTS/verification.json"

df -h /workspace
du -sh "$REPO" "$ENV_PREFIX" "$WEIGHTS"
