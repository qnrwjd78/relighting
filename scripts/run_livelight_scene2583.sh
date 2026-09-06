#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-1}"
REPO=/workspace/repos/LiveLight
ENV_PREFIX=/workspace/conda_envs/livelight
WEIGHTS=/workspace/weights/livelight
SOURCE=/workspace/data/objaverse_245_eval_2500_2999/unseen_7x7x5_power06_png_exclude_requested/scenes/scene_002583/source.png
POINT_MAP=/workspace/data/objaverse_245_eval_2500_2999/unseen_7x7x5_power06_source/scene_002583/source.npy
PBR_DEPTH=/workspace/data/objaverse_245_eval_2500_2999/unseen_7x7x5_power06_png_exclude_requested/scenes/scene_002583/pbr/depth.png
OUTPUT=/workspace/outputs/infer/relighting_external/scene_002583/livelight
CONFIG=/workspace/scripts/livelight_stage1_scene2583.yaml

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HOME="$WEIGHTS/.hf_home"
export HF_HUB_CACHE="$WEIGHTS/.hf_cache"
export MODELSCOPE_CACHE="$WEIGHTS/.modelscope_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1

mkdir -p "$OUTPUT"

required_files=(
  "$WEIGHTS/LiveLight/denoising_unet-110000.pth"
  "$WEIGHTS/LiveLight/reference_unet-110000.pth"
  "$WEIGHTS/LiveLight/light_guider-110000.pth"
  "$WEIGHTS/sd-image-variations-diffusers/unet/config.json"
  "$WEIGHTS/sd-image-variations-diffusers/unet/diffusion_pytorch_model.bin"
  "$WEIGHTS/sd-image-variations-diffusers/image_encoder/config.json"
  "$WEIGHTS/sd-image-variations-diffusers/image_encoder/pytorch_model.bin"
  "$WEIGHTS/sd-vae-ft-mse/config.json"
  "$WEIGHTS/sd-vae-ft-mse/diffusion_pytorch_model.bin"
)
for required_file in "${required_files[@]}"; do
  if [[ ! -s "$required_file" ]]; then
    echo "Missing or empty required LiveLight file: $required_file" >&2
    exit 1
  fi
done
if compgen -G "$WEIGHTS/LiveLight/*.aria2" >/dev/null; then
  echo "Official LiveLight checkpoints are still downloading (*.aria2 exists)." >&2
  exit 1
fi

"$ENV_PREFIX/bin/python" /workspace/scripts/prepare_livelight_depth.py \
  --point-map "$POINT_MAP" \
  --pbr-depth-png "$PBR_DEPTH" \
  --output-depth "$OUTPUT/depth.npy" \
  --output-vis "$OUTPUT/depth_vis.png" \
  --output-mask "$OUTPUT/depth_valid_mask.png" \
  --output-report "$OUTPUT/depth_report.json" \
  --width 512 --height 512 \
  --canonical-anchor-depth 256

cd "$REPO"
"$ENV_PREFIX/bin/python" inference_livelight_stage1.py \
  --input-image "$SOURCE" \
  --depth-npy "$OUTPUT/depth.npy" \
  --output-dir "$OUTPUT" \
  --ckpt-dir "$WEIGHTS/LiveLight" \
  --train-config "$CONFIG" \
  --device cuda \
  --num-inference-steps 50 \
  --seed 42 \
  --use-xformers \
  --light-u 0.5 \
  --light-v 0.35 \
  --light-z-rel 0.66 \
  --light-intensity 1.0 \
  --light-color 1.0,1.0,1.0 \
  2>&1 | tee "$OUTPUT/run.log"
