#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 {baseline|delta|pointmap} CHECKPOINT RUN_ROOT" >&2
  exit 2
fi

kind="$1"
checkpoint="$2"
run_root="$3"
manifest="data_train/objaverse_245_eval_2500_2999_rgb_random3/metadata_random3_heights.jsonl"
base_path="data/objaverse_245_eval_2500_2999/unseen_7x7x5_power06_png_exclude_requested"
predictions="$run_root/predictions"

mkdir -p "$predictions" "$run_root/videos"
python - "$kind" "$checkpoint" "$run_root" <<'PY'
import json, os, sys
from pathlib import Path
kind, checkpoint, run_root = sys.argv[1:]
payload = {
    "kind": kind,
    "checkpoint": str(Path(checkpoint).resolve()),
    "run_root": str(Path(run_root).resolve()),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "source_condition": "RGB PNG encoded by Wan VAE",
    "num_inference_steps": 50,
    "cfg_scale": 2.0,
    "inference_seed_key": "inference_seed",
}
Path(run_root).mkdir(parents=True, exist_ok=True)
(Path(run_root) / "execution.json").write_text(json.dumps(payload, indent=2) + "\n")
PY

common_args=(
  --manifest "$manifest"
  --base-path "$base_path"
  --output-dir "$predictions"
  --checkpoint "$checkpoint"
  --weights_dir weights/Wan2.2-TI2V-5B
  --height 480 --width 480 --num_inference_steps 50 --cfg_scale 2.0
  --seed-key inference_seed --tokenlight_max_lights 2
  --no-tokenlight_mask_tokens --no-use-mask-input --no-with-gt
  --skip-existing --eval --no-allow-missing --metric-device cuda
)

if [[ "$kind" == "pointmap" ]]; then
  python scripts/infer_manifest_pointmap_rgb_npy.py "${common_args[@]}"
elif [[ "$kind" == "baseline" || "$kind" == "delta" ]]; then
  python scripts/infer_manifest.py "${common_args[@]}"
else
  echo "unsupported kind: $kind" >&2
  exit 2
fi

python scripts/summarize_metrics_by_height.py \
  --metrics "$predictions/metrics.json" \
  --manifest "$manifest" \
  --output "$run_root/metrics_by_height.json"

python scripts/make_height_output_gt_videos_rgb.py \
  --predictions "$predictions" \
  --manifest "$manifest" \
  --base-path "$base_path" \
  --output-dir "$run_root/videos" \
  --fps 10
