from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.pretrain_weight import validate_wan22_weights, wan22_model_paths, wan22_tokenizer_path


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wan2.2-TI2V-5B inference with local DiffSynth weights.")
    parser.add_argument("--weights_dir", default="weights/Wan2.2-TI2V-5B")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--input_image", default="")
    parser.add_argument("--output", default="outputs/wan22_ti2v_5b.mp4")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1248)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tiled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quality", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_wan22_weights(args.weights_dir)
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    model_paths = wan22_model_paths(args.weights_dir)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_paths[1]),
            ModelConfig(path=model_paths[0]),
            ModelConfig(model_paths[2]),
        ],
        tokenizer_config=ModelConfig(wan22_tokenizer_path(args.weights_dir)),
    )

    input_image = None
    if args.input_image:
        input_image = Image.open(args.input_image).convert("RGB").resize((args.width, args.height))

    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        tiled=args.tiled,
        height=args.height,
        width=args.width,
        input_image=input_image,
        num_frames=args.num_frames,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        sigma_shift=args.sigma_shift,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_video(video, str(output), fps=args.fps, quality=args.quality)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
