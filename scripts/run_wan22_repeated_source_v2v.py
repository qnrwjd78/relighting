#!/usr/bin/env python3
"""Run Wan2.2-TI2V-5B V2V from a still image repeated over time.

This is a scene-preserving text-edit baseline: the resized source image is
repeated for every input-video frame, then a configurable amount of noise is
added by DiffSynth before denoising with the supplied prompt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.pretrain_weight import (  # noqa: E402
    validate_wan22_weights,
    wan22_model_paths,
    wan22_tokenizer_path,
)


DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "Wan2.2-TI2V-5B"
DEFAULT_NEGATIVE_PROMPT = (
    "visible light beam, visible light cone, visible lamp, moving light source, "
    "overexposure, blown highlights, flicker, pulsing light, camera movement, "
    "object movement, geometry change, material change, background change, "
    "color cast, haze, smoke, text, watermark, low quality, blurry, artifacts"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wan2.2-TI2V-5B video-to-video inference using a still RGB source "
            "repeated for every input frame."
        )
    )
    parser.add_argument(
        "--weights",
        "--weights-dir",
        "--weights_dir",
        dest="weights",
        default=str(DEFAULT_WEIGHTS),
        help="Local Wan2.2-TI2V-5B weight directory.",
    )
    parser.add_argument(
        "--input",
        "--input-image",
        "--input_image",
        dest="input",
        required=True,
        help="Source image path.",
    )
    parser.add_argument("--output", required=True, help="Output video path; must not already exist.")
    parser.add_argument("--prompt", required=True, help="Positive text prompt.")
    parser.add_argument(
        "--negative",
        "--negative-prompt",
        "--negative_prompt",
        dest="negative",
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative text prompt.",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--num-frames", "--num_frames", dest="num_frames", type=int, default=49)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--denoising-strength",
        "--denoising_strength",
        dest="denoising_strength",
        type=float,
        default=0.25,
    )
    parser.add_argument("--cfg", type=float, default=3.5, help="Classifier-free guidance scale.")
    parser.add_argument("--steps", type=int, default=50, help="Number of diffusion inference steps.")
    parser.add_argument("--sigma", type=float, default=5.0, help="DiffSynth scheduler sigma shift.")
    parser.add_argument(
        "--tiled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable VAE tiling (use --no-tiled to disable).",
    )

    args = parser.parse_args(argv)
    _validate_args(args, parser)
    return args


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    source = Path(args.input).expanduser()
    output = Path(args.output).expanduser()

    if not source.is_file():
        parser.error(f"input image does not exist or is not a file: {source}")
    if output.exists():
        parser.error(f"refusing to overwrite existing output: {output}")
    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if args.width % 16 or args.height % 16:
        parser.error("--width and --height must both be divisible by 16")
    if args.num_frames < 5 or (args.num_frames - 1) % 4:
        parser.error("--num-frames must be at least 5 and have the form 4k+1 (for example, 49)")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if not 0.0 < args.denoising_strength <= 1.0:
        parser.error("--denoising-strength must be in the interval (0, 1]")
    if args.cfg <= 0:
        parser.error("--cfg must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.sigma <= 0:
        parser.error("--sigma must be positive")


def _load_source(path: Path, width: int, height: int) -> Image.Image:
    try:
        with Image.open(path) as opened:
            source = ImageOps.exif_transpose(opened).convert("RGB")
            source.load()
    except Exception as exc:
        raise RuntimeError(f"failed to load source image {path}: {exc}") from exc
    return source.resize((width, height), resample=Image.Resampling.LANCZOS)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    weights_path = Path(args.weights).expanduser().resolve()

    # Fail on incomplete local weights before importing/loading the large models.
    validate_wan22_weights(weights_path)

    import torch
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
    from diffsynth.utils.data import save_video

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Wan2.2-TI2V-5B inference")

    source = _load_source(source_path, args.width, args.height)
    # Sharing the immutable PIL image avoids num_frames full CPU copies; DiffSynth
    # stacks the entries when preprocessing the input video.
    input_video = [source] * args.num_frames

    model_paths = wan22_model_paths(weights_path)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_paths[1]),
            ModelConfig(path=model_paths[0]),
            ModelConfig(model_paths[2]),
        ],
        tokenizer_config=ModelConfig(wan22_tokenizer_path(weights_path)),
    )

    print(
        "Running Wan2.2 repeated-source V2V: "
        f"size={args.width}x{args.height}, frames={args.num_frames}, "
        f"fps={args.fps:g}, denoising={args.denoising_strength:g}, "
        f"cfg={args.cfg:g}, steps={args.steps}, sigma={args.sigma:g}, seed={args.seed}"
    )
    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative,
        input_video=input_video,
        denoising_strength=args.denoising_strength,
        seed=args.seed,
        rand_device=pipe.device,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        cfg_scale=args.cfg,
        num_inference_steps=args.steps,
        sigma_shift=args.sigma,
        tiled=args.tiled,
    )

    if not video:
        raise RuntimeError("Wan pipeline returned no frames")
    if len(video) != args.num_frames:
        raise RuntimeError(f"Wan pipeline returned {len(video)} frames; expected {args.num_frames}")

    # Check once more because inference is long and another process may have
    # created the destination while it was running.
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(video, str(output_path), fps=args.fps, quality=5)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"video writer did not create a non-empty output: {output_path}")

    print(f"Saved {len(video)} frames to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
