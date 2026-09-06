from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.light_encoder import (
    GLOBAL_LIGHTOKEN_NAMES,
    LIGHTOKEN_NAMES,
    PER_LIGHTOKEN_NAMES,
    LightokenEncoder,
    parse_attrs_json,
)
from model.pretrain_weight import validate_wan22_weights, wan22_model_paths, wan22_tokenizer_path
from model.wan import TokenLightTypeEmbedding, model_fn_wan_video_tokenlight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TokenLight Wan2.2-TI2V-5B inference.")
    parser.add_argument("--weights_dir", default="weights/Wan2.2-TI2V-5B")
    parser.add_argument("--source", required=True)
    parser.add_argument("--attrs", required=True, help="Inline JSON or path to JSON attrs.")
    parser.add_argument("--mask", default="")
    parser.add_argument("--checkpoint", default="", help="Checkpoint containing LoRA and light_encoder weights.")
    parser.add_argument("--lora_checkpoint", default="")
    parser.add_argument("--light_checkpoint", default="")
    parser.add_argument("--prompt", default="photorealistic object relighting, preserve geometry and materials")
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--token_dim", type=int, default=0)
    parser.add_argument("--fourier_features", type=int, default=512)
    parser.add_argument("--fourier_sigma", type=float, default=5.0)
    parser.add_argument("--tokenlight_max_lights", "--max_lights", type=int, default=1)
    parser.add_argument("--tokenlight_mask_tokens", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_state(path: str) -> dict[str, torch.Tensor] | None:
    if not path:
        return None
    try:
        from safetensors.torch import load_file

        state = load_file(path)
    except Exception:
        state = torch.load(path, map_location="cpu", weights_only=False)
    return state.get("state_dict", state)


def strip_prefix(text: str, prefix: str) -> str:
    return text[len(prefix) :] if text.startswith(prefix) else text


def extract_lora_state(state: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor]:
    if not state:
        return {}
    result = {}
    for key, value in state.items():
        key = strip_prefix(strip_prefix(key, "module."), "pipe.dit.")
        if ".lora_A." in key or ".lora_B." in key or key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight"):
            result[key] = value
    return result


def extract_light_state(state: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor]:
    if not state:
        return {}
    result = {}
    direct_prefixes = ("fourier.", "projections.")
    for key, value in state.items():
        key = strip_prefix(key, "module.")
        if key.startswith("light_encoder."):
            result[key.removeprefix("light_encoder.")] = value
        elif key.startswith(direct_prefixes):
            result[key] = value
    return result


def extract_type_state(state: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor]:
    if not state:
        return {}
    result = {}
    for key, value in state.items():
        key = strip_prefix(key, "module.")
        if key.startswith("tokenlight_type_embedding."):
            result[key.removeprefix("tokenlight_type_embedding.")] = value
        elif key.startswith("embedding."):
            result[key] = value
    return result


def light_token_count_for_max_lights(max_lights: int) -> int:
    max_lights = max(1, int(max_lights))
    if max_lights == 1:
        return len(LIGHTOKEN_NAMES)
    return len(GLOBAL_LIGHTOKEN_NAMES) + max_lights * len(PER_LIGHTOKEN_NAMES)


def infer_light_encoder_shape(
    light_state: dict[str, torch.Tensor],
    *,
    requested_max_lights: int,
    requested_fourier_features: int,
) -> tuple[int, int]:
    max_lights = max(1, int(requested_max_lights))
    fourier_features = int(requested_fourier_features)
    weight = light_state.get("fourier.weight")
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        return max_lights, fourier_features

    attr_count = int(weight.shape[0])
    checkpoint_fourier_features = int(weight.shape[1])
    if checkpoint_fourier_features > 0 and checkpoint_fourier_features != fourier_features:
        print(
            "Using fourier_features="
            f"{checkpoint_fourier_features} from light checkpoint "
            f"(requested {fourier_features})."
        )
        fourier_features = checkpoint_fourier_features

    expected_attr_count = light_token_count_for_max_lights(max_lights)
    if attr_count == expected_attr_count:
        return max_lights, fourier_features

    inferred_max_lights = None
    if attr_count == len(LIGHTOKEN_NAMES):
        inferred_max_lights = 1
    else:
        global_count = len(GLOBAL_LIGHTOKEN_NAMES)
        per_light_count = len(PER_LIGHTOKEN_NAMES)
        if attr_count > global_count and (attr_count - global_count) % per_light_count == 0:
            inferred_max_lights = (attr_count - global_count) // per_light_count

    if inferred_max_lights is not None and inferred_max_lights > 0:
        print(
            "Using tokenlight_max_lights="
            f"{inferred_max_lights} from light checkpoint "
            f"(fourier.weight has {attr_count} attrs; requested {max_lights})."
        )
        max_lights = int(inferred_max_lights)
    return max_lights, fourier_features


def infer_type_embedding_num_types(
    type_state: dict[str, torch.Tensor],
    *,
    requested_num_types: int = 4,
) -> int:
    num_types = int(requested_num_types)
    weight = type_state.get("embedding.weight")
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        return num_types
    checkpoint_num_types = int(weight.shape[0])
    if checkpoint_num_types > 0 and checkpoint_num_types != num_types:
        print(
            "Using tokenlight type num_types="
            f"{checkpoint_num_types} from checkpoint "
            f"(requested {num_types})."
        )
        num_types = checkpoint_num_types
    return num_types


def load_attrs(value: str) -> dict[str, float]:
    path = Path(value)
    return parse_attrs_json(path.read_text(encoding="utf-8") if path.exists() else value)


def load_pipe(args: argparse.Namespace) -> WanVideoPipeline:
    validate_wan22_weights(args.weights_dir)
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    model_paths = wan22_model_paths(args.weights_dir)
    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_paths[1]),
            ModelConfig(path=model_paths[0]),
            ModelConfig(model_paths[2]),
        ],
        tokenizer_config=ModelConfig(wan22_tokenizer_path(args.weights_dir)),
    )


def encode_image_latents(pipe: WanVideoPipeline, image: Image.Image, args: argparse.Namespace) -> torch.Tensor:
    image = image.resize((args.width, args.height))
    pipe.load_models_to_device(["vae"])
    pixel_values = pipe.preprocess_image(image).transpose(0, 1)
    latents = pipe.vae.encode(
        [pixel_values.to(dtype=pipe.torch_dtype, device=pipe.device)],
        device=pipe.device,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    )
    return latents.to(dtype=pipe.torch_dtype, device=pipe.device)


@torch.no_grad()
def generate(
    pipe: WanVideoPipeline,
    light_encoder: LightokenEncoder,
    type_embedding: TokenLightTypeEmbedding | None,
    attrs: dict[str, float],
    source: Image.Image,
    mask: Image.Image | None,
    args: argparse.Namespace,
    extra_masks: list[Image.Image] | None = None,
):
    pipe.model_fn = lambda **kwargs: model_fn_wan_video_tokenlight(
        tokenlight_light_encoder=light_encoder,
        tokenlight_type_embedding=type_embedding,
        **kwargs,
    )
    pipe.scheduler.set_timesteps(args.num_inference_steps, denoising_strength=1.0, shift=5.0)
    inputs_posi = {"prompt": args.prompt}
    inputs_nega = {"prompt": args.prompt}
    inputs_shared = {
        "input_image": None,
        "end_image": None,
        "input_video": None,
        "denoising_strength": 1.0,
        "control_video": None,
        "reference_image": None,
        "camera_control_direction": None,
        "camera_control_speed": 1 / 54,
        "camera_control_origin": (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        "vace_video": None,
        "vace_video_mask": None,
        "vace_reference_image": None,
        "vace_scale": 1,
        "seed": args.seed,
        "rand_device": pipe.device,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "cfg_scale": 1,
        "cfg_merge": False,
        "sigma_shift": 5.0,
        "motion_bucket_id": None,
        "longcat_video": None,
        "tiled": True,
        "tile_size": (30, 52),
        "tile_stride": (15, 26),
        "sliding_window_size": None,
        "sliding_window_stride": None,
        "input_audio": None,
        "audio_sample_rate": 16000,
        "s2v_pose_video": None,
        "audio_embeds": None,
        "s2v_pose_latents": None,
        "motion_video": None,
        "animate_pose_video": None,
        "animate_face_video": None,
        "animate_inpaint_video": None,
        "animate_mask_video": None,
        "vap_video": None,
        "vap_prompt": " ",
        "negative_vap_prompt": " ",
        "wantodance_music_path": None,
        "wantodance_reference_image": None,
        "wantodance_fps": 30,
        "wantodance_keyframes": None,
        "wantodance_keyframes_mask": None,
        "framewise_decoding": False,
    }
    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)

    inputs_shared["tokenlight_attrs"] = [attrs]
    source_latents = encode_image_latents(pipe, source, args)
    inputs_shared["tokenlight_source_latents"] = source_latents
    if mask is not None:
        inputs_shared["tokenlight_mask_latents"] = encode_image_latents(pipe, mask, args)
    elif getattr(args, "tokenlight_mask_tokens", True):
        inputs_shared["tokenlight_mask_latents"] = torch.zeros_like(source_latents)
    if extra_masks:
        inputs_shared["tokenlight_extra_mask_latents"] = [
            encode_image_latents(pipe, extra_mask, args) for extra_mask in extra_masks
        ]
    inputs_posi["tokenlight_drop_light"] = False
    inputs_nega["tokenlight_drop_light"] = True

    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for index, timestep in enumerate(tqdm(pipe.scheduler.timesteps)):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pos = pipe.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
        if args.cfg_scale != 1.0:
            noise_neg = pipe.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
            noise = noise_neg + args.cfg_scale * (noise_pos - noise_neg)
        else:
            noise = noise_pos
        inputs_shared["latents"] = pipe.scheduler.step(noise, pipe.scheduler.timesteps[index], inputs_shared["latents"])

    pipe.load_models_to_device(["vae"])
    video = pipe.vae.decode(inputs_shared["latents"], device=pipe.device, tiled=True, tile_size=(30, 52), tile_stride=(15, 26))
    return pipe.vae_output_to_video(video)


def main() -> int:
    args = parse_args()
    pipe = load_pipe(args)
    combined = load_state(args.checkpoint)
    lora_state = load_state(args.lora_checkpoint) if args.lora_checkpoint else combined
    lora = extract_lora_state(lora_state)
    if lora:
        pipe.load_lora(pipe.dit, state_dict=lora, alpha=1.0)

    token_dim = args.token_dim if args.token_dim > 0 else int(pipe.dit.dim)
    light_checkpoint_state = load_state(args.light_checkpoint) if args.light_checkpoint else combined
    light_state = extract_light_state(light_checkpoint_state)
    max_lights, fourier_features = infer_light_encoder_shape(
        light_state,
        requested_max_lights=getattr(args, "tokenlight_max_lights", 1),
        requested_fourier_features=args.fourier_features,
    )
    light_encoder = LightokenEncoder(
        token_dim,
        fourier_features=fourier_features,
        fourier_sigma=args.fourier_sigma,
        max_lights=max_lights,
    ).to(device=pipe.device, dtype=pipe.torch_dtype)
    if light_state:
        light_encoder.load_state_dict(light_state, strict=False)
    light_encoder.eval()

    type_embedding = None
    type_state = extract_type_state(light_checkpoint_state)
    if type_state:
        num_types = infer_type_embedding_num_types(type_state, requested_num_types=4)
        type_embedding = TokenLightTypeEmbedding(token_dim, num_types=num_types).to(
            device=pipe.device,
            dtype=pipe.torch_dtype,
        )
        type_embedding.load_state_dict(type_state, strict=False)
        type_embedding.eval()

    source = Image.open(args.source).convert("RGB")
    mask = Image.open(args.mask).convert("RGB") if args.mask else None
    video = generate(pipe, light_encoder, type_embedding, load_attrs(args.attrs), source, mask, args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        video[0].save(output)
    else:
        save_video(video, str(output), fps=args.fps, quality=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
