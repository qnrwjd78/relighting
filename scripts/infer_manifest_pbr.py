#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import infer_manifest as base  # noqa: E402


DEFAULT_PROMPT = "photorealistic object relighting, preserve geometry and materials"

torch = None
Image = None
tqdm = None

encode_image_latents = None
extract_light_state = None
extract_lora_state = None
extract_type_state = None
load_pipe = None
load_state = None
infer_light_encoder_shape = None
infer_type_embedding_num_types = None
LightokenEncoder = None
parse_attrs_json = None
TokenLightPBRTypeEmbedding = None
model_fn_wan_video_tokenlight_pbr = None
tokenlight_pbr_type_count = None


def ensure_runtime_imports(*, include_model: bool) -> None:
    global torch, Image, tqdm
    global encode_image_latents, extract_light_state, extract_lora_state, extract_type_state, load_pipe, load_state
    global infer_light_encoder_shape, infer_type_embedding_num_types
    global LightokenEncoder, parse_attrs_json
    global TokenLightPBRTypeEmbedding, model_fn_wan_video_tokenlight_pbr, tokenlight_pbr_type_count

    if torch is None or Image is None or tqdm is None:
        import torch as _torch
        from PIL import Image as _Image
        from tqdm import tqdm as _tqdm

        torch = _torch
        Image = _Image
        tqdm = _tqdm

    if include_model and encode_image_latents is None:
        from model.infer_tokenlight import (  # noqa: WPS433
            encode_image_latents as _encode_image_latents,
            extract_light_state as _extract_light_state,
            extract_lora_state as _extract_lora_state,
            extract_type_state as _extract_type_state,
            infer_light_encoder_shape as _infer_light_encoder_shape,
            infer_type_embedding_num_types as _infer_type_embedding_num_types,
            load_pipe as _load_pipe,
            load_state as _load_state,
        )
        from model.lightoken_encoder import LightokenEncoder as _LightokenEncoder  # noqa: WPS433
        from model.lightoken_encoder import parse_attrs_json as _parse_attrs_json  # noqa: WPS433
        from model.tokenlight_wan_pbr import (  # noqa: WPS433
            TokenLightPBRTypeEmbedding as _TokenLightPBRTypeEmbedding,
            model_fn_wan_video_tokenlight_pbr as _model_fn_wan_video_tokenlight_pbr,
            tokenlight_pbr_type_count as _tokenlight_pbr_type_count,
        )

        encode_image_latents = _encode_image_latents
        extract_light_state = _extract_light_state
        extract_lora_state = _extract_lora_state
        extract_type_state = _extract_type_state
        infer_light_encoder_shape = _infer_light_encoder_shape
        infer_type_embedding_num_types = _infer_type_embedding_num_types
        load_pipe = _load_pipe
        load_state = _load_state
        LightokenEncoder = _LightokenEncoder
        parse_attrs_json = _parse_attrs_json
        TokenLightPBRTypeEmbedding = _TokenLightPBRTypeEmbedding
        model_fn_wan_video_tokenlight_pbr = _model_fn_wan_video_tokenlight_pbr
        tokenlight_pbr_type_count = _tokenlight_pbr_type_count


def add_bool_arg(parser: argparse.ArgumentParser, name: str, *, default: bool, help_text: str = "") -> None:
    option = name if name.startswith("--") else f"--{name}"
    dest = option[2:].replace("-", "_")
    parser.add_argument(option, dest=dest, action="store_true", help=help_text)
    parser.add_argument(f"--no-{option[2:]}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TokenLight PBR-stream inference from a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-path", default="", help="Base path for relative image paths inside the manifest.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default="", help="PBR checkpoint containing LoRA, light encoder, and type embedding.")
    parser.add_argument("--weights_dir", default="weights/Wan2.2-TI2V-5B")

    parser.add_argument("--source-key", default="input_image")
    parser.add_argument("--target-key", default="target_image")
    parser.add_argument("--target-fallback-key", default="video")
    parser.add_argument("--mask-key", default="inf_mask")
    parser.add_argument("--mask-fallback-key", default="mask")
    parser.add_argument("--attrs-key", default="attrs_json")
    parser.add_argument("--prompt-key", default="prompt")

    parser.add_argument("--pbr-streams", default="depth,normal")
    parser.add_argument("--pbr-stream-image-keys", default="depth:pbr_depth_image,normal:pbr_normal_image")
    parser.add_argument(
        "--pbr-mode",
        choices=("condition", "target"),
        default="condition",
        help=(
            "condition: feed clean PBR latents as conditioning tokens. "
            "target: feed timestep-noised GT PBR latents as target/denoising tokens."
        ),
    )

    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--gpu-devices",
        "--gpu_devices",
        default="",
        help="Comma-separated GPU ids for multi-process inference, e.g. 0,1,2,3. Use cpu for CPU.",
    )
    parser.add_argument("--token_dim", type=int, default=0)
    parser.add_argument("--fourier_features", type=int, default=512)
    parser.add_argument("--fourier_sigma", type=float, default=5.0)
    parser.add_argument("--tokenlight_max_lights", "--max-lights", type=int, default=2)
    add_bool_arg(
        parser,
        "--drop-source",
        default=False,
        help_text="Zero the source-image latent so inference matches the UniRelight source-drop branch.",
    )
    add_bool_arg(
        parser,
        "--drop-pbr",
        default=False,
        help_text="Zero all PBR stream latents while keeping their conditioning tokens.",
    )
    parser.add_argument(
        "--illum-head-checkpoint",
        default="",
        help="Optional illumination head checkpoint. If set, decode head(final_latents) next to each RGB output.",
    )
    parser.add_argument("--illum-output-suffix", default="_illum_head")
    add_bool_arg(
        parser,
        "--save-pbr-predictions",
        default=False,
        help_text="Denoise PBR streams from noise and save decoded predicted depth/normal images.",
    )
    parser.add_argument("--pbr-output-suffix-format", default="_pred_{stream}")

    add_bool_arg(parser, "--skip-existing", default=True)
    add_bool_arg(parser, "--with-gt", default=True)
    parser.add_argument("--limit", type=int, default=0)

    parser.add_argument("--eval", action="store_true", help="Compute PSNR/SSIM/LPIPS after inference.")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing predictions.")
    parser.add_argument("--metrics-output", default="")
    parser.add_argument("--metric-device", default="auto")
    parser.add_argument("--lpips-net", default="alex")
    add_bool_arg(parser, "--allow-missing", default=True)
    return parser.parse_args()


def parse_pbr_streams(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        streams = value
    else:
        streams = str(value).split(",")
    result = [stream.strip() for stream in streams if stream.strip()]
    if not result:
        raise ValueError("--pbr-streams must contain at least one stream")
    if len(set(result)) != len(result):
        raise ValueError(f"Duplicate PBR streams: {result}")
    return result


def parse_stream_key_map(value: str | Mapping[str, str] | None, streams: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if isinstance(value, Mapping):
        mapping.update({str(key): str(item) for key, item in value.items()})
    elif value:
        for item in str(value).split(","):
            if not item.strip():
                continue
            if ":" not in item:
                raise ValueError(f"Expected stream:key item, got {item!r}")
            name, key = item.split(":", 1)
            mapping[name.strip()] = key.strip()
    return {stream: mapping.get(stream, f"pbr_{stream}_image") for stream in streams}


def load_matching_state(module: torch.nn.Module, state: dict[str, torch.Tensor], name: str) -> int:
    if not state:
        print(f"[infer_manifest_pbr] {name}: no checkpoint tensors found", flush=True)
        return 0
    module_state = module.state_dict()
    matched = {
        key: value
        for key, value in state.items()
        if key in module_state and tuple(module_state[key].shape) == tuple(value.shape)
    }
    if matched:
        module.load_state_dict(matched, strict=False)
    print(
        f"[infer_manifest_pbr] {name}: checkpoint_tensors={len(state)} loaded={len(matched)}",
        flush=True,
    )
    if not matched:
        print(f"[infer_manifest_pbr] WARNING: loaded 0 tensors into {name}", flush=True)
    return len(matched)


def log_runtime_device(pipe) -> None:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    message = f"[infer_manifest_pbr] CUDA_VISIBLE_DEVICES={cuda_visible or '<unset>'} pipe.device={pipe.device}"
    if torch.cuda.is_available() and str(pipe.device).startswith("cuda"):
        index = torch.cuda.current_device()
        message += f" current_cuda={index}:{torch.cuda.get_device_name(index)}"
    print(message, flush=True)


def setup_pipeline(args: argparse.Namespace):
    ensure_runtime_imports(include_model=True)
    streams = parse_pbr_streams(args.pbr_streams)
    pipe = load_pipe(args)
    log_runtime_device(pipe)
    combined = load_state(args.checkpoint)
    lora = extract_lora_state(combined)
    print(
        f"[infer_manifest_pbr] checkpoint_tensors={0 if not combined else len(combined)} lora_tensors={len(lora)}",
        flush=True,
    )
    if lora:
        pipe.load_lora(pipe.dit, state_dict=lora, alpha=1.0)

    token_dim = args.token_dim if args.token_dim > 0 else int(pipe.dit.dim)
    light_state = extract_light_state(combined)
    max_lights, fourier_features = infer_light_encoder_shape(
        light_state,
        requested_max_lights=args.tokenlight_max_lights,
        requested_fourier_features=args.fourier_features,
    )
    light_encoder = LightokenEncoder(
        token_dim,
        fourier_features=fourier_features,
        fourier_sigma=args.fourier_sigma,
        max_lights=max_lights,
    ).to(device=pipe.device, dtype=pipe.torch_dtype)
    load_matching_state(light_encoder, light_state, "light_encoder")
    light_encoder.eval()

    type_state = extract_type_state(combined)
    num_types = infer_type_embedding_num_types(
        type_state,
        requested_num_types=tokenlight_pbr_type_count(len(streams)),
    )
    type_embedding = TokenLightPBRTypeEmbedding(
        token_dim,
        num_types=num_types,
    ).to(device=pipe.device, dtype=pipe.torch_dtype)
    load_matching_state(type_embedding, type_state, "tokenlight_type_embedding")
    type_embedding.eval()

    illum_head = None
    if args.illum_head_checkpoint:
        from model.illumination_latent_head import load_illumination_head_checkpoint  # noqa: WPS433

        illum_head, _, _ = load_illumination_head_checkpoint(args.illum_head_checkpoint, map_location="cpu")
        illum_head.to(device=pipe.device, dtype=pipe.torch_dtype)
        illum_head.eval()
        for param in illum_head.parameters():
            param.requires_grad_(False)
        print(f"[infer_manifest_pbr] loaded illumination head: {args.illum_head_checkpoint}", flush=True)
    return pipe, light_encoder, type_embedding, illum_head, streams


def pbr_image_path(row: dict[str, Any], args: argparse.Namespace, stream_name: str, key: str) -> Path:
    value = row.get(key)
    if not value:
        raise KeyError(
            f"Missing PBR stream `{stream_name}` key {key!r} in manifest row {row.get('_manifest_index')}"
        )
    return base.resolve_data(value, args)


def illum_prediction_path(pred: Path, args: argparse.Namespace) -> Path:
    return pred.with_name(f"{pred.stem}{args.illum_output_suffix}{pred.suffix}")


def pbr_prediction_path(pred: Path, args: argparse.Namespace, stream_name: str) -> Path:
    suffix = str(args.pbr_output_suffix_format).format(stream=stream_name)
    return pred.with_name(f"{pred.stem}{suffix}{pred.suffix}")


def runtime_no_grad(fn):
    def wrapper(*args, **kwargs):
        ensure_runtime_imports(include_model=False)
        with torch.no_grad():
            return fn(*args, **kwargs)

    return wrapper


def split_model_outputs(output):
    if isinstance(output, tuple):
        return output[0], output[1]
    return output, None


def combine_pbr_outputs(pos, neg, cfg_scale: float):
    if pos is None or neg is None:
        return pos
    if isinstance(pos, Mapping):
        return {
            stream_name: neg[stream_name] + cfg_scale * (value - neg[stream_name])
            for stream_name, value in pos.items()
        }
    return neg + cfg_scale * (pos - neg)


def pbr_output_for_stream(outputs, stream_name: str):
    if outputs is None:
        return None
    if isinstance(outputs, Mapping):
        return outputs.get(stream_name)
    return outputs


def make_latent_noise_like(latents_by_stream: Mapping[str, torch.Tensor], seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=next(iter(latents_by_stream.values())).device)
    generator.manual_seed(int(seed))
    return {
        stream_name: torch.randn(
            latents.shape,
            device=latents.device,
            dtype=latents.dtype,
            generator=generator,
        )
        for stream_name, latents in latents_by_stream.items()
    }


@runtime_no_grad
def generate_pbr(
    pipe,
    light_encoder: LightokenEncoder,
    type_embedding: TokenLightPBRTypeEmbedding,
    illum_head,
    attrs: dict[str, Any],
    source: Image.Image,
    pbr_images: dict[str, Image.Image],
    args: argparse.Namespace,
):
    pipe.model_fn = lambda **kwargs: model_fn_wan_video_tokenlight_pbr(
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
    if args.drop_source:
        source_latents = torch.zeros_like(source_latents)
    inputs_shared["tokenlight_source_latents"] = source_latents
    pbr_clean_latents = {
        stream_name: encode_image_latents(pipe, image, args)
        for stream_name, image in pbr_images.items()
    }
    if args.drop_pbr:
        pbr_clean_latents = {
            stream_name: torch.zeros_like(latents)
            for stream_name, latents in pbr_clean_latents.items()
        }
    predict_pbr = bool(args.save_pbr_predictions)
    pbr_is_target = args.pbr_mode == "target" or predict_pbr
    pbr_noise = None
    pbr_current_latents = None
    if predict_pbr:
        pbr_current_latents = make_latent_noise_like(pbr_clean_latents, int(args.seed) + 1_000_003)
    elif pbr_is_target:
        pbr_noise = make_latent_noise_like(pbr_clean_latents, int(args.seed) + 1_000_003)
    else:
        inputs_shared["tokenlight_pbr_stream_latents"] = pbr_clean_latents
    inputs_shared["tokenlight_pbr_stream_is_target"] = {
        stream_name: pbr_is_target for stream_name in pbr_images
    }
    inputs_posi["tokenlight_drop_light"] = False
    inputs_nega["tokenlight_drop_light"] = True

    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for index, timestep in enumerate(tqdm(pipe.scheduler.timesteps)):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        if predict_pbr:
            inputs_shared["tokenlight_pbr_stream_latents"] = pbr_current_latents
        elif pbr_is_target:
            inputs_shared["tokenlight_pbr_stream_latents"] = {
                stream_name: pipe.scheduler.add_noise(
                    pbr_clean_latents[stream_name],
                    pbr_noise[stream_name],
                    timestep,
                )
                for stream_name in pbr_clean_latents
            }
        noise_pos = pipe.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
        noise_pos, pbr_noise_pos = split_model_outputs(noise_pos)
        if args.cfg_scale != 1.0:
            noise_neg = pipe.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
            noise_neg, pbr_noise_neg = split_model_outputs(noise_neg)
            noise = noise_neg + args.cfg_scale * (noise_pos - noise_neg)
            pbr_noise_pred = combine_pbr_outputs(pbr_noise_pos, pbr_noise_neg, args.cfg_scale)
        else:
            noise = noise_pos
            pbr_noise_pred = pbr_noise_pos
        inputs_shared["latents"] = pipe.scheduler.step(noise, pipe.scheduler.timesteps[index], inputs_shared["latents"])
        if predict_pbr:
            updated_pbr_latents = {}
            for stream_name, stream_latents in pbr_current_latents.items():
                stream_noise = pbr_output_for_stream(pbr_noise_pred, stream_name)
                if stream_noise is None:
                    raise ValueError(f"Model did not return PBR prediction for stream `{stream_name}`")
                updated_pbr_latents[stream_name] = pipe.scheduler.step(
                    stream_noise,
                    pipe.scheduler.timesteps[index],
                    stream_latents,
                )
            pbr_current_latents = updated_pbr_latents

    final_latents = inputs_shared["latents"]
    pipe.load_models_to_device(["vae"])
    rgb_decoded = pipe.vae.decode(
        final_latents,
        device=pipe.device,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    )
    rgb_video = pipe.vae_output_to_video(rgb_decoded)
    illum_video = None
    if illum_head is not None:
        try:
            head_param = next(illum_head.parameters())
            head_device = head_param.device
            head_dtype = head_param.dtype
        except StopIteration:
            head_device = pipe.device
            head_dtype = pipe.torch_dtype
        illum_latents = illum_head(final_latents.to(device=head_device, dtype=head_dtype))
        illum_decoded = pipe.vae.decode(
            illum_latents.to(dtype=pipe.torch_dtype, device=pipe.device),
            device=pipe.device,
            tiled=True,
            tile_size=(30, 52),
            tile_stride=(15, 26),
        )
        illum_video = pipe.vae_output_to_video(illum_decoded)
    pbr_videos = {}
    if predict_pbr:
        for stream_name, stream_latents in pbr_current_latents.items():
            pbr_decoded = pipe.vae.decode(
                stream_latents,
                device=pipe.device,
                tiled=True,
                tile_size=(30, 52),
                tile_stride=(15, 26),
            )
            pbr_videos[stream_name] = pipe.vae_output_to_video(pbr_decoded)
    return rgb_video, illum_video, pbr_videos


def run_inference(rows: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> int:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required unless --eval-only is set")
    pipe, light_encoder, type_embedding, illum_head, streams = setup_pipeline(args)
    stream_keys = parse_stream_key_map(args.pbr_stream_image_keys, streams)
    completed = 0
    desc = Path(args.checkpoint).parent.name + "/" + Path(args.checkpoint).stem
    for row in tqdm(rows, desc=desc):
        pred = base.prediction_path(output_dir, row)
        illum_pred = illum_prediction_path(pred, args) if illum_head is not None else None
        pbr_preds = {
            stream_name: pbr_prediction_path(pred, args, stream_name)
            for stream_name in streams
        } if args.save_pbr_predictions else {}
        target = base.target_path(row, args)
        pbr_outputs_exist = all(path.exists() for path in pbr_preds.values())
        if args.skip_existing and pred.exists() and (illum_pred is None or illum_pred.exists()) and pbr_outputs_exist:
            if args.with_gt and target and target.exists() and not base.with_gt_path(pred).exists():
                base.ensure_runtime_imports(include_model=False)
                base.save_with_gt(pred, base.source_path(row, args), target)
            completed += 1
            continue

        source = Image.open(base.source_path(row, args)).convert("RGB")
        pbr_images = {
            stream_name: Image.open(pbr_image_path(row, args, stream_name, key)).convert("RGB")
            for stream_name, key in stream_keys.items()
        }
        infer_args = argparse.Namespace(**vars(args))
        infer_args.prompt = row.get(args.prompt_key) or args.prompt
        rgb_video, illum_video, pbr_videos = generate_pbr(
            pipe,
            light_encoder,
            type_embedding,
            illum_head,
            parse_attrs_json(row.get(args.attrs_key)),
            source,
            pbr_images,
            infer_args,
        )
        pred.parent.mkdir(parents=True, exist_ok=True)
        rgb_video[0].save(pred)
        if illum_pred is not None and illum_video is not None:
            illum_pred.parent.mkdir(parents=True, exist_ok=True)
            illum_video[0].save(illum_pred)
        for stream_name, stream_video in pbr_videos.items():
            pbr_pred = pbr_preds[stream_name]
            pbr_pred.parent.mkdir(parents=True, exist_ok=True)
            stream_video[0].save(pbr_pred)
        if args.with_gt and target and target.exists():
            base.ensure_runtime_imports(include_model=False)
            base.save_with_gt(pred, source, target)
        completed += 1
    return completed


def run_inference_worker(device_id: str, rows: list[dict[str, Any]], output_dir: str, args_dict: dict[str, Any]) -> int:
    args = base.apply_worker_device(device_id, argparse.Namespace(**args_dict))
    print(f"[infer_manifest_pbr:{device_id}] rows={len(rows)} device={args.device}", flush=True)
    completed = run_inference(rows, Path(output_dir), args)
    print(f"[infer_manifest_pbr:{device_id}] completed={completed}", flush=True)
    return completed


def run_inference_distributed(rows: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> int:
    devices = base.parse_gpu_devices(args.gpu_devices)
    if not devices:
        return run_inference(rows, output_dir, args)
    assigned = base.split_rows_by_device(rows, devices)
    if len(assigned) == 1:
        device_id, shard = assigned[0]
        worker_args = base.apply_worker_device(device_id, args)
        print(f"[infer_manifest_pbr] gpu_devices={','.join(devices)} rows={len(rows)}", flush=True)
        return run_inference(shard, output_dir, worker_args)

    print(
        f"[infer_manifest_pbr] gpu_devices={','.join(devices)} processes={len(assigned)} rows={len(rows)}",
        flush=True,
    )
    args_dict = vars(args)
    completed = 0
    context = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(assigned), mp_context=context) as executor:
        futures = [
            executor.submit(run_inference_worker, device_id, shard, output_dir.as_posix(), args_dict)
            for device_id, shard in assigned
        ]
        for future in concurrent.futures.as_completed(futures):
            completed += int(future.result())
    return completed


def write_snapshot(rows: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> None:
    streams = parse_pbr_streams(args.pbr_streams)
    snapshot = {
        "manifest": base.resolve_repo(args.manifest).as_posix(),
        "base_path": base.data_base(args).as_posix(),
        "output_dir": output_dir.as_posix(),
        "checkpoint": "" if not args.checkpoint else base.resolve_repo(args.checkpoint).as_posix(),
        "source_key": args.source_key,
        "target_key": args.target_key,
        "target_fallback_key": args.target_fallback_key,
        "mask_key": args.mask_key,
        "mask_fallback_key": args.mask_fallback_key,
        "attrs_key": args.attrs_key,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "tokenlight_max_lights": args.tokenlight_max_lights,
        "drop_source": bool(args.drop_source),
        "drop_pbr": bool(args.drop_pbr),
        "illum_head_checkpoint": (
            "" if not args.illum_head_checkpoint else base.resolve_repo(args.illum_head_checkpoint).as_posix()
        ),
        "illum_output_suffix": args.illum_output_suffix,
        "save_pbr_predictions": bool(args.save_pbr_predictions),
        "pbr_output_suffix_format": args.pbr_output_suffix_format,
        "pbr_streams": streams,
        "pbr_stream_image_keys": parse_stream_key_map(args.pbr_stream_image_keys, streams),
        "pbr_mode": args.pbr_mode,
        "gpu_devices": base.parse_gpu_devices(args.gpu_devices),
        "eval": args.eval,
        "eval_only": args.eval_only,
        "row_count": len(rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "infer_manifest_pbr_config_resolved.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    args.manifest = base.resolve_repo(args.manifest).as_posix()
    args.weights_dir = base.resolve_repo(args.weights_dir).as_posix()
    if args.checkpoint:
        args.checkpoint = base.resolve_repo(args.checkpoint).as_posix()
    if args.illum_head_checkpoint:
        args.illum_head_checkpoint = base.resolve_repo(args.illum_head_checkpoint).as_posix()

    rows = base.load_rows(Path(args.manifest), int(args.limit))
    output_dir = base.resolve_repo(args.output_dir)
    write_snapshot(rows, output_dir, args)

    completed = 0
    if not args.eval_only:
        completed = run_inference_distributed(rows, output_dir, args)
        print(f"[infer_manifest_pbr] completed={completed} output_dir={output_dir}", flush=True)

    if args.eval or args.eval_only:
        metrics = base.run_eval(rows, output_dir, args)
        metrics_output = base.resolve_repo(args.metrics_output) if args.metrics_output else output_dir / "metrics.json"
        metrics_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "manifest": args.manifest,
            "base_path": base.data_base(args).as_posix(),
            "pred_dir": output_dir.as_posix(),
            "metrics_output": metrics_output.as_posix(),
            "device": str(base.metric_device(args.metric_device)),
            "lpips_net": args.lpips_net,
            **metrics,
        }
        metrics_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[infer_manifest_pbr] wrote metrics: {metrics_output}", flush=True)
        print(json.dumps(payload["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
