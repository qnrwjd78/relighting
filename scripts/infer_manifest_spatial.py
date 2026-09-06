#!/usr/bin/env python3
"""Manifest inference for GT-mask or LGI spatial-conditioned TokenLight."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.infer_tokenlight import encode_image_latents  # noqa: E402
from model.tokenlight_wan_spatial import (  # noqa: E402
    SpatialConditionEncoder,
    SpatialPrefixTypeEmbedding,
    model_fn_wan_video_tokenlight_spatial,
)
from model.train_tokenlight_spatial_safe import SpatialConditionReader  # noqa: E402
from scripts import infer_manifest as base  # noqa: E402


def extract_module(state, prefix):
    result = {}
    for key, value in (state or {}).items():
        key = str(key).removeprefix("module.")
        if key.startswith(prefix + "."):
            result[key[len(prefix) + 1 :]] = value
    return result


def parse_args():
    parser = base.parse_args.__wrapped__() if hasattr(base.parse_args, "__wrapped__") else None
    if parser is not None:
        return parser.parse_args()
    # Mirror the stable base CLI and add spatial-only options.
    import argparse as _argparse
    p = _argparse.ArgumentParser(description=__doc__)
    for name, kwargs in (
        ("--manifest", {"required": True}), ("--base-path", {"default": ""}),
        ("--output-dir", {"required": True}), ("--checkpoint", {"required": True}),
        ("--weights_dir", {"default": "weights/Wan2.2-TI2V-5B"}),
        ("--spatial-kind", {"choices": ("gt_masks", "lgi"), "required": True}),
        ("--lgi-root", {"default": "data/objaverse_32_lgimap"}),
        ("--source-key", {"default": "input_image"}), ("--target-key", {"default": "target_image"}),
        ("--target-fallback-key", {"default": "video"}), ("--mask-key", {"default": "mask"}),
        ("--mask-fallback-key", {"default": ""}), ("--attrs-key", {"default": "attrs_json"}),
        ("--prompt-key", {"default": "prompt"}), ("--prompt", {"default": base.DEFAULT_PROMPT}),
        ("--height", {"type": int, "default": 480}), ("--width", {"type": int, "default": 480}),
        ("--num_frames", {"type": int, "default": 1}),
        ("--num_inference_steps", {"type": int, "default": 50}),
        ("--cfg_scale", {"type": float, "default": 1.0}), ("--seed", {"type": int, "default": 0}),
        ("--device", {"default": "cuda"}), ("--gpu-devices", {"default": ""}),
        ("--token_dim", {"type": int, "default": 0}),
        ("--fourier_features", {"type": int, "default": 512}),
        ("--fourier_sigma", {"type": float, "default": 5.0}),
        ("--tokenlight_max_lights", {"type": int, "default": 2}),
        ("--limit", {"type": int, "default": 0}), ("--metrics-output", {"default": ""}),
        ("--metric-device", {"default": "auto"}), ("--lpips-net", {"default": "alex"}),
    ):
        p.add_argument(name, **kwargs)
    p.add_argument("--eval", action="store_true"); p.add_argument("--eval-only", action="store_true")
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--with-gt", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow-missing", action=argparse.BooleanOptionalAction, default=True)
    p.set_defaults(use_mask_input=False, tokenlight_mask_tokens=False, fps=15)
    return p.parse_args()


def reader_args(args):
    return argparse.Namespace(
        spatial_condition_kind=args.spatial_kind, height=args.height, width=args.width,
        dataset_base_path=args.base_path, tokenlight_spatial_allow_missing=False,
        tokenlight_gt_object_mask_key="mask", tokenlight_gt_direct_mask_key="inf_mask",
        tokenlight_gt_shadow_mask_key="shadow_mask", tokenlight_gt_mask_threshold=0.5,
        tokenlight_lgi_root=args.lgi_root, tokenlight_lgi_dir_key="lgi_dir",
        tokenlight_lgi_position_prefix="position_", tokenlight_lgi_position_digits=2,
        tokenlight_lgi_position_offset=0, tokenlight_lgi_zero_triplet_invalid=False,
    )


def setup(args):
    pipe, light, type_embedding = base.setup_pipeline(args)
    state = base.load_state(args.checkpoint)
    channels = 3 if args.spatial_kind == "lgi" else 2
    spatial = SpatialConditionEncoder(
        input_channels=channels,
        token_dim=int(pipe.dit.dim),
        hidden_channels=64,
    ).to(device=pipe.device, dtype=pipe.torch_dtype)
    spatial_type = SpatialPrefixTypeEmbedding(int(pipe.dit.dim)).to(pipe.device, pipe.torch_dtype)
    spatial.load_state_dict(extract_module(state, "tokenlight_spatial_encoder"), strict=True)
    spatial_type.load_state_dict(extract_module(state, "tokenlight_spatial_type_embedding"), strict=True)
    spatial.eval(); spatial_type.eval()
    return pipe, light, type_embedding, spatial, spatial_type, SpatialConditionReader(reader_args(args))


@torch.no_grad()
def generate(pipe, light, type_embedding, spatial, spatial_type, reader, row, source, args):
    pipe.model_fn = lambda **kw: model_fn_wan_video_tokenlight_spatial(
        tokenlight_light_encoder=light, tokenlight_type_embedding=type_embedding,
        tokenlight_spatial_encoder=spatial, tokenlight_spatial_type_embedding=spatial_type, **kw)
    pipe.scheduler.set_timesteps(args.num_inference_steps, denoising_strength=1.0, shift=5.0)
    pos, neg = {"prompt": args.prompt}, {"prompt": args.prompt}
    shared = {
        "input_image": None, "end_image": None, "input_video": None, "denoising_strength": 1.0,
        "control_video": None, "reference_image": None, "seed": args.seed, "rand_device": pipe.device,
        "height": args.height, "width": args.width, "num_frames": 1, "cfg_scale": 1,
        "cfg_merge": False, "sigma_shift": 5.0, "tiled": True, "tile_size": (30, 52),
        "tile_stride": (15, 26), "framewise_decoding": False,
    }
    for unit in pipe.units:
        shared, pos, neg = pipe.unit_runner(unit, pipe, shared, pos, neg)
    condition, present, _ = reader.load(row)
    source_latents = encode_image_latents(pipe, source, args)
    shared.update(tokenlight_attrs=[base.attrs_from_row(row, args.attrs_key)],
                  tokenlight_source_latents=source_latents,
                  tokenlight_spatial_map=condition.unsqueeze(0).to(pipe.device),
                  tokenlight_spatial_present=torch.tensor([present], device=pipe.device))
    is_lgi_bridge = reader.kind == "lgi"
    if is_lgi_bridge:
        # LGI is trained as a source-at-t=0 -> target-at-t=1 latent bridge.
        # The normal Wan unit above initializes latents with Gaussian noise for
        # noise-to-data FlowMatch, which is the wrong endpoint for this model.
        shared["latents"] = source_latents.clone()
    pos.update(tokenlight_drop_light=False, tokenlight_drop_spatial=False)
    neg.update(tokenlight_drop_light=True, tokenlight_drop_spatial=True)
    pipe.load_models_to_device(pipe.in_iteration_models); models = {n: getattr(pipe, n) for n in pipe.in_iteration_models}
    for index, timestep in enumerate(pipe.scheduler.timesteps):
        t = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        pred = pipe.model_fn(**models, **shared, **pos, timestep=t)
        if args.cfg_scale != 1.0:
            uncond = pipe.model_fn(**models, **shared, **neg, timestep=t)
            pred = uncond + args.cfg_scale * (pred - uncond)
        # Wan's scheduler integrates from high sigma to low sigma and therefore
        # multiplies the model output by a negative delta.  LGI training uses
        # t=1-sigma and predicts the forward source-to-target drift, so negate
        # that drift at the scheduler boundary.  GT masks retain ordinary
        # noise-to-data FlowMatch behavior.
        step_pred = -pred if is_lgi_bridge else pred
        shared["latents"] = pipe.scheduler.step(step_pred, timestep, shared["latents"])
    pipe.load_models_to_device(["vae"])
    decoded = pipe.vae.decode(shared["latents"], device=pipe.device, tiled=True, tile_size=(30, 52), tile_stride=(15, 26))
    return pipe.vae_output_to_video(decoded)


def run(rows, output_dir, args):
    pipe, light, type_embedding, spatial, spatial_type, reader = setup(args); done = 0
    for row in rows:
        pred = base.prediction_path(output_dir, row); target = base.target_path(row, args)
        if args.skip_existing and pred.exists(): done += 1; continue
        source = base.Image.open(base.source_path(row, args)).convert("RGB")
        args.prompt = row.get(args.prompt_key) or args.prompt
        video = generate(pipe, light, type_embedding, spatial, spatial_type, reader, row, source, args)
        pred.parent.mkdir(parents=True, exist_ok=True); video[0].save(pred)
        if args.with_gt and target and target.exists(): base.save_with_gt(pred, source, target)
        done += 1
    return done


def main():
    args = parse_args(); args.manifest = base.resolve_repo(args.manifest).as_posix(); args.checkpoint = base.resolve_repo(args.checkpoint).as_posix()
    rows = base.load_rows(Path(args.manifest), args.limit); out = base.resolve_repo(args.output_dir)
    base.ensure_runtime_imports(include_model=False); run(rows, out, args)
    if args.eval:
        metrics = base.run_eval(rows, out, args); target = Path(args.metrics_output) if args.metrics_output else out / "metrics.json"
        target.parent.mkdir(parents=True, exist_ok=True); import json; target.write_text(json.dumps(metrics, indent=2) + "\n")
    return 0


if __name__ == "__main__": raise SystemExit(main())
