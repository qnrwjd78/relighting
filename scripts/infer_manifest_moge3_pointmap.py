#!/usr/bin/env python3
"""Manifest inference and evaluation for the MoGe-3 point/direction-stream model."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.infer_manifest as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MoGe-3 point/direction-stream inference and optional PSNR/SSIM/LPIPS evaluation."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-path", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--weights_dir", default="weights/Wan2.2-TI2V-5B")
    parser.add_argument("--moge-cache-root", default="data/moge3_pointmap_fixed32_480")

    parser.add_argument("--source-key", default="input_image")
    parser.add_argument("--target-key", default="target_image")
    parser.add_argument("--target-fallback-key", default="video")
    parser.add_argument("--mask-key", default="inf_mask")
    parser.add_argument("--mask-fallback-key", default="mask")
    parser.add_argument("--extra-mask-keys", default="")
    parser.add_argument("--eval-mask-key", default="mask")
    parser.add_argument("--attrs-key", default="attrs_json")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--seed-key", default="")
    parser.add_argument("--prediction-name-key", default="")

    parser.add_argument("--prompt", default=common.DEFAULT_PROMPT)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-devices", "--gpu_devices", default="")
    parser.add_argument("--token_dim", type=int, default=0)
    parser.add_argument("--fourier_features", type=int, default=512)
    parser.add_argument("--fourier_sigma", type=float, default=5.0)
    parser.add_argument("--tokenlight_max_lights", "--max-lights", type=int, default=1)
    common.add_bool_arg(parser, "--tokenlight_mask_tokens", default=False)
    common.add_bool_arg(parser, "--use-mask-input", default=False)

    parser.add_argument("--moge-hidden-channels", type=int, default=64)
    parser.add_argument("--moge-camera-distance", type=float, default=3.5)
    parser.add_argument("--moge-camera-metric-scale", type=float, default=0.75)
    parser.add_argument("--moge-light-metric-scale", type=float, default=14.0 / 15.0)

    common.add_bool_arg(parser, "--skip-existing", default=True)
    common.add_bool_arg(parser, "--with-gt", default=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--metrics-output", default="")
    parser.add_argument("--metric-device", default="auto")
    parser.add_argument("--lpips-net", default="alex")
    common.add_bool_arg(parser, "--allow-missing", default=True)
    return parser.parse_args()


def extract_prefixed_state(state: dict[str, Any] | None, prefix: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in (state or {}).items():
        normalized = key.removeprefix("module.")
        full_prefix = prefix + "."
        if normalized.startswith(full_prefix):
            result[normalized.removeprefix(full_prefix)] = value
    return result


def setup_pipeline(args: argparse.Namespace):
    common.ensure_runtime_imports(include_model=True)
    import torch

    from model.train_tokenlight_moge3_pointmap import (
        MoGeStreamTypeEmbedding,
        PointMapCache,
        PointMapConditioner,
    )

    pipe = common.load_pipe(args)
    state = common.load_state(args.checkpoint)
    lora_state = common.extract_lora_state(state)
    if lora_state:
        pipe.load_lora(pipe.dit, state_dict=lora_state, alpha=1.0)

    token_dim = args.token_dim if args.token_dim > 0 else int(pipe.dit.dim)
    light_state = common.extract_light_state(state)
    max_lights, fourier_features = common.infer_light_encoder_shape(
        light_state,
        requested_max_lights=args.tokenlight_max_lights,
        requested_fourier_features=args.fourier_features,
    )
    light_encoder = common.LightokenEncoder(
        token_dim,
        fourier_features=fourier_features,
        fourier_sigma=args.fourier_sigma,
        max_lights=max_lights,
    ).to(device=pipe.device, dtype=pipe.torch_dtype)
    light_encoder.load_state_dict(light_state, strict=False)
    light_encoder.eval()

    type_state = common.extract_type_state(state)
    type_embedding = None
    if type_state:
        num_types = common.infer_type_embedding_num_types(type_state, requested_num_types=4)
        type_embedding = common.TokenLightTypeEmbedding(token_dim, num_types=num_types).to(
            device=pipe.device, dtype=pipe.torch_dtype
        )
        type_embedding.load_state_dict(type_state, strict=False)
        type_embedding.eval()

    conditioner_state = extract_prefixed_state(state, "moge_conditioner")
    stream_type_state = extract_prefixed_state(state, "moge_type_embedding")
    if not conditioner_state or not stream_type_state:
        raise ValueError(
            "Checkpoint does not contain moge_conditioner/moge_type_embedding weights; "
            "use a checkpoint produced by train_tokenlight_moge3_pointmap.py."
        )
    conditioner = PointMapConditioner(
        token_dim,
        hidden_channels=args.moge_hidden_channels,
        camera_distance=args.moge_camera_distance,
        camera_metric_scale=args.moge_camera_metric_scale,
        light_metric_scale=args.moge_light_metric_scale,
    ).to(device=pipe.device, dtype=pipe.torch_dtype)
    conditioner.load_state_dict(conditioner_state, strict=True)
    conditioner.eval()
    stream_type_embedding = MoGeStreamTypeEmbedding(token_dim).to(
        device=pipe.device, dtype=pipe.torch_dtype
    )
    stream_type_embedding.load_state_dict(stream_type_state, strict=True)
    stream_type_embedding.eval()
    pipe.dit.eval()
    cache = PointMapCache(args.moge_cache_root)
    return pipe, light_encoder, type_embedding, conditioner, stream_type_embedding, cache


def generate_one(
    pipe,
    light_encoder,
    type_embedding,
    conditioner,
    stream_type_embedding,
    points,
    valid,
    attrs,
    source,
    mask,
    args,
    *,
    extra_masks,
):
    import model.infer_tokenlight as tokenlight_infer
    from model.train_tokenlight_moge3_pointmap import model_fn_wan_video_tokenlight_moge3

    points = points.unsqueeze(0).to(device=pipe.device, dtype=pipe.torch_dtype)
    valid = valid.unsqueeze(0).to(device=pipe.device, dtype=common.torch.bool)

    def model_adapter(*, tokenlight_light_encoder=None, tokenlight_type_embedding=None, **kwargs):
        return model_fn_wan_video_tokenlight_moge3(
            tokenlight_light_encoder=tokenlight_light_encoder,
            tokenlight_type_embedding=tokenlight_type_embedding,
            tokenlight_moge_conditioner=conditioner,
            tokenlight_moge_type_embedding=stream_type_embedding,
            tokenlight_moge_points=points,
            tokenlight_moge_valid=valid,
            **kwargs,
        )

    original_model_fn = tokenlight_infer.model_fn_wan_video_tokenlight
    tokenlight_infer.model_fn_wan_video_tokenlight = model_adapter
    try:
        return tokenlight_infer.generate(
            pipe,
            light_encoder,
            type_embedding,
            attrs,
            source,
            mask,
            args,
            extra_masks=extra_masks,
        )
    finally:
        tokenlight_infer.model_fn_wan_video_tokenlight = original_model_fn


def run_inference(rows: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> int:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required unless --eval-only is set")
    common.ensure_runtime_imports(include_model=True)
    pipe, light_encoder, type_embedding, conditioner, stream_type_embedding, cache = setup_pipeline(args)
    completed = 0
    desc = Path(args.checkpoint).parent.name + "/" + Path(args.checkpoint).stem
    for row in common.tqdm(rows, desc=desc):
        pred = common.prediction_path(output_dir, row, args)
        target = common.target_path(row, args)
        source_file = common.source_path(row, args)
        if args.skip_existing and pred.exists():
            if args.with_gt and target and target.exists() and not common.with_gt_path(pred).exists():
                common.save_with_gt(pred, source_file, target)
            completed += 1
            continue

        source = common.Image.open(source_file).convert("RGB")
        mask = None
        current_mask = common.mask_path(row, args)
        if args.use_mask_input and current_mask and current_mask.exists():
            mask = common.Image.open(current_mask).convert("RGB")
        extra_paths = common.extra_mask_paths(row, args)
        missing = [path for path in extra_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(missing[0])
        extra_masks = [common.Image.open(path).convert("RGB") for path in extra_paths]
        points, valid = cache.load(source_file)

        infer_args = argparse.Namespace(**vars(args))
        infer_args.prompt = row.get(args.prompt_key) or args.prompt
        if args.seed_key:
            if row.get(args.seed_key) in (None, ""):
                raise KeyError(f"Missing seed key {args.seed_key!r} in row {row.get('_manifest_index')}")
            infer_args.seed = int(row[args.seed_key])
        video = generate_one(
            pipe,
            light_encoder,
            type_embedding,
            conditioner,
            stream_type_embedding,
            points,
            valid,
            common.attrs_from_row(row, args.attrs_key),
            source,
            mask,
            infer_args,
            extra_masks=extra_masks,
        )
        pred.parent.mkdir(parents=True, exist_ok=True)
        video[0].save(pred)
        if args.with_gt and target and target.exists():
            common.save_with_gt(pred, source, target)
        completed += 1
    return completed


def run_worker(device_id: str, rows, output_dir: str, args_dict: dict[str, Any]) -> int:
    args = common.apply_worker_device(device_id, argparse.Namespace(**args_dict))
    print(f"[moge3:{device_id}] rows={len(rows)} device={args.device}", flush=True)
    return run_inference(rows, Path(output_dir), args)


def run_distributed(rows, output_dir: Path, args: argparse.Namespace) -> int:
    devices = common.parse_gpu_devices(args.gpu_devices)
    if not devices:
        return run_inference(rows, output_dir, args)
    assigned = common.split_rows_by_device(rows, devices)
    if len(assigned) == 1:
        device, shard = assigned[0]
        return run_inference(shard, output_dir, common.apply_worker_device(device, args))
    print(f"[moge3] gpu_devices={','.join(devices)} processes={len(assigned)} rows={len(rows)}", flush=True)
    context = mp.get_context("spawn")
    completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(assigned), mp_context=context) as executor:
        futures = [
            executor.submit(run_worker, device, shard, output_dir.as_posix(), vars(args))
            for device, shard in assigned
        ]
        for future in concurrent.futures.as_completed(futures):
            completed += int(future.result())
    return completed


def main() -> int:
    args = parse_args()
    args.manifest = common.resolve_repo(args.manifest).as_posix()
    args.weights_dir = common.resolve_repo(args.weights_dir).as_posix()
    args.moge_cache_root = common.resolve_repo(args.moge_cache_root).as_posix()
    if args.checkpoint:
        args.checkpoint = common.resolve_repo(args.checkpoint).as_posix()
    rows = common.load_rows(Path(args.manifest), int(args.limit))
    output_dir = common.resolve_repo(args.output_dir)
    common.write_snapshot(rows, output_dir, args)

    if not args.eval_only:
        completed = run_distributed(rows, output_dir, args)
        print(f"[moge3] completed={completed} output_dir={output_dir}", flush=True)
    if args.eval or args.eval_only:
        metrics = common.run_eval(rows, output_dir, args)
        metrics_output = (
            common.resolve_repo(args.metrics_output) if args.metrics_output else output_dir / "metrics.json"
        )
        metrics_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "manifest": args.manifest,
            "base_path": common.data_base(args).as_posix(),
            "pred_dir": output_dir.as_posix(),
            "device": str(common.metric_device(args.metric_device)),
            "lpips_net": args.lpips_net,
            **metrics,
        }
        metrics_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[moge3] wrote metrics: {metrics_output}", flush=True)
        print(json.dumps(payload["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
