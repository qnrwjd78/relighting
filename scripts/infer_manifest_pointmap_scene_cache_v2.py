#!/usr/bin/env python3
"""Inference for point-map checkpoints with NPY geometry and cached luminance source latents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import infer as infer_core  # noqa: E402
from scripts import infer_manifest as common  # noqa: E402
from scripts import infer_manifest_moge3_pointmap as old  # noqa: E402
from scripts.infer_manifest_scene_cache_v2 import SceneSourceLatents  # noqa: E402


def parse_args():
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--scene-cache-root", required=True)
    custom, remaining = preliminary.parse_known_args()
    original = sys.argv
    sys.argv = [original[0], *remaining]
    try:
        args = old.parse_args()
    finally:
        sys.argv = original
    return args, custom


def setup_pipeline(args):
    common.ensure_runtime_imports(include_model=True)
    from model.train_moge3_pointmap import MoGeStreamTypeEmbedding, PointMapConditioner

    pipe = common.load_pipe(args)
    state = common.load_state(args.checkpoint)
    lora = common.extract_lora_state(state)
    if lora:
        pipe.load_lora(pipe.dit, state_dict=lora, alpha=1.0)
    token_dim = args.token_dim if args.token_dim > 0 else int(pipe.dit.dim)
    light_state = common.extract_light_state(state)
    max_lights, features = common.infer_light_encoder_shape(
        light_state,
        requested_max_lights=args.tokenlight_max_lights,
        requested_fourier_features=args.fourier_features,
    )
    light_encoder = common.LightokenEncoder(
        token_dim, fourier_features=features, fourier_sigma=args.fourier_sigma, max_lights=max_lights
    ).to(device=pipe.device, dtype=pipe.torch_dtype)
    light_encoder.load_state_dict(light_state, strict=False)
    light_encoder.eval()
    type_state = common.extract_type_state(state)
    type_embedding = common.TokenLightTypeEmbedding(
        token_dim,
        num_types=common.infer_type_embedding_num_types(type_state, requested_num_types=4),
    ).to(device=pipe.device, dtype=pipe.torch_dtype)
    type_embedding.load_state_dict(type_state, strict=False)
    type_embedding.eval()
    conditioner_state = old.extract_prefixed_state(state, "moge_conditioner")
    stream_state = old.extract_prefixed_state(state, "moge_type_embedding")
    if not conditioner_state or not stream_state:
        raise ValueError("Checkpoint has no point-map conditioner weights")
    conditioner = PointMapConditioner(
        token_dim,
        hidden_channels=args.moge_hidden_channels,
        camera_distance=args.moge_camera_distance,
        camera_metric_scale=args.moge_camera_metric_scale,
        light_metric_scale=args.moge_light_metric_scale,
    ).to(device=pipe.device, dtype=pipe.torch_dtype)
    conditioner.load_state_dict(conditioner_state, strict=True)
    conditioner.eval()
    stream_embedding = MoGeStreamTypeEmbedding(token_dim).to(device=pipe.device, dtype=pipe.torch_dtype)
    stream_embedding.load_state_dict(stream_state, strict=True)
    stream_embedding.eval()
    pipe.dit.eval()
    return pipe, light_encoder, type_embedding, conditioner, stream_embedding


def load_point_map(row):
    path = Path(str(row["point_map"]))
    points_array = np.load(path, allow_pickle=False)
    if points_array.shape != (480, 480, 3) or points_array.dtype != np.float32:
        raise ValueError(f"Malformed point map: {path}")
    points = torch.from_numpy(np.array(points_array, copy=True)).permute(2, 0, 1).contiguous()
    valid = torch.isfinite(points).all(dim=0) & (points[2] > 0)
    points = torch.where(valid[None], points, torch.zeros_like(points))
    return points, valid


def main() -> int:
    args, custom = parse_args()
    if args.eval or args.eval_only:
        raise ValueError("Use utils/evaluate_predictions.py --target-transform luminance for this cache")
    devices = common.parse_gpu_devices(args.gpu_devices)
    if len(devices) > 1:
        raise ValueError("This scene-cache point-map launcher accepts one GPU")
    if devices:
        args = common.apply_worker_device(devices[0], args)
    args.manifest = common.resolve_repo(args.manifest).as_posix()
    args.weights_dir = common.resolve_repo(args.weights_dir).as_posix()
    args.checkpoint = common.resolve_repo(args.checkpoint).as_posix()
    rows = common.load_rows(Path(args.manifest), int(args.limit))
    output_dir = common.resolve_repo(args.output_dir)
    common.write_snapshot(rows, output_dir, args)
    (output_dir / "pointmap_scene_cache_inference.json").write_text(
        json.dumps(
            {
                "scene_cache_root": Path(custom.scene_cache_root).resolve().as_posix(),
                "point_map_field": "point_map",
                "source_condition": "cached luminance source_latent",
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    pipe, light, type_embedding, conditioner, stream_embedding = setup_pipeline(args)
    source_cache = SceneSourceLatents(Path(custom.scene_cache_root))
    completed = 0
    for row in common.tqdm(rows, desc=Path(args.checkpoint).parent.name + "/" + Path(args.checkpoint).stem):
        prediction = common.prediction_path(output_dir, row, args)
        target = common.target_path(row, args)
        source_path = common.source_path(row, args)
        if args.skip_existing and prediction.exists():
            completed += 1
            continue
        source_latents = source_cache.get(str(row["scene_id"]))

        def cached_source(current_pipe, image, current_args):
            del image, current_args
            return source_latents.unsqueeze(0).to(device=current_pipe.device, dtype=current_pipe.torch_dtype)

        infer_core.encode_image_latents = cached_source
        points, valid = load_point_map(row)
        source = Image.open(source_path).convert("RGB")
        current_args = argparse.Namespace(**vars(args))
        current_args.prompt = row.get(args.prompt_key) or args.prompt
        video = old.generate_one(
            pipe,
            light,
            type_embedding,
            conditioner,
            stream_embedding,
            points,
            valid,
            common.attrs_from_row(row, args.attrs_key),
            source,
            None,
            current_args,
            extra_masks=None,
        )
        prediction.parent.mkdir(parents=True, exist_ok=True)
        video[0].save(prediction)
        if args.with_gt and target and target.exists():
            common.save_with_gt(prediction, source, target)
        completed += 1
    print(f"completed={completed} output_dir={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
