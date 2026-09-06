#!/usr/bin/env python3
"""Fail-fast validation for the exp_1 shadow-mask VAE conditioning run."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import train as base  # noqa: E402
from model import train_decoder_safe as safe  # noqa: E402
from model import train_scene_cache_v2 as scene_cache  # noqa: E402
from model.train_scene_cache_shadow_safe_retained import (  # noqa: E402
    DistributedSceneQuotaBatchSampler,
)


DEFAULT_CONFIG = (
    ROOT
    / "configs/train_480/exp1_7x7x5_power06_rgb_shadow_mask_vae_scene64_15ep_b5x8_ga1_gb40.json"
)
DEFAULT_ACCELERATE_CONFIG = ROOT / "configs/accelerate_8gpu_ddp.yaml"
EXPECTED_MASK_FILENAME = "object_shadow_geometry_ray_clean_minarea00.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--accelerate-config", type=Path, default=DEFAULT_ACCELERATE_CONFIG)
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--expected-global-batch", type=int, default=40)
    parser.add_argument("--sample-count", type=int, default=32)
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def flatten(config: dict) -> dict:
    result = {}
    for section in config.values():
        if isinstance(section, dict):
            result.update(section)
    return result


def metadata_audit(metadata_path: Path, workspace: Path) -> tuple[list[dict], set[str]]:
    rows: list[dict] = []
    mask_paths: set[str] = set()
    identities: set[tuple[str, str]] = set()
    missing = []
    with metadata_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            identity = (str(row.get("scene_id")), str(row.get("sample_name")))
            if identity in identities:
                raise ValueError(f"Duplicate metadata identity at line {line_number}: {identity}")
            identities.add(identity)
            value = row.get("shadow_mask")
            if not isinstance(value, str) or not value:
                raise KeyError(f"Missing shadow_mask at metadata line {line_number}")
            if Path(value).name != EXPECTED_MASK_FILENAME:
                raise ValueError(
                    f"Unexpected shadow mask at metadata line {line_number}: {value}"
                )
            path = Path(value)
            path = path if path.is_absolute() else workspace / path
            if not path.is_file():
                missing.append(path.as_posix())
                if len(missing) >= 10:
                    break
            mask_paths.add(value)
            rows.append(row)
    if missing:
        raise FileNotFoundError(f"Missing shadow masks (first {len(missing)}): {missing}")
    return rows, mask_paths


def accelerate_config_audit(path: Path, expected_world: int) -> dict:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to validate the Accelerate config") from error
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed Accelerate config: {path}")
    expected = {
        "distributed_type": "MULTI_GPU",
        "num_processes": expected_world,
        "mixed_precision": "bf16",
        "use_cpu": False,
    }
    mismatches = {
        key: {"actual": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Accelerate config mismatch: {mismatches}")
    return payload


def cache_config_audit(
    cache_root: Path,
    metadata_path: Path,
    weights_root: Path,
    expected_rows: int,
    expected_world: int,
) -> dict:
    path = cache_root / "cache_config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "cache_kind": "vae_latent_cache",
        "mode": "custom",
        "transform": "none",
        "image_keys": ["shadow_mask"],
        "height": 480,
        "width": 480,
        "vae_dtype": "bf16",
        "save_dtype": "bf16",
        "row_count": expected_rows,
        "asset_count": expected_rows,
        "merged_index_rows": expected_rows,
        "partition_count": expected_world,
    }
    mismatches = {
        key: {"actual": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    resolved_expected = {
        "data_root": ROOT.resolve(),
        "metadata_path": metadata_path.resolve(),
        "output_dir": cache_root.resolve(),
        "vae_path": (weights_root / "Wan2.2_VAE.pth").resolve(),
    }
    for key, value in resolved_expected.items():
        actual = payload.get(key)
        if not isinstance(actual, str) or Path(actual).resolve() != value:
            mismatches[key] = {"actual": actual, "expected": value.as_posix()}
    if mismatches:
        raise ValueError(f"Mask-cache provenance mismatch: {mismatches}")
    return payload


def cache_index_audit(cache_root: Path, metadata_masks: set[str]) -> tuple[int, int]:
    index_path = cache_root / "index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing merged mask-cache index: {index_path}")
    paths: set[str] = set()
    asset_indices: set[int] = set()
    shards: set[str] = set()
    with index_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            path = row.get("path")
            if not isinstance(path, str) or not path:
                raise ValueError(f"Missing cache path at index line {line_number}")
            if path in paths:
                raise ValueError(f"Duplicate cache path at index line {line_number}: {path}")
            paths.add(path)
            asset_index = int(row.get("asset_index", -1))
            if asset_index in asset_indices:
                raise ValueError(
                    f"Duplicate cache asset_index at line {line_number}: {asset_index}"
                )
            asset_indices.add(asset_index)
            if row.get("keys") != ["shadow_mask"]:
                raise ValueError(f"Unexpected cache keys at line {line_number}: {row.get('keys')}")
            if row.get("shape") != [48, 1, 30, 30] or row.get("dtype") != "bfloat16":
                raise ValueError(
                    f"Unexpected cached tensor schema at line {line_number}: "
                    f"shape={row.get('shape')}, dtype={row.get('dtype')}"
                )
            shard = row.get("shard")
            if not isinstance(shard, str) or not shard:
                raise ValueError(f"Missing shard at index line {line_number}")
            shards.add(shard)
    if paths != metadata_masks:
        missing = sorted(metadata_masks - paths)[:10]
        extra = sorted(paths - metadata_masks)[:10]
        raise ValueError(
            f"Raw mask-cache index mismatch: missing={len(metadata_masks-paths)} {missing}, "
            f"extra={len(paths-metadata_masks)} {extra}"
        )
    expected_indices = set(range(len(metadata_masks)))
    if asset_indices != expected_indices:
        raise ValueError("Mask-cache asset_index values are not exactly contiguous from zero")
    missing_shards = sorted(
        shard for shard in shards if not (cache_root / shard).is_file()
    )
    if missing_shards:
        raise FileNotFoundError(f"Missing mask-cache shards: {missing_shards[:10]}")
    return len(paths), len(shards)


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    accelerate_config_path = args.accelerate_config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    values = flatten(config)
    metadata_path = resolve(values["dataset_metadata_path"])
    rgb_cache_root = resolve(values["dataset_base_path"])
    mask_cache_root = resolve(values["tokenlight_mask_latent_cache_dir"])
    weights_root = resolve(values["weights_dir"])

    expected_world = int(args.expected_world_size)
    expected_global_batch = int(args.expected_global_batch)
    accelerate_config_audit(accelerate_config_path, expected_world)
    batch = int(values["batch_size"])
    accumulation = int(values["gradient_accumulation_steps"])
    effective = expected_world * batch * accumulation
    if effective != expected_global_batch:
        raise ValueError(
            f"Global batch drift: {expected_world}*{batch}*{accumulation}={effective}, "
            f"expected {expected_global_batch}"
        )
    if values.get("balanced_task_batch") != f"position:{batch}":
        raise ValueError("balanced_task_batch must exactly match the per-rank position batch")
    if not bool(values.get("tokenlight_mask_tokens")):
        raise ValueError("tokenlight_mask_tokens must be enabled")
    if values.get("tokenlight_mask_image_key") != "shadow_mask":
        raise ValueError("tokenlight_mask_image_key must be shadow_mask")
    if float(values.get("tokenlight_rgb_latent_loss_weight", 0)) != 1.0:
        raise ValueError("Expected latent FlowMatch weight 1.0")
    if float(values.get("tokenlight_rgb_decoder_loss_weight", 0)) != 0.0:
        raise ValueError("This run must not enable decoder-space loss")

    rows, metadata_masks = metadata_audit(metadata_path, ROOT)
    if len(rows) != 326_803 or len(metadata_masks) != 326_803:
        raise ValueError(
            f"Unexpected metadata coverage: rows={len(rows)}, unique_masks={len(metadata_masks)}"
        )
    if len({str(row["scene_id"]) for row in rows}) != 1_482:
        raise ValueError("Expected 1,482 scenes in the combined exp_1 metadata")

    scene_cache_files = {str(row.get("_scene_cache_file") or "") for row in rows}
    if "" in scene_cache_files or len(scene_cache_files) != 1_482:
        raise ValueError(
            f"Unexpected scene-cache references: {len(scene_cache_files - {''})}"
        )
    missing_scene_caches = sorted(
        relative for relative in scene_cache_files if not (rgb_cache_root / relative).is_file()
    )
    if missing_scene_caches:
        raise FileNotFoundError(f"Missing RGB scene caches: {missing_scene_caches[:10]}")

    cache_config = cache_config_audit(
        mask_cache_root,
        metadata_path,
        weights_root,
        len(rows),
        expected_world,
    )
    index_rows, shard_count = cache_index_audit(mask_cache_root, metadata_masks)
    if int(cache_config.get("shard_count", -1)) != shard_count:
        raise ValueError(
            f"Mask-cache shard count mismatch: config={cache_config.get('shard_count')}, "
            f"index={shard_count}"
        )

    store = base.VaeLatentCacheStore(
        mask_cache_root,
        shard_lru=int(values.get("vae_latent_cache_shard_lru", 8)),
    )
    cache_masks = set(store.index)
    if cache_masks != metadata_masks:
        missing = sorted(metadata_masks - cache_masks)[:10]
        extra = sorted(cache_masks - metadata_masks)[:10]
        raise ValueError(
            f"Mask-cache index mismatch: missing={len(metadata_masks-cache_masks)} {missing}, "
            f"extra={len(cache_masks-metadata_masks)} {extra}"
        )

    previous_build_dataset = base.build_dataset
    base.build_dataset = scene_cache.build_scene_cache_dataset
    try:
        dataset = safe.build_safe_dataset(argparse.Namespace(**values))
    finally:
        base.build_dataset = previous_build_dataset
    if not isinstance(dataset, safe._MaskLatentCacheDataset):
        raise TypeError(f"Actual training composition did not attach mask cache: {type(dataset)}")
    if not isinstance(dataset.dataset, scene_cache.WanVaeSceneCacheV2Dataset):
        raise TypeError(f"Actual training composition did not use scene RGB cache: {type(dataset.dataset)}")
    indices = random.Random(480).sample(range(len(dataset)), min(int(args.sample_count), len(dataset)))
    nonfinite = 0
    all_zero = 0
    for index in indices:
        item = dataset[index]
        target = item["_tokenlight_input_latents"]
        source = item["_tokenlight_source_latents"]
        mask = item["_tokenlight_mask_latents"]
        for name, tensor in (("target", target), ("source", source), ("mask", mask)):
            if tuple(tensor.shape) != (48, 1, 30, 30):
                raise ValueError(f"Bad {name} latent shape at {index}: {tuple(tensor.shape)}")
            nonfinite += int(not bool(torch.isfinite(tensor.float()).all()))
        all_zero += int(not bool(mask.float().abs().any()))
    if nonfinite:
        raise ValueError(f"Found {nonfinite} non-finite sampled latent tensors")
    if all_zero:
        raise ValueError(f"Found {all_zero} exact-zero sampled mask latent tensors")

    collate_count = min(batch, len(indices))
    collated = base._collate_tokenlight_batch([dataset[index] for index in indices[:collate_count]])
    collated_mask = collated.get("_tokenlight_mask_latents")
    if not isinstance(collated_mask, torch.Tensor):
        raise TypeError("Actual collated training batch is missing _tokenlight_mask_latents")
    expected_collated_shape = (collate_count, 48, 1, 30, 30)
    if tuple(collated_mask.shape) != expected_collated_shape:
        raise ValueError(
            f"Unexpected collated mask latent shape: {tuple(collated_mask.shape)}, "
            f"expected={expected_collated_shape}"
        )

    previous_world = os.environ.get("WORLD_SIZE")
    os.environ["WORLD_SIZE"] = str(expected_world)
    try:
        sampler = DistributedSceneQuotaBatchSampler(
            rows,
            {"position": batch},
            seed=int(values.get("balanced_batch_seed", 480)),
        )
    finally:
        if previous_world is None:
            os.environ.pop("WORLD_SIZE", None)
        else:
            os.environ["WORLD_SIZE"] = previous_world
    if sampler.selected_rows % (batch * expected_world):
        raise ValueError("Scene sampler does not emit complete global batches")

    required_weights = [
        weights_root / "diffusion_pytorch_model-00001-of-00003.safetensors",
        weights_root / "diffusion_pytorch_model-00002-of-00003.safetensors",
        weights_root / "diffusion_pytorch_model-00003-of-00003.safetensors",
        weights_root / "models_t5_umt5-xxl-enc-bf16.pth",
        weights_root / "Wan2.2_VAE.pth",
    ]
    missing_weights = [path.as_posix() for path in required_weights if not path.is_file()]
    if missing_weights:
        raise FileNotFoundError(f"Missing base weights: {missing_weights}")
    gpu_count = torch.cuda.device_count()
    if gpu_count < expected_world:
        raise RuntimeError(f"Need {expected_world} visible GPUs, found {gpu_count}")

    report = {
        "status": "ready",
        "config": config_path.as_posix(),
        "accelerate_config": accelerate_config_path.as_posix(),
        "metadata_rows": len(rows),
        "scenes": len({str(row["scene_id"]) for row in rows}),
        "mask_cache_entries": len(store.index),
        "mask_cache_index_rows": index_rows,
        "mask_cache_shards": shard_count,
        "mask_filename": EXPECTED_MASK_FILENAME,
        "sampled_latents": len(indices),
        "sampled_zero_mask_latents": all_zero,
        "latent_shape": [48, 1, 30, 30],
        "world_size": expected_world,
        "per_rank_batch": batch,
        "gradient_accumulation": accumulation,
        "effective_global_batch": effective,
        "scene_quota_available_rows": sampler.available_rows,
        "scene_quota_selected_rows": sampler.selected_rows,
        "scene_quota_dropped_rows": sampler.dropped_rows,
        "optimizer_steps_per_epoch": math.ceil(len(sampler) / expected_world / accumulation),
        "visible_gpus": gpu_count,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
