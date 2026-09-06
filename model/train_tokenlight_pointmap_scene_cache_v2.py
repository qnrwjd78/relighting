#!/usr/bin/env python3
"""Baseline TokenLight training with Objaverse-245 NPY point maps.

This is isolated from the existing baseline and delta trainers.  It uses the
same baseline flow-matching loss, scene-64 sampler, source dropout, and retained
one-based checkpoint policy requested for the exp_1 runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import accelerate
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import train_tokenlight as base  # noqa: E402
from model import train_tokenlight_moge3_pointmap as geometry  # noqa: E402
from model import train_tokenlight_scene_cache_v2 as scene_cache  # noqa: E402
from model.train_tokenlight_scene_cache_v2_retained import OneBasedRetainedModelLogger  # noqa: E402


class NpyPointMapSceneDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, point_root: Path, object_mask_root: Path, max_open_scenes: int = 8) -> None:
        self.dataset = dataset
        self.point_root = point_root.resolve()
        self.object_mask_root = object_mask_root.resolve()
        self.max_open_scenes = max(1, int(max_open_scenes))
        self._open: OrderedDict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = OrderedDict()
        self.data = dataset.data
        self.repeat = getattr(dataset, "repeat", 1)
        self.load_from_cache = getattr(dataset, "load_from_cache", False)
        self.skip_tokenlight_file_filter = True
        self.is_vae_latent_cache_dataset = True
        self.is_scene_latent_cache_dataset = True
        scene_ids = {str(row["scene_id"]) for row in self.data}
        missing_points = [scene for scene in sorted(scene_ids) if not (self.point_root / scene / "source.npy").is_file()]
        missing_masks = [
            scene for scene in sorted(scene_ids)
            if not (self.object_mask_root / scene / "object_mask_15.npy").is_file()
        ]
        if missing_points or missing_masks:
            raise FileNotFoundError(
                f"Point-map dataset incomplete: missing_points={missing_points[:5]} ({len(missing_points)}), "
                f"missing_masks={missing_masks[:5]} ({len(missing_masks)})"
            )
        print(
            f"Point-map dataset: scenes={len(scene_ids)}, point_root={self.point_root}, "
            f"object_mask_root={self.object_mask_root}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_open"] = OrderedDict()
        return state

    def _geometry(self, scene_id: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self._open.get(scene_id)
        if cached is not None:
            self._open.move_to_end(scene_id)
            return cached
        points_array = np.load(self.point_root / scene_id / "source.npy", allow_pickle=False)
        mask_array = np.load(self.object_mask_root / scene_id / "object_mask_15.npy", allow_pickle=False)
        if points_array.shape != (480, 480, 3) or points_array.dtype != np.float32:
            raise ValueError(f"Malformed point map for {scene_id}: {points_array.shape}/{points_array.dtype}")
        if mask_array.shape != (15, 15):
            raise ValueError(f"Malformed object token mask for {scene_id}: {mask_array.shape}")
        points = torch.from_numpy(np.array(points_array, copy=True)).permute(2, 0, 1).contiguous()
        valid = torch.isfinite(points).all(dim=0) & (points[2] > 0)
        points = torch.where(valid[None], points, torch.zeros_like(points))
        object_mask = torch.from_numpy(np.array(mask_array, dtype=np.uint8, copy=True)).bool()
        cached = (points, valid, object_mask)
        self._open[scene_id] = cached
        self._open.move_to_end(scene_id)
        while len(self._open) > self.max_open_scenes:
            self._open.popitem(last=False)
        return cached

    def _get(self, index: int) -> dict[str, Any]:
        item = dict(self.dataset[int(index)])
        scene_id = str(item["scene_id"])
        points, valid, object_mask = self._geometry(scene_id)
        item["_tokenlight_moge_points"] = points
        item["_tokenlight_moge_valid"] = valid
        item["_tokenlight_moge_object_mask"] = object_mask
        return item

    def __getitem__(self, index: int):
        return self._get(int(index))

    def __getitems__(self, indices: list[int]):
        return [self._get(int(index)) for index in indices]


class TokenLightPointMapBaselineModule(base.TokenLightWanTrainingModule):
    def __init__(
        self,
        *args,
        tokenlight_moge_hidden_channels: int = 64,
        tokenlight_moge_camera_distance: float = 3.5,
        tokenlight_moge_camera_metric_scale: float = 0.75,
        tokenlight_moge_light_metric_scale: float = 14.0 / 15.0,
        tokenlight_moge_initial_scale: float = 1.0,
        tokenlight_moge_trainable_scale: bool = True,
        tokenlight_moge_source_dropout_prob: float = 0.3,
        tokenlight_moge_source_dropout_region_fraction: float = 0.3,
        **kwargs,
    ) -> None:
        checkpoint = kwargs.get("lora_checkpoint") or kwargs.get("resume_from_checkpoint")
        super().__init__(*args, **kwargs)
        token_dim = int(self.pipe.dit.dim)
        self.moge_conditioner = geometry.PointMapConditioner(
            token_dim,
            hidden_channels=tokenlight_moge_hidden_channels,
            camera_distance=tokenlight_moge_camera_distance,
            camera_metric_scale=tokenlight_moge_camera_metric_scale,
            light_metric_scale=tokenlight_moge_light_metric_scale,
            initial_scale=tokenlight_moge_initial_scale,
            trainable_scale=tokenlight_moge_trainable_scale,
        )
        self.moge_type_embedding = geometry.MoGeStreamTypeEmbedding(token_dim)
        self.tokenlight_moge_source_dropout_prob = float(tokenlight_moge_source_dropout_prob)
        self.tokenlight_moge_source_dropout_region_fraction = float(
            tokenlight_moge_source_dropout_region_fraction
        )
        auxiliary = base._load_checkpoint_state_dict(checkpoint)
        base._load_module_state_from_checkpoint(self.moge_conditioner, auxiliary, "moge_conditioner")
        base._load_module_state_from_checkpoint(self.moge_type_embedding, auxiliary, "moge_type_embedding")
        self.pipe.model_fn = self._tokenlight_model_fn

    def _tokenlight_model_fn(self, **kwargs):
        return geometry.model_fn_wan_video_tokenlight_moge3(
            **kwargs,
            tokenlight_light_encoder=self.light_encoder,
            tokenlight_type_embedding=self.tokenlight_type_embedding,
            tokenlight_moge_conditioner=self.moge_conditioner,
            tokenlight_moge_type_embedding=self.moge_type_embedding,
            tokenlight_moge_source_dropout_prob=self.tokenlight_moge_source_dropout_prob,
            tokenlight_moge_source_dropout_region_fraction=self.tokenlight_moge_source_dropout_region_fraction,
            tokenlight_moge_training=self.training,
        )

    def _attach_geometry(self, inputs, data):
        inputs_shared, inputs_posi, inputs_nega = inputs
        batch = int(inputs_shared["input_latents"].shape[0])
        device = inputs_shared["input_latents"].device
        points = geometry._stack_pointmaps(
            data.get("_tokenlight_moge_points"), batch=batch, item_ndim=3, name="point XYZ", device=device
        )
        valid = geometry._stack_pointmaps(
            data.get("_tokenlight_moge_valid"), batch=batch, item_ndim=2, name="point valid", device=device
        )
        object_mask = geometry._stack_pointmaps(
            data.get("_tokenlight_moge_object_mask"), batch=batch, item_ndim=2, name="object mask", device=device
        )
        if points.shape[1:] != (3, 480, 480) or valid.shape[1:] != (480, 480):
            raise ValueError(f"Unexpected point geometry: points={points.shape}, valid={valid.shape}")
        if object_mask.shape[1:] != (15, 15):
            raise ValueError(f"Unexpected token object mask: {object_mask.shape}")
        inputs_shared["tokenlight_moge_points"] = points.float()
        inputs_shared["tokenlight_moge_valid"] = valid.bool()
        inputs_shared["tokenlight_moge_object_mask"] = object_mask.bool()
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None and self._has_cached_latents(data):
            prepared = self._attach_geometry(self._cached_batched_inputs(data), data)
            loss = self.task_to_loss[self.task](self.pipe, *prepared)
        else:
            raise ValueError("Point-map baseline requires collated scene latent-cache rows")
        metrics = getattr(self.moge_conditioner, "_last_metrics", None)
        if isinstance(metrics, Mapping):
            current = getattr(self.pipe, "_tokenlight_loss_metrics", {})
            self.pipe._tokenlight_loss_metrics = {**current, **metrics}
        return loss

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        exported = super().export_trainable_state_dict(state_dict, remove_prefix=remove_prefix)
        for prefix in ("moge_conditioner.", "moge_type_embedding."):
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    exported[key] = value
        return exported


def parser() -> argparse.ArgumentParser:
    result = base.wan_parser("single")
    result.add_argument("--tokenlight_moge_point_root", required=True)
    result.add_argument("--tokenlight_moge_object_mask_root", required=True)
    result.add_argument("--tokenlight_moge_hidden_channels", type=int, default=64)
    result.add_argument("--tokenlight_moge_camera_distance", type=float, default=3.5)
    result.add_argument("--tokenlight_moge_camera_metric_scale", type=float, default=0.75)
    result.add_argument("--tokenlight_moge_light_metric_scale", type=float, default=14.0 / 15.0)
    result.add_argument("--tokenlight_moge_initial_scale", type=float, default=1.0)
    result.add_argument("--tokenlight_moge_trainable_scale", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--tokenlight_moge_source_dropout_prob", type=float, default=0.3)
    result.add_argument("--tokenlight_moge_source_dropout_region_fraction", type=float, default=0.3)
    return result


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", required=True)
    preliminary, _ = pre.parse_known_args()
    raw = base._load_json_config(preliminary.config)
    argument_parser = parser()
    base._apply_config_defaults(argument_parser, raw)
    argument_parser.set_defaults(config=preliminary.config)
    args = argument_parser.parse_args()
    if args.lora_checkpoint not in (None, "", "None", "null") or int(args.start_epoch) != 0:
        argument_parser.error("fresh point-map training requires lora_checkpoint=null and start_epoch=0")
    if not 0.0 <= args.tokenlight_moge_source_dropout_prob <= 1.0:
        argument_parser.error("source dropout probability must be in [0,1]")
    if not 0.0 < args.tokenlight_moge_source_dropout_region_fraction < 1.0:
        argument_parser.error("source dropout region fraction must be in (0,1)")
    base._resolve_weight_paths(args)
    base._append_timestamp_to_output_path(args)
    return args, raw


def main() -> int:
    args, raw = parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)
        ],
    )
    base.save_training_config_snapshot(args, raw, raw, accelerator)
    dataset = scene_cache.build_scene_cache_dataset(args)
    dataset = NpyPointMapSceneDataset(
        dataset,
        base._resolve_path(args.tokenlight_moge_point_root, base_path=ROOT),
        base._resolve_path(args.tokenlight_moge_object_mask_root, base_path=ROOT),
        max_open_scenes=int(args.vae_latent_cache_shard_lru),
    )
    model = TokenLightPointMapBaselineModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=getattr(args, "use_gradient_checkpointing_offload", False),
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        resume_from_checkpoint=getattr(args, "resume_from_checkpoint", None),
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        tokenlight_light_tokens=args.tokenlight_light_tokens,
        tokenlight_attrs_key=args.tokenlight_attrs_key,
        tokenlight_token_dim=args.tokenlight_token_dim,
        tokenlight_fourier_features=args.tokenlight_fourier_features,
        tokenlight_fourier_sigma=args.tokenlight_fourier_sigma,
        tokenlight_max_lights=args.tokenlight_max_lights,
        tokenlight_light_dropout=args.tokenlight_light_dropout,
        tokenlight_cfg_drop_prob=args.tokenlight_cfg_drop_prob,
        tokenlight_source_tokens=args.tokenlight_source_tokens,
        tokenlight_mask_tokens=args.tokenlight_mask_tokens,
        prompt_context_cache_size=args.prompt_context_cache_size,
        tokenlight_moge_hidden_channels=args.tokenlight_moge_hidden_channels,
        tokenlight_moge_camera_distance=args.tokenlight_moge_camera_distance,
        tokenlight_moge_camera_metric_scale=args.tokenlight_moge_camera_metric_scale,
        tokenlight_moge_light_metric_scale=args.tokenlight_moge_light_metric_scale,
        tokenlight_moge_initial_scale=args.tokenlight_moge_initial_scale,
        tokenlight_moge_trainable_scale=args.tokenlight_moge_trainable_scale,
        tokenlight_moge_source_dropout_prob=args.tokenlight_moge_source_dropout_prob,
        tokenlight_moge_source_dropout_region_fraction=args.tokenlight_moge_source_dropout_region_fraction,
    )
    base.BalancedTaskBatchSampler = scene_cache.SceneQuotaBatchSampler
    logger = OneBasedRetainedModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt)
    base.launch_tokenlight_training_task(accelerator, dataset, model, logger, args=args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
