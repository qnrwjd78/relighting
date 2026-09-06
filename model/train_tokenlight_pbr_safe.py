from __future__ import annotations

"""Safe RGB + depth/normal TokenLight training.

This is an isolated entrypoint.  It intentionally does not modify or import
``model.train_tokenlight_pbr`` because that legacy trainer is coupled to an
older base trainer and contains the retired decoder objective.  The joint
stream transformer implementation lives in the new
``model.tokenlight_wan_pbr_safe`` module
while optimization follows ``model.train_tokenlight_decoder_safe``.

Depth and normal are UniRelight-style streams.  For every sample they are
either noisy targets, clean conditions, or clean conditions with the source
RGB dropped.  Only latent velocity losses are supported here; every decoded
image-space loss is rejected by argument validation.
"""

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import accelerate
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import train_tokenlight as base  # noqa: E402
from model import train_tokenlight_decoder_safe as safe  # noqa: E402
from model import tokenlight_physics_runtime as physics_runtime  # noqa: E402
from model.lightoken_encoder import LightokenEncoder, parse_attrs_json  # noqa: E402
from model.tokenlight_wan_pbr_safe import (  # noqa: E402
    TokenLightPBRTypeEmbedding,
    model_fn_wan_video_tokenlight_pbr,
    tokenlight_pbr_type_count,
)


LOSS_IMPL_VERSION = "tokenlight_pbr_latent_safe_v1"
DEFAULT_CONFIG_PATH = "configs/train_480/fixed32_pbr_depth_normal.json"


def _csv_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, Mapping):
        items = value.keys()
    else:
        items = value
    names = [str(item).strip() for item in items if str(item).strip()]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate PBR stream names: {names}")
    return names


def _name_map(value: Any, *, cast) -> dict[str, Any]:
    if value in (None, "", "none", "None", "null"):
        return {}
    if isinstance(value, Mapping):
        raw = value
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith("{"):
            loaded = json.loads(text)
            if not isinstance(loaded, Mapping):
                raise ValueError("Expected a JSON object for a PBR stream map")
            raw = loaded
        else:
            raw = {}
            for item in text.split(","):
                if not item.strip():
                    continue
                key, mapped = item.split(":", 1)
                raw[key.strip()] = mapped.strip()
    else:
        raise TypeError(f"Unsupported PBR stream map: {type(value).__name__}")
    return {str(key): cast(item) for key, item in raw.items()}


def _sample_mask(value: Any, *, batch: int, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        result = value.to(device=device, dtype=torch.bool).flatten()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = torch.tensor(list(value), device=device, dtype=torch.bool).flatten()
    else:
        result = torch.full((batch,), bool(value), device=device, dtype=torch.bool)
    if result.numel() == 1:
        result = result.expand(batch)
    if result.numel() != batch:
        raise ValueError(f"Expected {batch} stream-role values, got {result.numel()}")
    return result


def _per_sample_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"Velocity shape mismatch: {tuple(pred.shape)} != {tuple(target.shape)}")
    return (pred.float() - target.float()).square().flatten(1).mean(dim=1)


def TokenLightPbrSafeFlowMatchLoss(pipe, **inputs):
    """Joint RGB/PBR rectified-flow loss with no decoder-space branch."""

    if "lora" in inputs:
        pipe.clear_lora(verbose=0)
        pipe.load_lora(pipe.dit, state_dict=inputs["lora"], hotload=True, verbose=0)

    rgb_clean = inputs["input_latents"]
    timestep_id, timestep, sigma_scalar, rgb_noisy, rgb_target = safe._sample_flow_state(
        pipe,
        rgb_clean,
        inputs,
    )
    sigma = sigma_scalar.to(device=rgb_clean.device, dtype=rgb_clean.dtype).reshape(
        (1,) + (1,) * (rgb_clean.ndim - 1)
    )

    model_rgb = rgb_noisy
    if "first_frame_latents" in inputs:
        model_rgb = rgb_noisy.clone()
        model_rgb[:, :, 0:1] = inputs["first_frame_latents"]
    inputs["latents"] = model_rgb

    clean_streams = inputs.get("tokenlight_pbr_input_latents_map")
    if not isinstance(clean_streams, Mapping) or not clean_streams:
        raise ValueError("PBR-safe training requires tokenlight_pbr_input_latents_map")

    stream_roles = inputs.get("tokenlight_pbr_is_target_map", {})
    stream_model_latents: dict[str, torch.Tensor] = {}
    stream_targets: dict[str, torch.Tensor] = {}
    stream_target_masks: dict[str, torch.Tensor] = {}
    for stream_name, clean in clean_streams.items():
        if clean.shape != rgb_clean.shape:
            raise ValueError(
                f"PBR stream {stream_name!r} must share the RGB latent grid: "
                f"{tuple(clean.shape)} != {tuple(rgb_clean.shape)}"
            )
        batch = int(clean.shape[0])
        target_mask = _sample_mask(
            stream_roles.get(stream_name, True) if isinstance(stream_roles, Mapping) else True,
            batch=batch,
            device=clean.device,
        )
        noise = torch.randn_like(clean) * float(inputs.get("noise_scale", 1.0))
        noisy = (1.0 - sigma) * clean + sigma * noise
        mask = target_mask.reshape(batch, 1, 1, 1, 1)
        stream_model_latents[str(stream_name)] = torch.where(mask, noisy, clean)
        stream_targets[str(stream_name)] = noise - clean
        stream_target_masks[str(stream_name)] = target_mask

    inputs["tokenlight_pbr_stream_latents"] = stream_model_latents
    inputs["tokenlight_pbr_stream_is_target"] = stream_target_masks
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    prediction = pipe.model_fn(**models, **inputs, timestep=timestep)
    if not isinstance(prediction, tuple) or len(prediction) != 2:
        raise ValueError("PBR model_fn must return (rgb_velocity, stream_velocity_map)")
    rgb_pred, pbr_pred = prediction
    if not isinstance(pbr_pred, Mapping):
        raise ValueError("PBR model_fn must return a mapping of stream predictions")

    rgb_target_for_loss = rgb_target
    if "first_frame_latents" in inputs:
        rgb_pred = rgb_pred[:, :, 1:]
        rgb_target_for_loss = rgb_target_for_loss[:, :, 1:]
    rgb_per_sample = _per_sample_mse(rgb_pred, rgb_target_for_loss)
    rgb_weights = inputs.get("tokenlight_rgb_sample_weights")
    if rgb_weights is None:
        rgb_weights = torch.ones_like(rgb_per_sample)
    else:
        rgb_weights = torch.as_tensor(
            rgb_weights,
            device=rgb_per_sample.device,
            dtype=torch.float32,
        ).flatten()
        if rgb_weights.numel() == 1:
            rgb_weights = rgb_weights.expand_as(rgb_per_sample)
        if rgb_weights.numel() != rgb_per_sample.numel():
            raise ValueError("RGB sample-weight count does not match the batch")
    rgb_raw = (rgb_per_sample * rgb_weights).mean()
    rgb_weight = float(inputs.get("tokenlight_rgb_latent_loss_weight", 1.0))
    latent_total = rgb_weight * rgb_raw

    metrics: dict[str, torch.Tensor] = {
        "train/pbr/rgb_raw_mse": rgb_raw.detach(),
        "train/pbr/rgb_latent_term": (rgb_weight * rgb_raw).detach(),
        "train/pbr/sigma": sigma_scalar.detach(),
    }
    stream_weights = inputs.get("tokenlight_pbr_loss_weights", {})
    pbr_outer_weight = float(inputs.get("tokenlight_pbr_latent_loss_weight", 1.0))
    for stream_name, target in stream_targets.items():
        stream_pred = pbr_pred.get(stream_name)
        if stream_pred is None:
            raise KeyError(f"Missing model output for PBR stream {stream_name!r}")
        per_sample = _per_sample_mse(stream_pred, target)
        target_mask = stream_target_masks[stream_name].to(device=per_sample.device, dtype=torch.float32)
        # Mean over every sample, including zero-valued condition samples.  The
        # resulting objective is invariant to how samples are split over ranks.
        raw = (per_sample * target_mask).mean()
        stream_weight = float(stream_weights.get(stream_name, 1.0))
        term = pbr_outer_weight * stream_weight * raw
        latent_total = latent_total + term
        metrics[f"train/pbr/{stream_name}_raw_mse"] = raw.detach()
        metrics[f"train/pbr/{stream_name}_latent_term"] = term.detach()
        metrics[f"train/pbr/{stream_name}_target_fraction"] = target_mask.mean().detach()

    scheduler_weight = safe._scheduler_training_weight(pipe, timestep_id, latent_total)
    total = scheduler_weight * latent_total
    source_drop = inputs.get("tokenlight_source_drop_mask")
    if source_drop is not None:
        metrics["train/pbr/source_drop_fraction"] = torch.as_tensor(
            source_drop,
            device=total.device,
            dtype=torch.float32,
        ).mean().detach()
    metrics["train/pbr/scheduler_weight"] = scheduler_weight.detach()
    metrics["train/pbr/latent_total_before_scheduler"] = latent_total.detach()
    metrics["train/pbr/total_loss"] = total.detach()
    pipe._tokenlight_loss_metrics = metrics
    return total


class TokenLightPbrSafeTrainingModule(base.TokenLightWanTrainingModule):
    """Base TokenLight module extended with safe joint PBR streams."""

    def __init__(
        self,
        *args,
        tokenlight_pbr_streams: Any = "depth,normal",
        tokenlight_pbr_stream_image_keys: Any = "depth:pbr_depth_image,normal:pbr_normal_image",
        tokenlight_pbr_stream_loss_weights: Any = "depth:0.1,normal:0.1",
        tokenlight_pbr_latent_loss_weight: float = 1.0,
        tokenlight_rgb_latent_loss_weight: float = 1.0,
        tokenlight_pbr_target_prob: float = 0.70,
        tokenlight_pbr_condition_prob: float = 0.18,
        tokenlight_pbr_source_drop_prob: float = 0.12,
        tokenlight_source_drop_rgb_loss_weight: float = 1.0,
        dataset_base_path: str | None = None,
        **kwargs,
    ) -> None:
        original = dict(kwargs)
        parent = dict(kwargs)
        parent["tokenlight_light_tokens"] = False
        parent["tokenlight_source_tokens"] = False
        parent["tokenlight_mask_tokens"] = False
        super().__init__(*args, **parent)

        streams = _csv_names(tokenlight_pbr_streams)
        if not streams:
            raise ValueError("At least one PBR stream is required")
        image_keys = _name_map(tokenlight_pbr_stream_image_keys, cast=str)
        loss_weights = _name_map(tokenlight_pbr_stream_loss_weights, cast=float)
        self.tokenlight_pbr_streams = streams
        self.tokenlight_pbr_stream_image_keys = {
            name: image_keys.get(name, f"pbr_{name}_image") for name in streams
        }
        self.tokenlight_pbr_stream_loss_weights = {
            name: float(loss_weights.get(name, 1.0)) for name in streams
        }
        self.tokenlight_pbr_latent_loss_weight = float(tokenlight_pbr_latent_loss_weight)
        self.tokenlight_rgb_latent_loss_weight = float(tokenlight_rgb_latent_loss_weight)
        self.tokenlight_pbr_target_prob = float(tokenlight_pbr_target_prob)
        self.tokenlight_pbr_condition_prob = float(tokenlight_pbr_condition_prob)
        self.tokenlight_pbr_source_drop_prob = float(tokenlight_pbr_source_drop_prob)
        self.tokenlight_source_drop_rgb_loss_weight = float(tokenlight_source_drop_rgb_loss_weight)
        self.dataset_base_path = dataset_base_path
        self.decoder_optimizer_step = 0

        self.tokenlight_attrs_key = str(original.get("tokenlight_attrs_key", "attrs_json"))
        self.tokenlight_light_tokens = bool(original.get("tokenlight_light_tokens", True))
        self.tokenlight_source_tokens = bool(original.get("tokenlight_source_tokens", True))
        self.tokenlight_mask_tokens = False
        self.tokenlight_cfg_drop_prob = float(original.get("tokenlight_cfg_drop_prob", 0.1))
        token_dim = int(original.get("tokenlight_token_dim", 0) or 0)
        if token_dim <= 0:
            token_dim = int(self.pipe.dit.dim)
        self.light_encoder = (
            LightokenEncoder(
                token_dim=token_dim,
                fourier_features=int(original.get("tokenlight_fourier_features", 512)),
                fourier_sigma=float(original.get("tokenlight_fourier_sigma", 5.0)),
                max_lights=int(original.get("tokenlight_max_lights", 2)),
                dropout=float(original.get("tokenlight_light_dropout", 0.0)),
            )
            if self.tokenlight_light_tokens
            else None
        )
        self.tokenlight_type_embedding = TokenLightPBRTypeEmbedding(
            token_dim,
            num_types=tokenlight_pbr_type_count(len(streams)),
        )
        checkpoint = base._load_checkpoint_state_dict(
            original.get("lora_checkpoint") or original.get("resume_from_checkpoint")
        )
        base._load_module_state_from_checkpoint(self.light_encoder, checkpoint, "light_encoder")
        type_state = base._extract_module_state(checkpoint, "tokenlight_type_embedding")
        source_weight = type_state.get("embedding.weight")
        if isinstance(source_weight, torch.Tensor) and tuple(source_weight.shape) == tuple(
            self.tokenlight_type_embedding.embedding.weight.shape
        ):
            self.tokenlight_type_embedding.load_state_dict(type_state, strict=False)
            print(
                "Loaded tokenlight_type_embedding from a matching PBR checkpoint: "
                f"shape={tuple(source_weight.shape)}"
            )
        elif isinstance(source_weight, torch.Tensor) and source_weight.ndim == 2 and int(
            source_weight.shape[0]
        ) == 4 and int(source_weight.shape[1]) == token_dim:
            # RGB TokenLight rows are [source, mask, light, target].  PBR rows
            # are [source, light, rgb_target, depth_target, depth_condition,
            # normal_target, normal_condition, ...].  Preserve only semantic
            # matches; new physical-stream rows keep their random init.
            with torch.no_grad():
                target_weight = self.tokenlight_type_embedding.embedding.weight
                target_weight[0].copy_(source_weight[0])
                target_weight[1].copy_(source_weight[2])
                target_weight[2].copy_(source_weight[3])
                for stream_index in range(len(streams)):
                    target_weight[3 + 2 * stream_index].copy_(source_weight[3])
            print(
                "Migrated legacy RGB type embedding into the PBR schema: "
                "source<-0, light<-2, rgb_target<-3, pbr_target_rows<-3; "
                "condition rows remain newly initialized"
            )
        elif source_weight is not None:
            raise ValueError(
                "Unsupported tokenlight_type_embedding checkpoint shape for PBR-safe training: "
                f"{tuple(source_weight.shape)}"
            )

        self.pipe.model_fn = self._tokenlight_pbr_model_fn
        self.task_to_loss["sft"] = self._pbr_sft_loss
        self.task_to_loss["sft:train"] = self._pbr_sft_loss

    def _tokenlight_pbr_model_fn(self, **kwargs):
        return model_fn_wan_video_tokenlight_pbr(
            tokenlight_light_encoder=self.light_encoder,
            tokenlight_type_embedding=self.tokenlight_type_embedding,
            **kwargs,
        )

    def _pbr_sft_loss(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        del inputs_nega
        merged = dict(inputs_shared)
        merged.update(inputs_posi)
        return TokenLightPbrSafeFlowMatchLoss(pipe, **merged)

    def _sample_unirelight_modes(self, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        total = (
            self.tokenlight_pbr_target_prob
            + self.tokenlight_pbr_condition_prob
            + self.tokenlight_pbr_source_drop_prob
        )
        draws = torch.rand((batch,), device=device)
        target_boundary = self.tokenlight_pbr_target_prob / total
        condition_boundary = (
            self.tokenlight_pbr_target_prob + self.tokenlight_pbr_condition_prob
        ) / total
        return draws < target_boundary, draws >= condition_boundary

    def _attach_pbr_latents(
        self,
        inputs_shared: dict[str, Any],
        pbr_latents: Mapping[str, torch.Tensor],
    ) -> dict[str, Any]:
        first = next(iter(pbr_latents.values()))
        batch = int(first.shape[0])
        target_mask, source_drop = self._sample_unirelight_modes(batch, first.device)
        inputs_shared["tokenlight_pbr_input_latents_map"] = dict(pbr_latents)
        inputs_shared["tokenlight_pbr_is_target_map"] = {
            name: target_mask for name in self.tokenlight_pbr_streams
        }
        inputs_shared["tokenlight_pbr_loss_weights"] = dict(self.tokenlight_pbr_stream_loss_weights)
        inputs_shared["tokenlight_pbr_latent_loss_weight"] = self.tokenlight_pbr_latent_loss_weight
        inputs_shared["tokenlight_rgb_latent_loss_weight"] = self.tokenlight_rgb_latent_loss_weight
        inputs_shared["tokenlight_source_drop_mask"] = source_drop
        rgb_weights = torch.ones((batch,), device=first.device, dtype=torch.float32)
        rgb_weights = torch.where(
            source_drop,
            torch.full_like(rgb_weights, self.tokenlight_source_drop_rgb_loss_weight),
            rgb_weights,
        )
        inputs_shared["tokenlight_rgb_sample_weights"] = rgb_weights
        if bool(source_drop.any()) and "tokenlight_source_latents" in inputs_shared:
            mask = source_drop.reshape(batch, 1, 1, 1, 1)
            inputs_shared["tokenlight_source_latents"] = torch.where(
                mask,
                torch.zeros_like(inputs_shared["tokenlight_source_latents"]),
                inputs_shared["tokenlight_source_latents"],
            )
        return inputs_shared

    @staticmethod
    def _batch_values(value: Any, batch: int, name: str) -> list[Any]:
        if batch == 1:
            return [value]
        if not isinstance(value, list) or len(value) != batch:
            raise ValueError(f"Expected {batch} samples for {name!r}")
        return value

    def _load_pbr_videos(self, value: Any, *, batch: int, name: str) -> list[list[Any]]:
        samples = self._batch_values(value, batch, name)
        videos: list[list[Any]] = []
        for sample in samples:
            frames = base._as_frames(sample)
            loaded = []
            for frame in frames:
                if isinstance(frame, (str, Path)):
                    path = Path(frame)
                    if not path.is_absolute():
                        path = Path(self.dataset_base_path or ".") / path
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing PBR image {name!r}: {path}")
                    with Image.open(path) as image:
                        loaded.append(image.convert("RGB").copy())
                else:
                    loaded.append(frame)
            videos.append(loaded)
        return videos

    def _pbr_latents_from_data(
        self,
        data: Mapping[str, Any],
        inputs_shared: dict[str, Any],
        batch: int,
    ) -> dict[str, torch.Tensor]:
        result = {}
        for stream_name in self.tokenlight_pbr_streams:
            image_key = self.tokenlight_pbr_stream_image_keys[stream_name]
            if image_key not in data:
                raise KeyError(f"Missing PBR metadata/data key {image_key!r}")
            videos = self._load_pbr_videos(data[image_key], batch=batch, name=image_key)
            result[stream_name] = self._encode_video_latents_batched(videos, inputs_shared)
        return result

    def _batched_inputs(self, data):
        inputs_shared, inputs_posi, inputs_nega = super()._batched_inputs(data)
        batch = self._batch_size_from_data(data)
        pbr_latents = self._pbr_latents_from_data(data, inputs_shared, batch)
        return self._attach_pbr_latents(inputs_shared, pbr_latents), inputs_posi, inputs_nega

    def _cached_batched_inputs(self, data):
        inputs_shared, inputs_posi, inputs_nega = super()._cached_batched_inputs(data)
        batch = self._batch_size_from_data(data)
        pbr_latents = self._pbr_latents_from_data(data, inputs_shared, batch)
        return self._attach_pbr_latents(inputs_shared, pbr_latents), inputs_posi, inputs_nega

    def get_pipeline_inputs(self, data):
        inputs_shared, inputs_posi, inputs_nega = super().get_pipeline_inputs(data)
        inputs_shared["tokenlight_pbr_images"] = {
            stream_name: data[self.tokenlight_pbr_stream_image_keys[stream_name]]
            for stream_name in self.tokenlight_pbr_streams
        }
        return inputs_shared, inputs_posi, inputs_nega

    def _prepare_tokenlight_inputs(self, inputs, data):
        inputs_shared, inputs_posi, inputs_nega = super()._prepare_tokenlight_inputs(inputs, data)
        images = inputs_shared.pop("tokenlight_pbr_images", None)
        if images:
            pbr_latents = {
                name: self._encode_image_latents(base._as_frames(image)[0], inputs_shared)
                for name, image in images.items()
            }
            inputs_shared = self._attach_pbr_latents(inputs_shared, pbr_latents)
        return inputs_shared, inputs_posi, inputs_nega


def build_pbr_dataset(args):
    dataset = base.build_dataset(args)
    rejected = safe._load_rejected_scene_ids(getattr(args, "dataset_reject_metadata_path", None))
    if rejected:
        before = len(dataset.data)
        dataset.data[:] = [row for row in dataset.data if not safe._row_is_rejected(row, rejected)]
        print(f"Reject-list filter: kept {len(dataset.data)} of {before} rows")

    required = ["video", "input_image", "attrs_json"] + list(args.tokenlight_pbr_stream_image_keys_map.values())
    missing = []
    for index, row in enumerate(dataset.data):
        absent = [key for key in required if row.get(key) in (None, "")]
        if absent:
            missing.append((index, absent))
            if len(missing) >= 10:
                break
    if missing:
        raise ValueError(f"PBR metadata is missing required keys (first 10): {missing}")
    if not dataset.data:
        raise ValueError("PBR dataset has no usable rows")
    return dataset


def _resolve_data_path(base_path: str | None, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(base_path or ".") / path
    return path


def validate_dataset_sample(args) -> None:
    """Read metadata/cache entries and fully decode a few PBR PNG samples."""

    dataset = build_pbr_dataset(args)
    count = min(len(dataset), max(1, int(args.dataset_validation_samples)))
    summaries = []
    for index in range(count):
        sample = dataset[index]
        parse_attrs_json(sample["attrs_json"])
        pbr_sizes = {}
        for stream_name, image_key in args.tokenlight_pbr_stream_image_keys_map.items():
            path = _resolve_data_path(args.dataset_base_path, sample[image_key])
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                pbr_sizes[stream_name] = [image.width, image.height, image.mode]
        latent = sample.get("_tokenlight_input_latents")
        summaries.append(
            {
                "index": index,
                "scene": sample.get("scene_id", sample.get("scene_folder")),
                "pbr": pbr_sizes,
                "cached_rgb_latent_shape": list(latent.shape) if isinstance(latent, torch.Tensor) else None,
            }
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "dataset_rows": len(dataset.data),
                "samples_checked": count,
                "streams": args.tokenlight_pbr_streams_list,
                "samples": summaries,
            },
            indent=2,
            sort_keys=True,
        )
    )


def pbr_safe_parser(train_mode: str) -> argparse.ArgumentParser:
    parser = safe.safe_parser(train_mode)
    parser.description = f"Safe TokenLight RGB + PBR joint-stream {train_mode} trainer."
    parser.add_argument("--tokenlight_pbr_streams", default="depth,normal")
    parser.add_argument(
        "--tokenlight_pbr_stream_image_keys",
        default="depth:pbr_depth_image,normal:pbr_normal_image",
    )
    parser.add_argument(
        "--tokenlight_pbr_stream_loss_weights",
        default="depth:0.1,normal:0.1",
    )
    parser.add_argument("--tokenlight_pbr_latent_loss_weight", type=float, default=1.0)
    parser.add_argument("--tokenlight_pbr_decoder_loss_weight", type=float, default=0.0)
    parser.add_argument("--tokenlight_pbr_target_prob", type=float, default=0.70)
    parser.add_argument("--tokenlight_pbr_condition_prob", type=float, default=0.18)
    parser.add_argument("--tokenlight_pbr_source_drop_prob", type=float, default=0.12)
    parser.add_argument("--tokenlight_source_drop_rgb_loss_weight", type=float, default=1.0)
    parser.add_argument("--validate_dataset_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dataset_validation_samples", type=int, default=2)
    parser.set_defaults(tokenlight_mask_tokens=False, tokenlight_rgb_decoder_loss_weight=0.0)
    return parser


def _validate_pbr_args(args, raw_config: Mapping[str, Any], parser: argparse.ArgumentParser) -> None:
    safe._validate_safe_args(args, raw_config, parser)
    if args.task not in {"sft", "sft:train"}:
        parser.error("PBR-safe training supports only sft and sft:train")
    if float(args.tokenlight_rgb_decoder_loss_weight) != 0.0:
        parser.error("PBR-safe training forbids RGB decoder loss")
    if float(args.tokenlight_pbr_decoder_loss_weight) != 0.0:
        parser.error("PBR-safe training forbids PBR decoder loss")
    if bool(args.tokenlight_mask_tokens):
        parser.error("This joint PBR trainer does not mix an additional mask-token stream")

    streams = _csv_names(args.tokenlight_pbr_streams)
    if not streams:
        parser.error("At least one PBR stream is required")
    image_keys = _name_map(args.tokenlight_pbr_stream_image_keys, cast=str)
    loss_weights = _name_map(args.tokenlight_pbr_stream_loss_weights, cast=float)
    missing_image_keys = [name for name in streams if name not in image_keys]
    if missing_image_keys:
        parser.error(f"Missing image keys for PBR streams: {missing_image_keys}")
    unknown_image_keys = sorted(set(image_keys) - set(streams))
    unknown_loss_weights = sorted(set(loss_weights) - set(streams))
    if unknown_image_keys or unknown_loss_weights:
        parser.error(
            f"Unknown PBR stream mappings: image_keys={unknown_image_keys}, "
            f"loss_weights={unknown_loss_weights}"
        )
    for name, weight in loss_weights.items():
        if not math.isfinite(weight) or weight < 0:
            parser.error(f"Invalid loss weight for PBR stream {name!r}: {weight}")
    if not math.isfinite(float(args.tokenlight_pbr_latent_loss_weight)) or float(
        args.tokenlight_pbr_latent_loss_weight
    ) < 0:
        parser.error("tokenlight_pbr_latent_loss_weight must be finite and non-negative")

    probabilities = [
        float(args.tokenlight_pbr_target_prob),
        float(args.tokenlight_pbr_condition_prob),
        float(args.tokenlight_pbr_source_drop_prob),
    ]
    if any(not math.isfinite(value) or value < 0 for value in probabilities) or sum(probabilities) <= 0:
        parser.error("PBR target/condition/source-drop probabilities must be non-negative with a positive sum")
    source_drop_rgb_weight = float(args.tokenlight_source_drop_rgb_loss_weight)
    if not math.isfinite(source_drop_rgb_weight) or source_drop_rgb_weight < 0:
        parser.error("tokenlight_source_drop_rgb_loss_weight must be finite and non-negative")
    if float(args.tokenlight_pbr_source_drop_prob) > 0 and source_drop_rgb_weight <= 0:
        parser.error(
            "source-drop samples need a positive RGB latent loss weight because this trainer has no decoder/aux loss"
        )
    if int(args.dataset_validation_samples) < 1:
        parser.error("dataset_validation_samples must be at least 1")

    known = base._parser_destinations(parser)
    unknown_pbr = sorted(key for key in raw_config.get("pbr", {}) if key not in known)
    if unknown_pbr:
        parser.error(f"unknown key(s) in pbr config: {', '.join(unknown_pbr)}")

    args.tokenlight_pbr_streams_list = streams
    args.tokenlight_pbr_stream_image_keys_map = {name: image_keys[name] for name in streams}
    args.tokenlight_pbr_stream_loss_weights_map = {
        name: float(loss_weights.get(name, 1.0)) for name in streams
    }


def parse_pbr_safe_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--train_mode", choices=("single", "zero3"), default="single")
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    config_path = pre_args.config or os.environ.get("TOKENLIGHT_TRAIN_CONFIG", DEFAULT_CONFIG_PATH)
    raw_config = base._load_json_config(config_path)
    parser = pbr_safe_parser(pre_args.train_mode)
    base._apply_config_defaults(parser, raw_config)
    parser.set_defaults(config=config_path, train_mode=pre_args.train_mode)
    args = parser.parse_args()
    raw_config = base._load_json_config(args.config)
    _validate_pbr_args(args, raw_config, parser)
    base._resolve_weight_paths(args)
    base._append_timestamp_to_output_path(args)
    return args, raw_config, raw_config


def main() -> None:
    args, raw_config, merged_config = parse_pbr_safe_args()
    if args.validate_dataset_only:
        validate_dataset_sample(args)
        return

    args.loss_impl_version = LOSS_IMPL_VERSION
    args.pbr_decoder_loss = "forbidden"
    args.pbr_type_embedding_layout = (
        "source,light,rgb_target," + ",".join(
            role
            for stream in args.tokenlight_pbr_streams_list
            for role in (f"{stream}_target", f"{stream}_condition")
        )
    )
    args.pbr_type_embedding_migration = (
        "legacy_rgb:[source0->source0,light2->light1,target3->rgb2_and_pbr_targets];"
        "pbr_condition_rows=new_init"
    )
    accelerator_kwargs = {
        "kwargs_handlers": [
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            )
        ]
    }
    if args.train_mode == "single":
        accelerator_kwargs["gradient_accumulation_steps"] = args.gradient_accumulation_steps
        accelerator_kwargs["mixed_precision"] = args.mixed_precision
    accelerator = accelerate.Accelerator(**accelerator_kwargs)
    base.save_training_config_snapshot(args, raw_config, merged_config, accelerator)

    dataset = build_pbr_dataset(args)
    model = TokenLightPbrSafeTrainingModule(
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
        device=(
            "cpu"
            if args.initialize_model_on_cpu or getattr(args, "enable_model_cpu_offload", False)
            else accelerator.device
        ),
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        tokenlight_light_tokens=args.tokenlight_light_tokens,
        tokenlight_source_tokens=args.tokenlight_source_tokens,
        tokenlight_mask_tokens=False,
        tokenlight_attrs_key=args.tokenlight_attrs_key,
        tokenlight_token_dim=args.tokenlight_token_dim,
        tokenlight_fourier_features=args.tokenlight_fourier_features,
        tokenlight_fourier_sigma=args.tokenlight_fourier_sigma,
        tokenlight_max_lights=args.tokenlight_max_lights,
        tokenlight_light_dropout=args.tokenlight_light_dropout,
        tokenlight_cfg_drop_prob=args.tokenlight_cfg_drop_prob,
        prompt_context_cache_size=args.prompt_context_cache_size,
        tokenlight_pbr_streams=args.tokenlight_pbr_streams,
        tokenlight_pbr_stream_image_keys=args.tokenlight_pbr_stream_image_keys,
        tokenlight_pbr_stream_loss_weights=args.tokenlight_pbr_stream_loss_weights,
        tokenlight_pbr_latent_loss_weight=args.tokenlight_pbr_latent_loss_weight,
        tokenlight_rgb_latent_loss_weight=args.tokenlight_rgb_latent_loss_weight,
        tokenlight_pbr_target_prob=args.tokenlight_pbr_target_prob,
        tokenlight_pbr_condition_prob=args.tokenlight_pbr_condition_prob,
        tokenlight_pbr_source_drop_prob=args.tokenlight_pbr_source_drop_prob,
        tokenlight_source_drop_rgb_loss_weight=args.tokenlight_source_drop_rgb_loss_weight,
        dataset_base_path=args.dataset_base_path,
    )
    model_logger = base._make_model_logger(args)
    physics_runtime.launch_physics_training_task(
        accelerator,
        dataset,
        model,
        model_logger,
        args=args,
    )


if __name__ == "__main__":
    main()
