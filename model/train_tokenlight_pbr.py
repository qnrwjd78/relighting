from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import accelerate
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Sampler
from tqdm import tqdm

from model.decoder_space_loss import decoder_space_loss
from model.illumination_latent_head import (
    illumination_latent_head_loss,
    load_illumination_head_checkpoint,
)
from model.lightoken_encoder import LightokenEncoder, attrs_from_batch
from model.tokenlight_wan_pbr import (
    TokenLightPBRTypeEmbedding,
    model_fn_wan_video_tokenlight_pbr,
    tokenlight_pbr_type_count,
)
from model.train_tokenlight import (
    DEFAULT_TRAIN_CONFIG_PATH,
    TRAIN_CONFIG_BY_MODE,
    TOKENLIGHT_DEFAULT_PROMPT,
    TokenLightWanTrainingModule,
    VaeLatentCache,
    _append_timestamp_to_output_path,
    _apply_config_defaults,
    _as_frames,
    _call_compatible_method,
    _collect_train_metrics,
    _coerce_trainable_parameter_dtype,
    _constructor_resume_checkpoint,
    _csv_value,
    _load_checkpoint_state_dict,
    _load_json_config,
    _mean_across_processes,
    _make_model_logger,
    _maybe_load_full_training_state,
    _normalize_tokenlight_metadata_rows,
    _preferred_trainable_dtype,
    _resolve_weight_paths,
    _require_nonempty_light_attrs,
    _resolve_distributed_world_size,
    _trainable_dtype_counts,
    _trainable_parameters,
    _collate_tokenlight_batch,
    build_dataset as build_tokenlight_dataset,
    get_optimizer_class,
    initialize_deepspeed_gradient_checkpointing,
    launch_data_process_task,
    save_training_config_snapshot,
    save_training_runtime_snapshot,
    wan_parser,
    OffloadTrainingManager,
    TokenLightTensorBoardMetrics,
)


PBR_CONFIG_BY_MODE = {
    "single": "configs/train_480/pbr.json",
    "zero3": "configs/train_480/pbr.json",
}


def _bool_mask(value: Any, *, batch: int, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        mask = value.to(device=device, dtype=torch.bool).flatten()
        return mask.expand(batch) if mask.numel() == 1 else mask
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return torch.tensor(list(value), device=device, dtype=torch.bool).flatten()
    return torch.full((batch,), bool(value), device=device, dtype=torch.bool)


def _rank0_broadcast_random_draws(batch: int, device: torch.device) -> torch.Tensor:
    if batch <= 0:
        return torch.empty((0,), device=device)
    dist = torch.distributed
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return torch.rand((batch,), device=device)

    local_batch = torch.tensor([batch], device=device, dtype=torch.int64)
    max_batch = local_batch.clone()
    dist.all_reduce(max_batch, op=dist.ReduceOp.MAX)

    draws = torch.empty((int(max_batch.item()),), device=device, dtype=torch.float32)
    if dist.get_rank() == 0:
        draws.uniform_()
    dist.broadcast(draws, src=0)
    return draws[:batch]


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=pred.device, dtype=torch.float32).view(-1, 1, 1, 1, 1)
    error = (pred.float() - target.float()).pow(2) * mask
    denom = mask.sum() * math.prod(pred.shape[1:])
    return error.sum() / denom.clamp_min(1.0)


def _weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: Any = None) -> torch.Tensor:
    if weights is None:
        return F.mse_loss(pred.float(), target.float())
    weights = torch.as_tensor(weights, device=pred.device, dtype=torch.float32).flatten()
    if weights.numel() == 1 and bool(torch.isclose(weights[0], torch.ones_like(weights[0]))):
        return F.mse_loss(pred.float(), target.float())
    weights = weights.view(-1, *([1] * (pred.ndim - 1)))
    error = (pred.float() - target.float()).pow(2) * weights
    denom = weights.sum() * math.prod(pred.shape[1:])
    return error.sum() / denom.clamp_min(1.0)


def _decode_latents(pipe, latents: torch.Tensor, inputs: dict[str, Any]) -> torch.Tensor:
    pipe.load_models_to_device(["vae"])
    return pipe.vae.decode(
        latents.to(dtype=pipe.torch_dtype, device=pipe.device),
        device=pipe.device,
        tiled=bool(inputs.get("tokenlight_log_luminance_decode_tiled", False)),
        tile_size=inputs.get("tile_size", (30, 52)),
        tile_stride=inputs.get("tile_stride", (15, 26)),
    )


def _log_luminance(video: torch.Tensor, eps: float) -> torch.Tensor:
    rgb = ((video.float() + 1.0) * 0.5).clamp_min(float(eps))
    weights = torch.tensor([0.2126, 0.7152, 0.0722], device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1, 1)
    luminance = (rgb[:, :3] * weights).sum(dim=1, keepdim=True).clamp_min(float(eps))
    return torch.log(luminance)


def _source_drop_log_luminance_loss(
    pipe,
    *,
    pred: torch.Tensor,
    noise: torch.Tensor,
    target_log_luminance: torch.Tensor | None,
    mask: Any,
    inputs: dict[str, Any],
) -> torch.Tensor:
    mask = _bool_mask(mask, batch=int(pred.shape[0]), device=pred.device)
    if not bool(mask.any()):
        return pred.new_zeros(())
    if target_log_luminance is None:
        return pred.new_zeros(())

    pred_x0 = noise - pred
    pred_video = _decode_latents(pipe, pred_x0[mask], inputs)
    eps = float(inputs.get("tokenlight_log_luminance_eps", 1e-3))
    pred_log = _log_luminance(pred_video, eps)
    target_mask = mask.to(device=pred_log.device)
    target_log = target_log_luminance.to(device=pred_log.device, dtype=pred_log.dtype)[target_mask]
    return F.mse_loss(pred_log, target_log).to(device=pred.device)


def _source_drop_illumination_loss(
    *,
    pred: torch.Tensor,
    noise: torch.Tensor,
    target_illum_latents: torch.Tensor | None,
    mask: Any,
    inputs: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    head = inputs.get("tokenlight_source_drop_illum_head")
    if head is None or target_illum_latents is None:
        return pred.new_zeros(()), {}
    mask = _bool_mask(mask, batch=int(pred.shape[0]), device=pred.device)
    if not bool(mask.any()):
        return pred.new_zeros(()), {}

    try:
        head_param = next(head.parameters())
        head_device = head_param.device
        head_dtype = head_param.dtype
    except StopIteration:
        head_device = pred.device
        head_dtype = pred.dtype

    pred_x0 = noise - pred
    pred_illum = head(pred_x0[mask].to(device=head_device, dtype=head_dtype))
    target = target_illum_latents.to(device=pred_illum.device, dtype=pred_illum.dtype)[mask]
    if tuple(pred_illum.shape) != tuple(target.shape):
        raise ValueError(
            "Source-drop illumination target shape mismatch: "
            f"pred={tuple(pred_illum.shape)} target={tuple(target.shape)}. "
            "Build the illumination cache at the same resolution as training."
        )

    config = inputs.get("tokenlight_source_drop_illum_head_config")
    mean = inputs.get("tokenlight_source_drop_illum_latent_mean")
    std = inputs.get("tokenlight_source_drop_illum_latent_std")
    loss, metrics = illumination_latent_head_loss(
        pred_illum,
        target,
        multiscale_weights=getattr(config, "multiscale_weights", (1.0, 0.5, 0.25)),
        cosine_weight=float(getattr(config, "cosine_weight", 0.1)),
        mean=mean,
        std=std,
        normalize=bool(inputs.get("tokenlight_source_drop_illum_latent_normalize", True)),
        norm_eps=float(inputs.get("tokenlight_source_drop_illum_latent_norm_eps", 1e-6)),
        loss_type=getattr(config, "loss_type", "mse"),
    )
    return loss.to(device=pred.device), metrics


def _parse_pbr_streams(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text[0] in "[{":
            parsed = json.loads(text)
            items = parsed.keys() if isinstance(parsed, Mapping) else parsed
        else:
            items = text.split(",")
    elif isinstance(value, Mapping):
        items = value.keys()
    else:
        items = value
    streams = []
    for item in items:
        name = str(item).strip()
        if name:
            streams.append(name)
    if len(set(streams)) != len(streams):
        raise ValueError(f"Duplicate PBR stream names: {streams}")
    return streams


def _parse_pbr_stream_map(value: Any, *, cast=str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        raw = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        if text[0] == "{":
            parsed = json.loads(text)
            if not isinstance(parsed, Mapping):
                raise ValueError(f"Expected a mapping for PBR stream map, got {type(parsed).__name__}")
            raw = parsed
        else:
            raw = {}
            for item in text.split(","):
                item = item.strip()
                if not item:
                    continue
                if ":" in item:
                    key, val = item.split(":", 1)
                elif "=" in item:
                    key, val = item.split("=", 1)
                else:
                    raise ValueError(f"Expected `stream:value` in PBR stream map item `{item}`")
                raw[key.strip()] = val.strip()
    else:
        raw = dict(value)
    result = {}
    for key, val in raw.items():
        name = str(key).strip()
        if name:
            result[name] = cast(val)
    return result


def _floating_module_dtype_counts(module: torch.nn.Module | None) -> dict[str, int]:
    if module is None:
        return {}
    counts: dict[str, int] = {}
    for parameter in module.parameters():
        if parameter.is_floating_point():
            counts[str(parameter.dtype)] = counts.get(str(parameter.dtype), 0) + int(parameter.numel())
    return counts


def _coerce_floating_module_dtype(module: torch.nn.Module | None, dtype: torch.dtype) -> dict[str, int]:
    before = _floating_module_dtype_counts(module)
    if module is None:
        return before
    for parameter in module.parameters():
        if parameter.is_floating_point() and parameter.dtype != dtype:
            parameter.data = parameter.data.to(dtype=dtype)
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.data.to(dtype=dtype)
    return before


def _pbr_latent_items(
    stream_latents: Mapping[str, torch.Tensor] | Sequence[tuple[str, torch.Tensor]] | None,
    legacy_latents: torch.Tensor | None,
) -> tuple[list[tuple[str, torch.Tensor]], bool]:
    if stream_latents is None:
        return ([] if legacy_latents is None else [("pbr", legacy_latents)]), legacy_latents is not None
    if isinstance(stream_latents, Mapping):
        items = [(str(name), latents) for name, latents in stream_latents.items()]
    else:
        items = [(str(name), latents) for name, latents in stream_latents]
    names = [name for name, _ in items]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate PBR latent stream names: {names}")
    return items, False


def _pbr_stream_value(stream_map: Any, name: str, default: Any) -> Any:
    if isinstance(stream_map, Mapping):
        return stream_map.get(name, default)
    return default


def _pbr_stream_loss_weight(inputs: dict[str, Any], name: str) -> float:
    weights = inputs.get("tokenlight_pbr_loss_weights")
    if isinstance(weights, Mapping):
        return float(weights.get(name, inputs.get("tokenlight_pbr_loss_weight", 1.0)))
    return float(inputs.get("tokenlight_pbr_loss_weight", 1.0))


def PBRFlowMatchSFTLoss(pipe, **inputs):
    if "lora" in inputs:
        pipe.clear_lora(verbose=0)
        pipe.load_lora(pipe.dit, state_dict=inputs["lora"], hotload=True, verbose=0)

    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    rgb_input = inputs["input_latents"]
    rgb_noise = torch.randn_like(rgb_input) * inputs.get("noise_scale", 1.0)
    inputs["latents"] = pipe.scheduler.add_noise(rgb_input, rgb_noise, timestep)
    rgb_target = pipe.scheduler.training_target(rgb_input, rgb_noise, timestep)

    pbr_targets: dict[str, torch.Tensor] = {}
    pbr_inputs: dict[str, torch.Tensor] = {}
    pbr_noises: dict[str, torch.Tensor] = {}
    pbr_masks: dict[str, torch.Tensor] = {}
    pbr_latents: dict[str, torch.Tensor] = {}
    pbr_items, legacy_pbr_input = _pbr_latent_items(
        inputs.get("tokenlight_pbr_input_latents_map"),
        inputs.get("tokenlight_pbr_input_latents"),
    )
    for stream_name, pbr_input in pbr_items:
        batch = int(pbr_input.shape[0])
        pbr_mask = _bool_mask(
            _pbr_stream_value(
                inputs.get("tokenlight_pbr_is_target_map"),
                stream_name,
                inputs.get("tokenlight_pbr_is_target", True),
            ),
            batch=batch,
            device=pbr_input.device,
        )
        pbr_noise = torch.randn_like(pbr_input) * inputs.get("noise_scale", 1.0)
        noisy_pbr = pipe.scheduler.add_noise(pbr_input, pbr_noise, timestep)
        pbr_targets[stream_name] = pipe.scheduler.training_target(pbr_input, pbr_noise, timestep)
        pbr_inputs[stream_name] = pbr_input
        pbr_noises[stream_name] = pbr_noise
        pbr_masks[stream_name] = pbr_mask
        mask = pbr_mask.view(-1, 1, 1, 1, 1)
        pbr_latents[stream_name] = torch.where(mask, noisy_pbr, pbr_input)
    if pbr_latents:
        if legacy_pbr_input:
            inputs["tokenlight_pbr_latents"] = pbr_latents["pbr"]
            inputs["tokenlight_pbr_is_target"] = pbr_masks["pbr"]
        else:
            inputs["tokenlight_pbr_stream_latents"] = pbr_latents
            inputs["tokenlight_pbr_stream_is_target"] = pbr_masks

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    if isinstance(noise_pred, tuple):
        rgb_pred, pbr_pred = noise_pred
    else:
        rgb_pred, pbr_pred = noise_pred, None

    log_luminance_target = inputs.get("tokenlight_log_luminance_target")
    source_drop_log_luminance_mask = inputs.get("tokenlight_source_drop_log_luminance_mask")
    source_drop_illum_target = inputs.get("tokenlight_source_drop_illum_target_latents")

    if "first_frame_latents" in inputs:
        rgb_pred = rgb_pred[:, :, 1:]
        rgb_target = rgb_target[:, :, 1:]
        rgb_clean_for_reconstruction = rgb_input[:, :, 1:]
        rgb_noise_for_reconstruction = rgb_noise[:, :, 1:]
        if isinstance(log_luminance_target, torch.Tensor):
            log_luminance_target = log_luminance_target[:, :, 1:]
        if isinstance(source_drop_illum_target, torch.Tensor):
            source_drop_illum_target = source_drop_illum_target[:, :, 1:]
    else:
        rgb_noise_for_reconstruction = rgb_noise
        rgb_clean_for_reconstruction = rgb_input

    rgb_latent_raw = _weighted_mse(rgb_pred, rgb_target, inputs.get("tokenlight_rgb_loss_weight"))
    rgb_latent_weight = float(inputs.get("tokenlight_rgb_latent_loss_weight", 1.0))
    rgb_loss = rgb_latent_weight * rgb_latent_raw
    loss = rgb_loss
    loss_metrics = {
        "train/pbr/rgb_loss": rgb_loss.detach(),
        "train/pbr/rgb_latent_raw_mse": rgb_latent_raw.detach(),
    }
    rgb_decoder_weight = float(inputs.get("tokenlight_rgb_decoder_loss_weight", 0.0))
    if rgb_decoder_weight != 0.0:
        rgb_decoder_raw = decoder_space_loss(
            pipe,
            pred_x0_latents=rgb_noise_for_reconstruction - rgb_pred,
            target_video=inputs.get("tokenlight_rgb_decoder_target_video"),
            target_latents=(
                None
                if inputs.get("tokenlight_rgb_decoder_target_video") is not None
                else rgb_clean_for_reconstruction
            ),
            transform=inputs.get("tokenlight_rgb_decoder_transform", "rgb"),
            loss_type=inputs.get("tokenlight_decoder_loss_type", "mse"),
            sample_weights=inputs.get("tokenlight_rgb_loss_weight"),
            tiled=bool(inputs.get("tokenlight_decoder_tiled", False)),
            tile_size=inputs.get("tile_size", (30, 52)),
            tile_stride=inputs.get("tile_stride", (15, 26)),
            luminance_eps=float(inputs.get("tokenlight_decoder_luminance_eps", 1e-3)),
            charbonnier_eps=float(inputs.get("tokenlight_decoder_charbonnier_eps", 1e-3)),
        )
        rgb_decoder_loss = rgb_decoder_weight * rgb_decoder_raw
        loss = loss + rgb_decoder_loss
        loss_metrics["train/pbr/rgb_decoder_loss"] = rgb_decoder_loss.detach()
        loss_metrics["train/pbr/rgb_decoder_raw"] = rgb_decoder_raw.detach()
    if pbr_pred is not None and pbr_targets:
        pbr_pred_map = pbr_pred if isinstance(pbr_pred, Mapping) else {"pbr": pbr_pred}
        pbr_latent_weight = float(inputs.get("tokenlight_pbr_latent_loss_weight", 1.0))
        pbr_decoder_weight = float(inputs.get("tokenlight_pbr_decoder_loss_weight", 0.0))
        decoder_transforms = inputs.get("tokenlight_pbr_decoder_transforms", {})
        for stream_name, pbr_target in pbr_targets.items():
            stream_pred = pbr_pred_map.get(stream_name)
            if stream_pred is None:
                raise ValueError(f"Model did not return prediction for PBR stream `{stream_name}`")
            pbr_raw_loss = _masked_mse(
                stream_pred,
                pbr_target,
                pbr_masks[stream_name],
            )
            stream_weight = _pbr_stream_loss_weight(inputs, stream_name)
            pbr_weighted_loss = pbr_latent_weight * stream_weight * pbr_raw_loss
            loss = loss + pbr_weighted_loss
            loss_metrics[f"train/pbr/{stream_name}_latent_loss"] = pbr_weighted_loss.detach()
            loss_metrics[f"train/pbr/{stream_name}_loss"] = pbr_weighted_loss.detach()
            loss_metrics[f"train/pbr/{stream_name}_raw_mse"] = pbr_raw_loss.detach()
            if pbr_decoder_weight != 0.0:
                stream_mask = pbr_masks[stream_name]
                if bool(stream_mask.any()):
                    transform = (
                        decoder_transforms.get(stream_name, "rgb")
                        if isinstance(decoder_transforms, Mapping)
                        else "rgb"
                    )
                    decoder_target_videos = inputs.get("tokenlight_pbr_decoder_target_videos", {})
                    raw_target_video = decoder_target_videos.get(stream_name)
                    if raw_target_video is not None:
                        raw_target_video = raw_target_video[
                            stream_mask.to(device=raw_target_video.device)
                        ]
                    decoder_raw = decoder_space_loss(
                        pipe,
                        pred_x0_latents=(pbr_noises[stream_name] - stream_pred)[stream_mask],
                        target_video=raw_target_video,
                        target_latents=(
                            None
                            if raw_target_video is not None
                            else pbr_inputs[stream_name][stream_mask]
                        ),
                        transform=transform,
                        loss_type=inputs.get("tokenlight_decoder_loss_type", "mse"),
                        tiled=bool(inputs.get("tokenlight_decoder_tiled", False)),
                        tile_size=inputs.get("tile_size", (30, 52)),
                        tile_stride=inputs.get("tile_stride", (15, 26)),
                        luminance_eps=float(inputs.get("tokenlight_decoder_luminance_eps", 1e-3)),
                        charbonnier_eps=float(inputs.get("tokenlight_decoder_charbonnier_eps", 1e-3)),
                    )
                    decoder_weighted = pbr_decoder_weight * stream_weight * decoder_raw
                    loss = loss + decoder_weighted
                    loss_metrics[f"train/pbr/{stream_name}_decoder_loss"] = decoder_weighted.detach()
                    loss_metrics[f"train/pbr/{stream_name}_decoder_raw"] = decoder_raw.detach()
    if source_drop_log_luminance_mask is not None:
        log_luminance_weight = float(inputs.get("tokenlight_source_drop_log_luminance_loss_weight", 0.0))
        if log_luminance_weight > 0:
            log_luminance_loss = _source_drop_log_luminance_loss(
                pipe,
                pred=rgb_pred,
                noise=rgb_noise_for_reconstruction,
                target_log_luminance=log_luminance_target,
                mask=source_drop_log_luminance_mask,
                inputs=inputs,
            )
            log_luminance_weighted_loss = log_luminance_weight * log_luminance_loss
            loss = loss + log_luminance_weighted_loss
            loss_metrics["train/pbr/log_luminance_loss"] = log_luminance_weighted_loss.detach()
            loss_metrics["train/pbr/log_luminance_raw_mse"] = log_luminance_loss.detach()
        illum_weight = float(inputs.get("tokenlight_source_drop_illum_loss_weight", 0.0))
        if illum_weight > 0 and inputs.get("tokenlight_source_drop_illum_head") is not None:
            illum_loss, illum_metrics = _source_drop_illumination_loss(
                pred=rgb_pred,
                noise=rgb_noise_for_reconstruction,
                target_illum_latents=source_drop_illum_target,
                mask=source_drop_log_luminance_mask,
                inputs=inputs,
            )
            illum_weighted_loss = illum_weight * illum_loss
            loss = loss + illum_weighted_loss
            loss_metrics["train/pbr/illum_loss"] = illum_weighted_loss.detach()
            loss_metrics["train/pbr/illum_raw_loss"] = illum_loss.detach()
            for key, value in illum_metrics.items():
                loss_metrics[f"train/pbr/illum_{key}"] = value.detach()

    training_weight = pipe.scheduler.training_weight(timestep)
    if not isinstance(training_weight, torch.Tensor):
        training_weight = loss.new_tensor(float(training_weight))
    weighted_total = loss * training_weight
    loss_metrics["train/pbr/total_loss_before_scheduler_weight"] = loss.detach()
    loss_metrics["train/pbr/total_loss"] = weighted_total.detach()
    loss_metrics["train/pbr/scheduler_training_weight"] = training_weight.detach()
    pipe._tokenlight_loss_metrics = loss_metrics
    return weighted_total


def _load_matching_module_state(module: torch.nn.Module | None, state_dict: dict[str, torch.Tensor], prefix: str) -> None:
    if module is None or not state_dict:
        return
    module_state = module.state_dict()
    loaded = {}
    for key, value in state_dict.items():
        candidates = (f"{prefix}.", f"module.{prefix}.")
        for candidate in candidates:
            if not key.startswith(candidate):
                continue
            name = key[len(candidate) :]
            if name in module_state and tuple(module_state[name].shape) == tuple(value.shape):
                loaded[name] = value
            break
    if loaded:
        missing, unexpected = module.load_state_dict(loaded, strict=False)
        print(f"Loaded matching {prefix}: tensors={len(loaded)}, missing={len(missing)}, unexpected={len(unexpected)}")


class TokenLightPBRWanTrainingModule(TokenLightWanTrainingModule):
    def __init__(
        self,
        *args,
        tokenlight_pbr_image_key: str = "pbr_image",
        tokenlight_pbr_mode_key: str = "pbr_mode",
        tokenlight_pbr_aux_type: str = "shading",
        tokenlight_pbr_loss_weight: float = 1.0,
        tokenlight_pbr_streams: Any = None,
        tokenlight_pbr_stream_image_keys: Any = None,
        tokenlight_pbr_stream_loss_weights: Any = None,
        tokenlight_pbr_latent_loss_weight: float = 1.0,
        tokenlight_pbr_decoder_loss_weight: float = 0.0,
        tokenlight_pbr_decoder_transforms: Any = None,
        tokenlight_pbr_default_mode: str = "target",
        tokenlight_pbr_conditioning_strategy: str = "metadata",
        tokenlight_pbr_unirelight_target_prob: float = 0.70,
        tokenlight_pbr_unirelight_condition_prob: float = 0.18,
        tokenlight_pbr_unirelight_source_drop_prob: float = 0.12,
        tokenlight_source_drop_rgb_loss_weight: float = 0.0,
        tokenlight_source_drop_log_luminance_loss_weight: float = 1.0,
        tokenlight_source_drop_illum_loss_weight: float = 1.0,
        tokenlight_source_drop_illum_head_path: str | None = None,
        tokenlight_source_drop_illum_cache_dir: str | None = None,
        tokenlight_source_drop_illum_cache_key: str = "video",
        tokenlight_source_drop_illum_cache_open_shards: int = 16,
        tokenlight_source_drop_illum_latent_normalize: bool = True,
        tokenlight_source_drop_illum_latent_norm_eps: float = 1e-6,
        tokenlight_log_luminance_eps: float = 1e-3,
        tokenlight_log_luminance_decode_tiled: bool = False,
        **kwargs,
    ) -> None:
        super_kwargs = dict(kwargs)
        super_kwargs["tokenlight_light_tokens"] = False
        super_kwargs["tokenlight_source_tokens"] = False
        super_kwargs["tokenlight_mask_tokens"] = False
        super().__init__(*args, **super_kwargs)
        self.tokenlight_pbr_image_key = tokenlight_pbr_image_key
        self.tokenlight_pbr_mode_key = tokenlight_pbr_mode_key
        self.tokenlight_pbr_aux_type = tokenlight_pbr_aux_type
        self.tokenlight_pbr_loss_weight = float(tokenlight_pbr_loss_weight)
        self.tokenlight_pbr_default_mode = tokenlight_pbr_default_mode
        pbr_streams = _parse_pbr_streams(tokenlight_pbr_streams)
        self.tokenlight_pbr_legacy_stream = not pbr_streams
        if self.tokenlight_pbr_legacy_stream:
            pbr_streams = ["pbr"]
        image_key_map = _parse_pbr_stream_map(tokenlight_pbr_stream_image_keys, cast=str)
        loss_weight_map = _parse_pbr_stream_map(tokenlight_pbr_stream_loss_weights, cast=float)
        self.tokenlight_pbr_streams = pbr_streams
        self.tokenlight_pbr_stream_image_keys = {
            name: image_key_map.get(
                name,
                self.tokenlight_pbr_image_key if name == "pbr" else f"pbr_{name}_image",
            )
            for name in pbr_streams
        }
        self.tokenlight_pbr_stream_loss_weights = {
            name: float(loss_weight_map.get(name, self.tokenlight_pbr_loss_weight)) for name in pbr_streams
        }
        self.tokenlight_pbr_latent_loss_weight = float(tokenlight_pbr_latent_loss_weight)
        self.tokenlight_pbr_decoder_loss_weight = float(tokenlight_pbr_decoder_loss_weight)
        decoder_transform_map = _parse_pbr_stream_map(tokenlight_pbr_decoder_transforms, cast=str)
        self.tokenlight_pbr_decoder_transforms = {
            name: str(decoder_transform_map.get(name, "rgb")) for name in pbr_streams
        }
        self.tokenlight_pbr_conditioning_strategy = str(tokenlight_pbr_conditioning_strategy)
        self.tokenlight_pbr_unirelight_target_prob = float(tokenlight_pbr_unirelight_target_prob)
        self.tokenlight_pbr_unirelight_condition_prob = float(tokenlight_pbr_unirelight_condition_prob)
        self.tokenlight_pbr_unirelight_source_drop_prob = float(tokenlight_pbr_unirelight_source_drop_prob)
        self.tokenlight_source_drop_rgb_loss_weight = float(tokenlight_source_drop_rgb_loss_weight)
        self.tokenlight_source_drop_log_luminance_loss_weight = float(tokenlight_source_drop_log_luminance_loss_weight)
        self.tokenlight_source_drop_illum_loss_weight = float(tokenlight_source_drop_illum_loss_weight)
        self.tokenlight_source_drop_illum_cache_key = str(tokenlight_source_drop_illum_cache_key or "video")
        self.tokenlight_source_drop_illum_latent_normalize = bool(tokenlight_source_drop_illum_latent_normalize)
        self.tokenlight_source_drop_illum_latent_norm_eps = float(tokenlight_source_drop_illum_latent_norm_eps)
        self.tokenlight_log_luminance_eps = float(tokenlight_log_luminance_eps)
        self.tokenlight_log_luminance_decode_tiled = bool(tokenlight_log_luminance_decode_tiled)
        self.tokenlight_attrs_key = kwargs.get("tokenlight_attrs_key", "attrs_json")
        self.tokenlight_light_tokens = bool(kwargs.get("tokenlight_light_tokens", True))
        self.tokenlight_source_tokens = bool(kwargs.get("tokenlight_source_tokens", True))
        self.tokenlight_mask_tokens = False
        self.tokenlight_cfg_drop_prob = float(kwargs.get("tokenlight_cfg_drop_prob", 0.1))

        token_dim = int(kwargs.get("tokenlight_token_dim", 0) or 0)
        if token_dim <= 0:
            token_dim = int(self.pipe.dit.dim)
        self.light_encoder = (
            LightokenEncoder(
                token_dim=token_dim,
                fourier_features=int(kwargs.get("tokenlight_fourier_features", 512)),
                fourier_sigma=float(kwargs.get("tokenlight_fourier_sigma", 5.0)),
                max_lights=int(kwargs.get("tokenlight_max_lights", 2)),
                dropout=float(kwargs.get("tokenlight_light_dropout", 0.0)),
            )
            if self.tokenlight_light_tokens
            else None
        )
        self.tokenlight_type_embedding = TokenLightPBRTypeEmbedding(
            token_dim,
            num_types=tokenlight_pbr_type_count(len(self.tokenlight_pbr_streams)),
        )
        self.source_drop_illum_head = None
        self.source_drop_illum_head_config = None
        self.source_drop_illum_latent_mean = None
        self.source_drop_illum_latent_std = None
        if tokenlight_source_drop_illum_head_path not in (None, "", "None", "none", "null"):
            head, head_config, head_extra = load_illumination_head_checkpoint(
                tokenlight_source_drop_illum_head_path,
                map_location="cpu",
            )
            head.eval()
            for param in head.parameters():
                param.requires_grad_(False)
            head_dtype = getattr(self.pipe, "torch_dtype", torch.bfloat16)
            if isinstance(head_dtype, torch.dtype) and head_dtype.is_floating_point:
                _coerce_floating_module_dtype(head, head_dtype)
            self.source_drop_illum_head = head
            self.source_drop_illum_head_config = head_config
            self.source_drop_illum_latent_mean = head_extra.get("latent_mean")
            self.source_drop_illum_latent_std = head_extra.get("latent_std")
            print(f"Loaded source-drop illumination head: {tokenlight_source_drop_illum_head_path}")
        self.source_drop_illum_cache = None
        if tokenlight_source_drop_illum_cache_dir not in (None, "", "None", "none", "null"):
            self.source_drop_illum_cache = VaeLatentCache(
                [str(tokenlight_source_drop_illum_cache_dir)],
                max_open_shards=int(tokenlight_source_drop_illum_cache_open_shards),
            )
        checkpoint_state = _load_checkpoint_state_dict(kwargs.get("lora_checkpoint") or kwargs.get("resume_from_checkpoint"))
        _load_matching_module_state(self.light_encoder, checkpoint_state, "light_encoder")
        _load_matching_module_state(self.tokenlight_type_embedding, checkpoint_state, "tokenlight_type_embedding")

        self.pipe.model_fn = self._tokenlight_model_fn
        self.task_to_loss["sft"] = lambda pipe, inputs_shared, inputs_posi, inputs_nega: PBRFlowMatchSFTLoss(
            pipe, **inputs_shared, **inputs_posi
        )
        self.task_to_loss["sft:train"] = self.task_to_loss["sft"]

    def _tokenlight_model_fn(self, **kwargs):
        return model_fn_wan_video_tokenlight_pbr(
            tokenlight_light_encoder=self.light_encoder,
            tokenlight_type_embedding=self.tokenlight_type_embedding,
            **kwargs,
        )

    def _pbr_target_mask_from_data(self, data, batch: int, device: torch.device) -> torch.Tensor:
        value = data.get(self.tokenlight_pbr_mode_key, data.get("depth_mode", self.tokenlight_pbr_default_mode))
        if isinstance(value, list):
            modes = value
        else:
            modes = [value] * batch
        target_values = []
        for item in modes:
            text = str(item or self.tokenlight_pbr_default_mode).lower()
            target_values.append(text not in {"condition", "cond", "input", "clean", "provided"})
        return torch.tensor(target_values, device=device, dtype=torch.bool)

    def _pbr_videos_from_data(self, data, batch: int, stream_name: str = "pbr"):
        pbr_key = self.tokenlight_pbr_stream_image_keys[stream_name]
        pbr = data.get(pbr_key)
        if not isinstance(pbr, list) or len(pbr) != batch:
            raise ValueError(f"Expected {batch} `{pbr_key}` images for PBR stream `{stream_name}`")
        return [_as_frames(item) for item in pbr]

    def _cached_pbr_latents_from_data(self, data, batch: int) -> dict[str, torch.Tensor]:
        pbr_latents = {}
        for stream_name in self.tokenlight_pbr_streams:
            pbr_key = self.tokenlight_pbr_stream_image_keys[stream_name]
            latent_key = f"_tokenlight_{pbr_key}_latents"
            if latent_key not in data:
                raise KeyError(
                    f"Cached PBR stream `{stream_name}` is missing `{latent_key}`. "
                    f"Build the VAE cache with image key `{pbr_key}` and pass it in --data_file_keys."
                )
            pbr_latents[stream_name] = self._stack_cached_latents(
                data.get(latent_key),
                batch=batch,
                name=f"pbr_{stream_name}",
            )
        return pbr_latents

    def _cached_target_videos_from_paths(self, data, batch: int):
        videos = data.get("video")
        if isinstance(videos, str):
            videos = [videos]
        if not isinstance(videos, list) or len(videos) != batch:
            raise ValueError(f"Expected {batch} cached target image paths for log-luminance loss")
        base_path = self.dataset_base_path or Path(".")
        result = []
        for item in videos:
            if not isinstance(item, str):
                result.append(_as_frames(item))
                continue
            path = Path(item)
            if not path.is_absolute():
                path = base_path / path
            with Image.open(path) as image:
                result.append([image.convert("RGB")])
        return result

    def _source_drop_illum_latents_from_cache(self, data, batch: int) -> torch.Tensor | None:
        direct = data.get("_tokenlight_source_drop_illum_latents")
        if direct is not None:
            return self._stack_cached_latents(direct, batch=batch, name="illum_target")
        if self.source_drop_illum_cache is None:
            return None
        values = data.get(self.tokenlight_source_drop_illum_cache_key)
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list) or len(values) != batch:
            raise ValueError(
                f"Expected {batch} `{self.tokenlight_source_drop_illum_cache_key}` paths "
                "for source-drop illumination cache"
            )
        tensors = [
            self.source_drop_illum_cache.load(str(value), dataset_base_path=self.dataset_base_path)
            for value in values
        ]
        return torch.stack(tensors, dim=0)

    def _attach_pbr_latents_to_inputs(
        self,
        inputs_shared: dict[str, Any],
        data,
        batch: int,
        pbr_latents: dict[str, torch.Tensor],
        *,
        cached: bool,
    ):
        pbr_device = next(iter(pbr_latents.values())).device
        pbr_target, source_drop = self._sample_conditioning_modes(
            data,
            batch,
            pbr_device,
        )
        if self.tokenlight_pbr_legacy_stream:
            inputs_shared["tokenlight_pbr_input_latents"] = pbr_latents["pbr"]
            inputs_shared["tokenlight_pbr_is_target"] = pbr_target
            inputs_shared["tokenlight_pbr_loss_weight"] = self.tokenlight_pbr_stream_loss_weights["pbr"]
        else:
            inputs_shared["tokenlight_pbr_input_latents_map"] = pbr_latents
            inputs_shared["tokenlight_pbr_is_target_map"] = {
                stream_name: pbr_target for stream_name in self.tokenlight_pbr_streams
            }
            inputs_shared["tokenlight_pbr_loss_weights"] = dict(self.tokenlight_pbr_stream_loss_weights)
        inputs_shared["tokenlight_source_drop_log_luminance_mask"] = source_drop
        inputs_shared["tokenlight_source_drop_log_luminance_loss_weight"] = (
            self.tokenlight_source_drop_log_luminance_loss_weight
        )
        if self.source_drop_illum_head is not None:
            inputs_shared["tokenlight_source_drop_illum_head"] = self.source_drop_illum_head
            inputs_shared["tokenlight_source_drop_illum_head_config"] = self.source_drop_illum_head_config
            inputs_shared["tokenlight_source_drop_illum_loss_weight"] = self.tokenlight_source_drop_illum_loss_weight
            inputs_shared["tokenlight_source_drop_illum_latent_normalize"] = (
                self.tokenlight_source_drop_illum_latent_normalize
            )
            inputs_shared["tokenlight_source_drop_illum_latent_norm_eps"] = (
                self.tokenlight_source_drop_illum_latent_norm_eps
            )
            if isinstance(self.source_drop_illum_latent_mean, torch.Tensor):
                inputs_shared["tokenlight_source_drop_illum_latent_mean"] = self.source_drop_illum_latent_mean.to(
                    device=pbr_device,
                    dtype=torch.float32,
                )
            if isinstance(self.source_drop_illum_latent_std, torch.Tensor):
                inputs_shared["tokenlight_source_drop_illum_latent_std"] = self.source_drop_illum_latent_std.to(
                    device=pbr_device,
                    dtype=torch.float32,
                )
        inputs_shared["tokenlight_log_luminance_eps"] = self.tokenlight_log_luminance_eps
        inputs_shared["tokenlight_log_luminance_decode_tiled"] = self.tokenlight_log_luminance_decode_tiled
        inputs_shared["tokenlight_pbr_latent_loss_weight"] = self.tokenlight_pbr_latent_loss_weight
        inputs_shared["tokenlight_pbr_decoder_loss_weight"] = self.tokenlight_pbr_decoder_loss_weight
        inputs_shared["tokenlight_pbr_decoder_transforms"] = dict(self.tokenlight_pbr_decoder_transforms)
        if self.tokenlight_pbr_decoder_loss_weight != 0.0:
            decoder_targets = {}
            for stream_name in self.tokenlight_pbr_streams:
                image_key = self.tokenlight_pbr_stream_image_keys[stream_name]
                value = data.get(image_key)
                if value is None:
                    raise KeyError(
                        f"PBR decoder stream `{stream_name}` requires raw target PNG key `{image_key}`"
                    )
                decoder_targets[stream_name] = self._load_decoder_target_video_batch(
                    value,
                    batch=batch,
                    name=image_key,
                    height=int(inputs_shared["height"]),
                    width=int(inputs_shared["width"]),
                )
            inputs_shared["tokenlight_pbr_decoder_target_videos"] = decoder_targets
        if bool(source_drop.any()):
            if "tokenlight_source_latents" in inputs_shared:
                source_mask = source_drop.view(-1, 1, 1, 1, 1)
                inputs_shared["tokenlight_source_latents"] = torch.where(
                    source_mask,
                    torch.zeros_like(inputs_shared["tokenlight_source_latents"]),
                    inputs_shared["tokenlight_source_latents"],
                )
            rgb_weights = torch.ones((batch,), device=source_drop.device, dtype=torch.float32)
            rgb_weights = torch.where(
                source_drop,
                torch.full_like(rgb_weights, self.tokenlight_source_drop_rgb_loss_weight),
                rgb_weights,
            )
            inputs_shared["tokenlight_rgb_loss_weight"] = rgb_weights
            if self.source_drop_illum_head is not None and self.tokenlight_source_drop_illum_loss_weight > 0:
                target_illum = self._source_drop_illum_latents_from_cache(data, batch)
                if target_illum is None:
                    raise ValueError(
                        "Source-drop illumination head is enabled, but no illumination cache was provided. "
                        "Pass --tokenlight_source_drop_illum_cache_dir."
                    )
                inputs_shared["tokenlight_source_drop_illum_target_latents"] = self._stack_cached_latents(
                    target_illum,
                    batch=batch,
                    name="illum_target",
                )
            if self.tokenlight_source_drop_log_luminance_loss_weight > 0:
                target_videos = self._cached_target_videos_from_paths(data, batch) if cached else data["video"]
                inputs_shared["tokenlight_log_luminance_target"] = self._log_luminance_target_from_videos(target_videos)
        return inputs_shared

    def _sample_conditioning_modes(self, data, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        metadata_target = self._pbr_target_mask_from_data(data, batch, device)
        source_drop = torch.zeros((batch,), device=device, dtype=torch.bool)
        if self.tokenlight_pbr_conditioning_strategy.lower() not in {"unirelight", "mixed", "70_18_12"}:
            return metadata_target, source_drop

        total = (
            self.tokenlight_pbr_unirelight_target_prob
            + self.tokenlight_pbr_unirelight_condition_prob
            + self.tokenlight_pbr_unirelight_source_drop_prob
        )
        if total <= 0:
            return metadata_target, source_drop
        target_boundary = self.tokenlight_pbr_unirelight_target_prob / total
        condition_boundary = (
            self.tokenlight_pbr_unirelight_target_prob + self.tokenlight_pbr_unirelight_condition_prob
        ) / total
        draws = _rank0_broadcast_random_draws(batch, device)
        pbr_target = draws < target_boundary
        source_drop = draws >= condition_boundary
        return pbr_target, source_drop

    def _log_luminance_target_from_videos(self, videos) -> torch.Tensor:
        tensors = []
        for video in videos:
            frames = _as_frames(video)
            tensor = self.pipe.preprocess_video(
                frames,
                torch_dtype=torch.float32,
                device=self.pipe.device,
                min_value=0,
                max_value=1,
            )
            tensors.append(tensor)
        pixels = torch.cat(tensors, dim=0).float()
        weights = torch.tensor([0.2126, 0.7152, 0.0722], device=pixels.device, dtype=pixels.dtype).view(1, 3, 1, 1, 1)
        luminance = (pixels[:, :3] * weights).sum(dim=1, keepdim=True).clamp_min(self.tokenlight_log_luminance_eps)
        return torch.log(luminance)

    def _batched_inputs(self, data):
        inputs_shared, inputs_posi, inputs_nega = super()._batched_inputs(data)
        batch = self._batch_size_from_data(data)
        pbr_latents = {
            stream_name: self._encode_video_latents_batched(
                self._pbr_videos_from_data(data, batch, stream_name),
                inputs_shared,
            )
            for stream_name in self.tokenlight_pbr_streams
        }
        inputs_shared = self._attach_pbr_latents_to_inputs(inputs_shared, data, batch, pbr_latents, cached=False)
        return inputs_shared, inputs_posi, inputs_nega

    def _cached_batched_inputs(self, data):
        inputs_shared, inputs_posi, inputs_nega = super()._cached_batched_inputs(data)
        batch = self._batch_size_from_data(data)
        pbr_latents = self._cached_pbr_latents_from_data(data, batch)
        inputs_shared = self._attach_pbr_latents_to_inputs(inputs_shared, data, batch, pbr_latents, cached=True)
        return inputs_shared, inputs_posi, inputs_nega

    def get_pipeline_inputs(self, data):
        inputs_shared, inputs_posi, inputs_nega = super().get_pipeline_inputs(data)
        pbr_images = {}
        for stream_name in self.tokenlight_pbr_streams:
            pbr_key = self.tokenlight_pbr_stream_image_keys[stream_name]
            if pbr_key in data:
                pbr_images[stream_name] = _as_frames(data[pbr_key])[0]
        if pbr_images:
            if self.tokenlight_pbr_legacy_stream and "pbr" in pbr_images:
                inputs_shared["tokenlight_pbr_image"] = pbr_images["pbr"]
            else:
                inputs_shared["tokenlight_pbr_images"] = pbr_images
            inputs_shared["tokenlight_pbr_is_target"] = str(
                data.get(self.tokenlight_pbr_mode_key, self.tokenlight_pbr_default_mode)
            ).lower() not in {"condition", "cond", "input", "clean", "provided"}
            if self.tokenlight_pbr_decoder_loss_weight != 0.0:
                inputs_shared["tokenlight_pbr_decoder_target_videos"] = {
                    stream_name: self._load_decoder_target_video_batch(
                        data[self.tokenlight_pbr_stream_image_keys[stream_name]],
                        batch=1,
                        name=self.tokenlight_pbr_stream_image_keys[stream_name],
                        height=int(inputs_shared["height"]),
                        width=int(inputs_shared["width"]),
                    )
                    for stream_name in pbr_images
                }
        return inputs_shared, inputs_posi, inputs_nega

    def _prepare_tokenlight_inputs(self, inputs, data):
        inputs_shared, inputs_posi, inputs_nega = super()._prepare_tokenlight_inputs(inputs, data)
        if "tokenlight_pbr_image" in inputs_shared:
            inputs_shared["tokenlight_pbr_input_latents"] = self._encode_image_latents(
                inputs_shared["tokenlight_pbr_image"],
                inputs_shared,
            )
            inputs_shared["tokenlight_pbr_loss_weight"] = self.tokenlight_pbr_stream_loss_weights["pbr"]
        if "tokenlight_pbr_images" in inputs_shared:
            pbr_latents = {
                stream_name: self._encode_image_latents(image, inputs_shared)
                for stream_name, image in inputs_shared["tokenlight_pbr_images"].items()
            }
            inputs_shared["tokenlight_pbr_input_latents_map"] = pbr_latents
            inputs_shared["tokenlight_pbr_is_target_map"] = {
                stream_name: inputs_shared.get("tokenlight_pbr_is_target", True) for stream_name in pbr_latents
            }
            inputs_shared["tokenlight_pbr_loss_weights"] = {
                stream_name: self.tokenlight_pbr_stream_loss_weights[stream_name] for stream_name in pbr_latents
            }
        inputs_shared.pop("tokenlight_pbr_image", None)
        inputs_shared.pop("tokenlight_pbr_images", None)
        return inputs_shared, inputs_posi, inputs_nega


def _parse_task_batch(value: str | dict[str, int] | None) -> dict[str, int] | None:
    if value in (None, "", "none", "None", "null"):
        return None
    if isinstance(value, dict):
        result = {str(key): int(item) for key, item in value.items() if int(item) > 0}
        return result or None
    result = {}
    for item in str(value).split(","):
        if not item.strip():
            continue
        key, count = item.split(":", 1)
        result[key.strip()] = int(count)
    return {key: count for key, count in result.items() if count > 0} or None


class BalancedTaskBatchSampler(Sampler[list[int]]):
    def __init__(self, rows: Sequence[dict[str, Any]], task_batch: dict[str, int], *, seed: int = 0) -> None:
        self.rows = rows
        self.task_batch = task_batch
        self.seed = int(seed)
        self.batch_size = int(sum(task_batch.values()))
        self.drop_last = False
        self.pools: dict[str, list[int]] = {task: [] for task in task_batch}
        for index, row in enumerate(rows):
            task = str(row.get("task", ""))
            if task in self.pools:
                self.pools[task].append(index)
        missing = [task for task, pool in self.pools.items() if not pool]
        if missing:
            raise ValueError(f"Cannot build balanced batches; missing task rows: {missing}")
        self._length = max(math.ceil(len(self.pools[task]) / count) for task, count in self.task_batch.items())

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterable[list[int]]:
        rng = random.Random(self.seed)
        pools = {task: list(indices) for task, indices in self.pools.items()}
        cursors = {task: 0 for task in self.task_batch}
        for indices in pools.values():
            rng.shuffle(indices)
        for _ in range(len(self)):
            batch = []
            for task, count in self.task_batch.items():
                pool = pools[task]
                for _ in range(count):
                    if cursors[task] >= len(pool):
                        rng.shuffle(pool)
                        cursors[task] = 0
                    batch.append(pool[cursors[task]])
                    cursors[task] += 1
            rng.shuffle(batch)
            yield batch


def _configure_deepspeed_batch_size(accelerator, args, batch_size: int) -> None:
    plugin = getattr(getattr(accelerator, "state", None), "deepspeed_plugin", None)
    ds_config = getattr(plugin, "deepspeed_config", None)
    if not isinstance(ds_config, dict):
        return
    grad_accum = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
    num_processes = _resolve_distributed_world_size(accelerator)
    ds_config["train_micro_batch_size_per_gpu"] = int(batch_size)
    ds_config["gradient_accumulation_steps"] = grad_accum
    ds_config["train_batch_size"] = int(batch_size) * grad_accum * num_processes
    print(
        "DeepSpeed batch setup: "
        f"train_micro_batch_size_per_gpu={ds_config['train_micro_batch_size_per_gpu']}, "
        f"gradient_accumulation_steps={ds_config['gradient_accumulation_steps']}, "
        f"num_processes={num_processes}, "
        f"train_batch_size={ds_config['train_batch_size']}"
    )


def wan_pbr_parser(train_mode: str = "single") -> argparse.ArgumentParser:
    parser = wan_parser(train_mode)
    parser.description = f"TokenLight PBR Wan2.2-TI2V-5B {train_mode} trainer."
    parser.set_defaults(
        output_path="model/train/tokenlight_pbr_wan22_lora",
        data_file_keys="video,input_image,pbr_image",
        tokenlight_mask_tokens=False,
        tokenlight_max_lights=2,
        batch_size=5,
        balanced_task_batch="single_light:2,double_light:2,ambient_only:1",
        balanced_batch_seed=0,
    )
    parser.add_argument("--tokenlight_pbr_image_key", default="pbr_image")
    parser.add_argument("--tokenlight_pbr_mode_key", default="pbr_mode")
    parser.add_argument(
        "--tokenlight_pbr_aux_type",
        choices=("shading", "albedo", "normal", "roughness", "depth"),
        default="shading",
    )
    parser.add_argument("--tokenlight_pbr_loss_weight", type=float, default=1.0)
    parser.add_argument("--tokenlight_pbr_streams", default=None)
    parser.add_argument("--tokenlight_pbr_stream_image_keys", default=None)
    parser.add_argument("--tokenlight_pbr_stream_loss_weights", default=None)
    parser.add_argument("--tokenlight_pbr_latent_loss_weight", type=float, default=1.0)
    parser.add_argument("--tokenlight_pbr_decoder_loss_weight", type=float, default=0.0)
    parser.add_argument(
        "--tokenlight_pbr_decoder_transforms",
        default=None,
        help="Per-stream decoder transform, e.g. depth:illuminance,normal:rgb.",
    )
    parser.add_argument("--tokenlight_pbr_default_mode", choices=("target", "condition"), default="target")
    parser.add_argument("--tokenlight_pbr_conditioning_strategy", default="metadata")
    parser.add_argument("--tokenlight_pbr_unirelight_target_prob", type=float, default=0.70)
    parser.add_argument("--tokenlight_pbr_unirelight_condition_prob", type=float, default=0.18)
    parser.add_argument("--tokenlight_pbr_unirelight_source_drop_prob", type=float, default=0.12)
    parser.add_argument("--tokenlight_source_drop_rgb_loss_weight", type=float, default=0.0)
    parser.add_argument("--tokenlight_source_drop_log_luminance_loss_weight", type=float, default=1.0)
    parser.add_argument("--tokenlight_source_drop_illum_loss_weight", type=float, default=1.0)
    parser.add_argument("--tokenlight_source_drop_illum_head_path", default=None)
    parser.add_argument("--tokenlight_source_drop_illum_cache_dir", default=None)
    parser.add_argument("--tokenlight_source_drop_illum_cache_key", default="video")
    parser.add_argument("--tokenlight_source_drop_illum_cache_open_shards", type=int, default=16)
    parser.add_argument("--tokenlight_source_drop_illum_latent_normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tokenlight_source_drop_illum_latent_norm_eps", type=float, default=1e-6)
    parser.add_argument("--tokenlight_log_luminance_eps", type=float, default=1e-3)
    parser.add_argument("--tokenlight_log_luminance_decode_tiled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--repartition_label",
        default=None,
        help="Optional output label for full-checkpoint repartitioning, e.g. step-8000.",
    )
    parser.add_argument(
        "--repartition_overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow a repartition output full checkpoint directory to be replaced.",
    )
    parser.add_argument(
        "--universal_checkpoint_dir",
        default=None,
        help="Optional directory for the intermediate DeepSpeed Universal checkpoint.",
    )
    parser.add_argument(
        "--universal_overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rebuild the intermediate Universal checkpoint if it already exists.",
    )
    parser.add_argument(
        "--universal_num_extract_workers",
        type=int,
        default=1,
        help="Worker count for DeepSpeed ds_to_universal ZeRO fragment extraction.",
    )
    parser.add_argument(
        "--universal_num_merge_workers",
        type=int,
        default=1,
        help="Worker count for DeepSpeed ds_to_universal slice merging.",
    )
    return parser


def _default_config_for_mode(train_mode: str) -> str:
    return os.environ.get("TOKENLIGHT_TRAIN_CONFIG", PBR_CONFIG_BY_MODE.get(train_mode, DEFAULT_TRAIN_CONFIG_PATH))


def parse_tokenlight_pbr_args(train_mode: str = "single"):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=_default_config_for_mode(train_mode))
    pre_args, _ = pre_parser.parse_known_args()
    raw_config = _load_json_config(pre_args.config)

    parser = wan_pbr_parser(train_mode)
    _apply_config_defaults(parser, raw_config)
    parser.set_defaults(config=pre_args.config)
    args = parser.parse_args()
    args.train_mode = train_mode
    raw_config = _load_json_config(args.config)

    args.data_file_keys = _csv_value(args.data_file_keys)
    _resolve_weight_paths(args)
    _append_timestamp_to_output_path(args)
    return args, raw_config, raw_config


def _pbr_image_keys_from_args(args) -> list[str]:
    streams = _parse_pbr_streams(getattr(args, "tokenlight_pbr_streams", None))
    if not streams:
        return [str(args.tokenlight_pbr_image_key)]
    image_key_map = _parse_pbr_stream_map(getattr(args, "tokenlight_pbr_stream_image_keys", None), cast=str)
    return [image_key_map.get(name, str(args.tokenlight_pbr_image_key) if name == "pbr" else f"pbr_{name}_image") for name in streams]


def build_dataset(args):
    dataset = build_tokenlight_dataset(args)
    if dataset.data:
        normalized = _normalize_tokenlight_metadata_rows(dataset.data)
        pbr_image_keys = _pbr_image_keys_from_args(args)
        dataset.data = [row for row in normalized if all(row.get(key) for key in pbr_image_keys)]
    return dataset


def launch_tokenlight_pbr_training_task(
    accelerator,
    dataset,
    model,
    model_logger,
    args=None,
    **kwargs,
):
    del kwargs
    batch_size = int(args.batch_size)
    task_batch = _parse_task_batch(args.balanced_task_batch)

    trainable_dtype = _preferred_trainable_dtype(model)
    before_dtype_counts = _coerce_trainable_parameter_dtype(model, trainable_dtype)
    after_dtype_counts = _trainable_dtype_counts(model)
    illum_head = getattr(model, "source_drop_illum_head", None)
    illum_head_before_dtype_counts = _coerce_floating_module_dtype(illum_head, trainable_dtype)
    illum_head_after_dtype_counts = _floating_module_dtype_counts(illum_head)
    if accelerator.is_main_process:
        print(
            "Trainable parameter dtype setup: "
            f"target={trainable_dtype}, before={before_dtype_counts}, after={after_dtype_counts}"
        )
        if illum_head is not None:
            print(
                "Source-drop illumination head dtype setup: "
                f"target={trainable_dtype}, before={illum_head_before_dtype_counts}, "
                f"after={illum_head_after_dtype_counts}"
            )

    optimizer_class = get_optimizer_class(args.customized_optimizer)
    optimizer = optimizer_class(_trainable_parameters(model), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    if task_batch is None:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate_tokenlight_batch,
            drop_last=batch_size > 1,
            num_workers=args.dataset_num_workers,
        )
    else:
        if sum(task_batch.values()) != batch_size:
            raise ValueError(f"balanced_task_batch sums to {sum(task_batch.values())}, but batch_size={batch_size}")
        batch_sampler = BalancedTaskBatchSampler(dataset.data, task_batch, seed=int(args.balanced_batch_seed))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=_collate_tokenlight_batch,
            num_workers=args.dataset_num_workers,
        )

    enable_model_cpu_offload = getattr(args, "enable_model_cpu_offload", False)
    enable_optimizer_cpu_offload = getattr(args, "enable_optimizer_cpu_offload", False)
    cpu_offload_split_threshold = getattr(args, "cpu_offload_split_threshold", None)
    _configure_deepspeed_batch_size(accelerator, args, batch_size)
    if enable_model_cpu_offload:
        optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
        model.pipe.device = accelerator.device
        offload_manager = OffloadTrainingManager(
            model,
            accelerator.device,
            enable_optimizer_cpu_offload,
            cpu_offload_split_threshold,
        )
    else:
        model.to(device=accelerator.device)
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
        offload_manager = None

    _maybe_load_full_training_state(args, accelerator, model_logger, model=model)
    save_training_runtime_snapshot(args, accelerator, model)
    tb_metrics = TokenLightTensorBoardMetrics(args.output_path, enabled=args.enable_tensorboard_log)
    initialize_deepspeed_gradient_checkpointing(accelerator)
    local_step = int(getattr(model_logger, "num_steps", 0))
    try:
        for epoch_id in range(int(args.start_epoch), int(args.start_epoch) + int(args.num_epochs)):
            iterator = tqdm(
                dataloader,
                desc=f"pbr epoch {epoch_id}",
                disable=not accelerator.is_local_main_process,
            )
            for data in iterator:
                with accelerator.accumulate(model):
                    loss = model(data)
                    accelerator.backward(loss)
                    if enable_model_cpu_offload:
                        offload_manager.after_backward()
                    optimizer.step()
                    scheduler.step()
                    metrics = _collect_train_metrics(accelerator, model, optimizer, loss)
                    optimizer.zero_grad()
                    _call_compatible_method(
                        model_logger,
                        "on_step_end",
                        accelerator,
                        model,
                        args.save_steps,
                        loss=loss,
                        epoch=epoch_id,
                        optimizer=optimizer,
                        scheduler=scheduler,
                    )
                    logger_step = getattr(model_logger, "num_steps", None)
                    if logger_step is None or int(logger_step) <= local_step:
                        local_step += 1
                    else:
                        local_step = int(logger_step)
                    tb_metrics.log(accelerator, local_step, metrics)
                    if metrics and "train/loss" in metrics:
                        iterator.set_postfix(loss=f"{metrics['train/loss']:.4f}", step=local_step)
            if args.save_steps is None:
                _call_compatible_method(
                    model_logger,
                    "on_epoch_end",
                    accelerator,
                    model,
                    epoch_id,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
        _call_compatible_method(
            model_logger,
            "on_training_end",
            accelerator,
            model,
            args.save_steps,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    finally:
        tb_metrics.close(accelerator)


def main(train_mode: str = "single") -> None:
    args, raw_config, merged_config = parse_tokenlight_pbr_args(train_mode)
    accelerator_kwargs = {
        "kwargs_handlers": [
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)
        ],
    }
    if "even_batches" in inspect.signature(accelerate.Accelerator).parameters:
        accelerator_kwargs["even_batches"] = False
    if train_mode == "single":
        accelerator_kwargs["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    accelerator = accelerate.Accelerator(**accelerator_kwargs)
    if hasattr(accelerator, "even_batches"):
        accelerator.even_batches = False
    save_training_config_snapshot(args, raw_config, merged_config, accelerator)
    dataset = build_dataset(args)
    model = TokenLightPBRWanTrainingModule(
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
        resume_from_checkpoint=_constructor_resume_checkpoint(getattr(args, "resume_from_checkpoint", None)),
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device="cpu" if (args.initialize_model_on_cpu or getattr(args, "enable_model_cpu_offload", False)) else accelerator.device,
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
        tokenlight_pbr_image_key=args.tokenlight_pbr_image_key,
        tokenlight_pbr_mode_key=args.tokenlight_pbr_mode_key,
        tokenlight_pbr_aux_type=args.tokenlight_pbr_aux_type,
        tokenlight_pbr_loss_weight=args.tokenlight_pbr_loss_weight,
        tokenlight_pbr_streams=args.tokenlight_pbr_streams,
        tokenlight_pbr_stream_image_keys=args.tokenlight_pbr_stream_image_keys,
        tokenlight_pbr_stream_loss_weights=args.tokenlight_pbr_stream_loss_weights,
        tokenlight_pbr_latent_loss_weight=args.tokenlight_pbr_latent_loss_weight,
        tokenlight_pbr_decoder_loss_weight=args.tokenlight_pbr_decoder_loss_weight,
        tokenlight_pbr_decoder_transforms=args.tokenlight_pbr_decoder_transforms,
        tokenlight_pbr_default_mode=args.tokenlight_pbr_default_mode,
        tokenlight_pbr_conditioning_strategy=args.tokenlight_pbr_conditioning_strategy,
        tokenlight_pbr_unirelight_target_prob=args.tokenlight_pbr_unirelight_target_prob,
        tokenlight_pbr_unirelight_condition_prob=args.tokenlight_pbr_unirelight_condition_prob,
        tokenlight_pbr_unirelight_source_drop_prob=args.tokenlight_pbr_unirelight_source_drop_prob,
        tokenlight_source_drop_rgb_loss_weight=args.tokenlight_source_drop_rgb_loss_weight,
        tokenlight_source_drop_log_luminance_loss_weight=args.tokenlight_source_drop_log_luminance_loss_weight,
        tokenlight_source_drop_illum_loss_weight=args.tokenlight_source_drop_illum_loss_weight,
        tokenlight_source_drop_illum_head_path=args.tokenlight_source_drop_illum_head_path,
        tokenlight_source_drop_illum_cache_dir=args.tokenlight_source_drop_illum_cache_dir,
        tokenlight_source_drop_illum_cache_key=args.tokenlight_source_drop_illum_cache_key,
        tokenlight_source_drop_illum_cache_open_shards=args.tokenlight_source_drop_illum_cache_open_shards,
        tokenlight_source_drop_illum_latent_normalize=args.tokenlight_source_drop_illum_latent_normalize,
        tokenlight_source_drop_illum_latent_norm_eps=args.tokenlight_source_drop_illum_latent_norm_eps,
        tokenlight_log_luminance_eps=args.tokenlight_log_luminance_eps,
        tokenlight_log_luminance_decode_tiled=args.tokenlight_log_luminance_decode_tiled,
        tokenlight_rgb_latent_loss_weight=args.tokenlight_rgb_latent_loss_weight,
        tokenlight_rgb_decoder_loss_weight=args.tokenlight_rgb_decoder_loss_weight,
        tokenlight_rgb_decoder_transform=args.tokenlight_rgb_decoder_transform,
        tokenlight_decoder_loss_type=args.tokenlight_decoder_loss_type,
        tokenlight_decoder_tiled=args.tokenlight_decoder_tiled,
        tokenlight_decoder_luminance_eps=args.tokenlight_decoder_luminance_eps,
        tokenlight_decoder_charbonnier_eps=args.tokenlight_decoder_charbonnier_eps,
        dataset_base_path=args.dataset_base_path,
    )
    model_logger = _make_model_logger(args)
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_tokenlight_pbr_training_task,
        "sft:train": launch_tokenlight_pbr_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)


if __name__ == "__main__":
    main()
