from __future__ import annotations

"""Stable opt-in decoder-space training for TokenLight.

This entrypoint intentionally leaves ``model/train_tokenlight.py`` untouched.
When the decoder weight is zero it delegates to DiffSynth's original
``FlowMatchSFTLoss``.  The decoder-on path compares decoded flow states at the
sampled timestep and uses a single-GPU-safe FP32 optimizer path.
"""

import argparse
import json
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import accelerate
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.checkpoint import checkpoint


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import train_tokenlight as base  # noqa: E402


LOSS_IMPL_VERSION = "tokenlight_decoder_current_t_v3"
DECODER_TRANSFORMS = ("rgb", "luminance", "log_luminance")
DECODER_LOSS_TYPES = ("mse", "l1", "charbonnier")


def _as_scalar_tensor(value: Any, reference: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=reference.device, dtype=torch.float32).reshape(())
    return reference.new_tensor(float(value), dtype=torch.float32)


def _scheduler_training_weight(pipe, timestep_id: int, reference: torch.Tensor) -> torch.Tensor:
    weights = getattr(pipe.scheduler, "linear_timesteps_weights", None)
    if isinstance(weights, torch.Tensor):
        return _as_scalar_tensor(weights[timestep_id], reference)
    scheduler_timestep = pipe.scheduler.timesteps[timestep_id]
    return _as_scalar_tensor(pipe.scheduler.training_weight(scheduler_timestep), reference)


def _sample_flow_state(pipe, clean_latents: torch.Tensor, inputs: Mapping[str, Any]):
    total = len(pipe.scheduler.timesteps)
    max_boundary = int(float(inputs.get("max_timestep_boundary", 1.0)) * total)
    min_boundary = int(float(inputs.get("min_timestep_boundary", 0.0)) * total)
    min_boundary = max(0, min(total, min_boundary))
    max_boundary = max(0, min(total, max_boundary))
    if max_boundary <= min_boundary:
        raise ValueError(
            f"Empty timestep interval: min={min_boundary}, max={max_boundary}, total={total}"
        )

    # Keep the sampled scheduler index authoritative. Casting 1000 Wan
    # timesteps to BF16 before scheduler lookup collapses them to ~294 unique
    # values, which can select a different sigma.
    timestep_id = int(torch.randint(min_boundary, max_boundary, (1,), device="cpu").item())
    scheduler_timestep = pipe.scheduler.timesteps[timestep_id]
    model_timestep = scheduler_timestep.reshape(1).to(dtype=pipe.torch_dtype, device=pipe.device)
    sigma_value = pipe.scheduler.sigmas[timestep_id]
    sigma_scalar = _as_scalar_tensor(sigma_value, clean_latents)
    sigma = sigma_scalar.to(dtype=clean_latents.dtype).reshape(
        (1,) + (1,) * (clean_latents.ndim - 1)
    )
    noise = torch.randn_like(clean_latents) * float(inputs.get("noise_scale", 1.0))
    noisy_latents = (1.0 - sigma) * clean_latents + sigma * noise
    velocity_target = noise - clean_latents
    return timestep_id, model_timestep, sigma_scalar, noisy_latents, velocity_target


def _decoder_ramp(inputs: Mapping[str, Any]) -> float:
    step = max(0, int(inputs.get("tokenlight_optimizer_step", 0)))
    warmup = max(0, int(inputs.get("tokenlight_decoder_warmup_steps", 0)))
    ramp_steps = max(0, int(inputs.get("tokenlight_decoder_ramp_steps", 0)))
    if step < warmup:
        return 0.0
    if ramp_steps == 0:
        return 1.0
    return min(1.0, max(0.0, (step - warmup + 1) / float(ramp_steps)))


def _vae_model_and_scale(pipe):
    vae = getattr(pipe, "vae", None)
    model = getattr(vae, "model", None)
    if not isinstance(model, nn.Module):
        raise RuntimeError("Wan VAE model is unavailable for decoder-space loss")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("VAE must remain frozen; decoder loss only differentiates its latent input")
    model.eval()
    return model, getattr(vae, "scale", None)


def _decode_unclamped(
    pipe,
    latents: torch.Tensor,
    *,
    gradient_checkpointing: bool,
) -> torch.Tensor:
    """Decode without DiffSynth's final clamp_(-1, 1).

    The wrapper clamp has zero gradient for saturated predictions.  Calling the
    frozen Wan decoder directly preserves useful gradients outside that range.
    """

    model, scale = _vae_model_and_scale(pipe)
    try:
        parameter = next(model.parameters())
    except StopIteration:
        parameter = None
    if parameter is not None and parameter.device != latents.device:
        raise RuntimeError(
            "Decoder loss requires the VAE and DiT graph to be resident on the same device. "
            f"VAE is on {parameter.device}, latents are on {latents.device}. "
            "Do not combine this trainer with generic model offload."
        )

    def decode_and_clear_cache(value: torch.Tensor) -> torch.Tensor:
        # Wan's causal decoder stores temporal features on the module.  Those
        # references must not retain the prediction graph after forward, and
        # must also be cleared after checkpoint recomputation in backward.
        try:
            return model.decode(value, scale)
        finally:
            clear_cache = getattr(model, "clear_cache", None)
            if callable(clear_cache):
                clear_cache()

    outputs = []
    for sample in latents.split(1, dim=0):
        if gradient_checkpointing and sample.requires_grad:
            decoded = checkpoint(
                decode_and_clear_cache,
                sample,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            decoded = decode_and_clear_cache(sample)
        outputs.append(decoded)
    return torch.cat(outputs, dim=0)


def _transform_video(video: torch.Tensor, transform: str, eps: float) -> torch.Tensor:
    transform = str(transform).lower().replace("-", "_")
    video = video.float()
    if transform == "rgb":
        return video
    rgb = (video + 1.0) * 0.5
    weights = rgb.new_tensor((0.2126, 0.7152, 0.0722)).reshape(1, 3, 1, 1, 1)
    luminance = (rgb[:, :3] * weights).sum(dim=1, keepdim=True)
    if transform == "luminance":
        return luminance
    if transform == "log_luminance":
        return torch.log(luminance.clamp_min(float(eps)))
    raise ValueError(f"Unsupported decoder transform: {transform!r}")


def _pointwise_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_type: str,
    charbonnier_eps: float,
) -> torch.Tensor:
    delta = pred.float() - target.float()
    if loss_type == "mse":
        return delta.square()
    if loss_type == "l1":
        return delta.abs()
    if loss_type == "charbonnier":
        return (delta.square() + float(charbonnier_eps) ** 2).sqrt()
    raise ValueError(f"Unsupported decoder loss type: {loss_type!r}")


def _mean_with_mask(error: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=error.device, dtype=error.dtype)
    denominator = mask.sum(dim=(1, 2, 3, 4)) * int(error.shape[1])
    numerator = (error * mask).sum(dim=(1, 2, 3, 4))
    valid = denominator > 0
    if not bool(valid.any()):
        return error.sum() * 0.0
    return (numerator[valid] / denominator[valid]).mean()


def _region_loss_with_floor(
    error: torch.Tensor,
    mask: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    min_fraction: float,
) -> torch.Tensor:
    mask = mask.to(device=error.device, dtype=torch.float32)
    if mask.ndim == 4:
        mask = mask.unsqueeze(2)
    if mask.ndim != 5 or int(mask.shape[0]) != int(error.shape[0]):
        raise ValueError(f"Region mask must be [B,1,T,H,W], got {tuple(mask.shape)}")
    if tuple(mask.shape[2:]) != tuple(error.shape[2:]):
        mask = F.interpolate(mask, size=error.shape[2:], mode="nearest")
    mask = mask.clamp(0.0, 1.0).to(dtype=error.dtype) * valid_mask

    channels = int(error.shape[1])
    numerator = (error * mask).sum(dim=(1, 2, 3, 4))
    region_pixels = mask.sum(dim=(1, 2, 3, 4))
    valid_pixels = valid_mask.sum(dim=(1, 2, 3, 4))
    floor_pixels = valid_pixels * float(min_fraction)
    denominator = torch.maximum(region_pixels, floor_pixels) * channels
    nonempty = region_pixels > 0
    if not bool(nonempty.any()):
        return error.sum() * 0.0
    return (numerator[nonempty] / denominator[nonempty].clamp_min(1.0)).mean()


def safe_decoder_space_loss(
    pipe,
    *,
    pred_current_latents: torch.Tensor,
    target_current_latents: torch.Tensor,
    region_masks: Mapping[str, torch.Tensor] | None,
    region_weights: Mapping[str, float],
    full_loss_weight: float,
    transform: str,
    loss_type: str,
    luminance_eps: float,
    charbonnier_eps: float,
    min_region_fraction: float,
    max_samples: int,
    gradient_checkpointing: bool,
    exclude_first_frame: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if pred_current_latents.shape != target_current_latents.shape:
        raise ValueError(
            f"Decoder latent shape mismatch: {tuple(pred_current_latents.shape)} != "
            f"{tuple(target_current_latents.shape)}"
        )

    batch = int(pred_current_latents.shape[0])
    count = min(batch, max(1, int(max_samples)))
    if count < batch:
        indices = torch.randperm(batch, device=pred_current_latents.device)[:count]
        pred_current_latents = pred_current_latents.index_select(0, indices)
        target_current_latents = target_current_latents.index_select(0, indices)
        if region_masks is not None:
            region_masks = {
                name: mask.index_select(0, indices.to(device=mask.device))
                for name, mask in region_masks.items()
            }

    # Decode the stop-gradient target first, before allocating the prediction
    # decoder graph.  Both branches are evaluated at the sampled flow timestep:
    # D(x0 + sigma*v_pred) is compared with D(x0 + sigma*v_target) = D(x_t).
    with torch.no_grad():
        target_video = _decode_unclamped(
            pipe,
            target_current_latents.detach(),
            gradient_checkpointing=False,
        )
    pred_video = _decode_unclamped(
        pipe,
        pred_current_latents,
        gradient_checkpointing=bool(gradient_checkpointing),
    )

    pred_video = _transform_video(pred_video, transform, luminance_eps)
    target_video = _transform_video(target_video, transform, luminance_eps)
    error = _pointwise_loss(
        pred_video,
        target_video,
        loss_type=loss_type,
        charbonnier_eps=charbonnier_eps,
    )

    valid_mask = torch.ones(
        (error.shape[0], 1, error.shape[2], error.shape[3], error.shape[4]),
        device=error.device,
        dtype=error.dtype,
    )
    if exclude_first_frame:
        if int(error.shape[2]) <= 1:
            raise ValueError("Cannot exclude the only decoded frame from decoder loss")
        valid_mask[:, :, 0] = 0

    full_loss = _mean_with_mask(error, valid_mask)
    total = float(full_loss_weight) * full_loss
    components = {"full": full_loss.detach()}
    for name, weight in region_weights.items():
        weight = float(weight)
        if weight == 0.0:
            continue
        if region_masks is None or name not in region_masks:
            raise KeyError(f"Enabled decoder region {name!r} has no mask")
        region_loss = _region_loss_with_floor(
            error,
            region_masks[name],
            valid_mask,
            min_fraction=float(min_region_fraction),
        )
        total = total + weight * region_loss
        components[name] = region_loss.detach()
    components["total"] = total.detach()
    return total, components


def TokenLightSafeDecoderFlowMatchLoss(pipe, **inputs):
    latent_weight = float(inputs.get("tokenlight_rgb_latent_loss_weight", 1.0))
    decoder_weight = float(inputs.get("tokenlight_rgb_decoder_loss_weight", 0.0))

    # Preserve the successful path exactly: same native function, RNG order,
    # scheduler lookup, dtype behavior, and weighting.
    if decoder_weight == 0.0 and latent_weight == 1.0:
        return base.FlowMatchSFTLoss(pipe, **inputs)
    if latent_weight == 0.0 and decoder_weight == 0.0:
        raise ValueError("Latent and decoder weights cannot both be zero")

    if "lora" in inputs:
        pipe.clear_lora(verbose=0)
        pipe.load_lora(pipe.dit, state_dict=inputs["lora"], hotload=True, verbose=0)

    clean_latents = inputs["input_latents"]
    timestep_id, timestep, sigma, noisy_latents, velocity_target = _sample_flow_state(
        pipe, clean_latents, inputs
    )
    model_latents = noisy_latents
    if "first_frame_latents" in inputs:
        model_latents = noisy_latents.clone()
        model_latents[:, :, 0:1] = inputs["first_frame_latents"]
    inputs["latents"] = model_latents

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    velocity_pred_full = pipe.model_fn(**models, **inputs, timestep=timestep)

    velocity_pred = velocity_pred_full
    latent_target = velocity_target
    if "first_frame_latents" in inputs:
        velocity_pred = velocity_pred[:, :, 1:]
        latent_target = latent_target[:, :, 1:]

    latent_mse = F.mse_loss(velocity_pred.float(), latent_target.float())
    scheduler_weight = _scheduler_training_weight(pipe, timestep_id, latent_mse)
    latent_term = latent_weight * scheduler_weight * latent_mse
    total = latent_term
    metrics = {
        "train/rgb/latent_mse": latent_mse.detach(),
        "train/rgb/latent_term": latent_term.detach(),
        "train/rgb/sigma": sigma.detach(),
        "train/rgb/scheduler_weight": scheduler_weight.detach(),
    }

    ramp = _decoder_ramp(inputs)
    if decoder_weight != 0.0 and ramp > 0.0:
        sigma_view = sigma.to(device=noisy_latents.device, dtype=noisy_latents.dtype).reshape(
            (1,) + (1,) * (noisy_latents.ndim - 1)
        )
        # Rectified-flow pixel supervision at the sampled timestep.  A perfect
        # velocity prediction makes pred_current exactly equal model_latents.
        # This is one denoiser forward plus two VAE decodes; it does not unroll
        # the complete inference trajectory.
        pred_current = clean_latents + sigma_view * velocity_pred_full
        target_current = model_latents.detach()
        exclude_first = "first_frame_latents" in inputs
        if exclude_first:
            # Use the exact conditioning prefix seen by the model in both VAE
            # branches, then exclude its decoded pixel frame from the loss.
            pred_current = pred_current.clone()
            pred_current[:, :, 0:1] = model_latents[:, :, 0:1].detach()

        decoder_raw, decoder_components = safe_decoder_space_loss(
            pipe,
            pred_current_latents=pred_current,
            target_current_latents=target_current,
            region_masks=inputs.get("tokenlight_rgb_decoder_region_masks"),
            region_weights=inputs.get("tokenlight_rgb_decoder_region_weights", {}),
            full_loss_weight=float(inputs.get("tokenlight_rgb_decoder_full_loss_weight", 1.0)),
            transform=str(inputs.get("tokenlight_rgb_decoder_transform", "rgb")),
            loss_type=str(inputs.get("tokenlight_decoder_loss_type", "l1")),
            luminance_eps=float(inputs.get("tokenlight_decoder_luminance_eps", 1e-3)),
            charbonnier_eps=float(inputs.get("tokenlight_decoder_charbonnier_eps", 1e-3)),
            min_region_fraction=float(inputs.get("tokenlight_decoder_min_region_fraction", 0.1)),
            max_samples=int(inputs.get("tokenlight_decoder_max_samples_per_batch", 1)),
            gradient_checkpointing=bool(inputs.get("tokenlight_decoder_gradient_checkpointing", True)),
            exclude_first_frame=exclude_first,
        )
        decoder_term = float(decoder_weight) * float(ramp) * decoder_raw
        total = total + decoder_term
        metrics["train/rgb/decoder_raw"] = decoder_raw.detach()
        metrics["train/rgb/decoder_term"] = decoder_term.detach()
        for name, value in decoder_components.items():
            metrics[f"train/rgb/decoder_{name}"] = value.detach()
    metrics["train/rgb/decoder_ramp"] = latent_mse.new_tensor(float(ramp))
    metrics["train/rgb/decoder_effective_weight"] = latent_mse.new_tensor(
        float(decoder_weight) * float(ramp)
    )
    metrics["train/rgb/total_loss"] = total.detach()
    pipe._tokenlight_loss_metrics = metrics
    return total


class _MaskLatentCacheDataset(torch.utils.data.Dataset):
    """Attach a separately cached conditioning mask without changing base.py."""

    def __init__(self, dataset, cache_dir: str | Path, mask_key: str, shard_lru: int) -> None:
        self.dataset = dataset
        self.mask_key = str(mask_key)
        self.store = base.VaeLatentCacheStore(
            base._resolve_path(cache_dir, base_path=REPO_ROOT),
            shard_lru=shard_lru,
        )
        self.data = dataset.data
        self.repeat = getattr(dataset, "repeat", 1)
        self.load_from_cache = getattr(dataset, "load_from_cache", False)
        self.is_vae_latent_cache_dataset = True
        self.skip_tokenlight_file_filter = getattr(dataset, "skip_tokenlight_file_filter", False)

    def __len__(self):
        return len(self.dataset)

    def _get(self, index: int):
        item = self.dataset[index]
        mask_path = item.get(self.mask_key)
        if not isinstance(mask_path, str) or not mask_path:
            raise KeyError(f"Missing mask key {self.mask_key!r} in cached metadata row")
        item["_tokenlight_mask_latents"] = self.store.get(mask_path)
        return item

    def __getitem__(self, index: int):
        return self._get(int(index))

    def __getitems__(self, indices: list[int]):
        return [self._get(int(index)) for index in indices]


class _ExtraMaskLatentCacheDataset(torch.utils.data.Dataset):
    """Attach multiple independently cached VAE mask prefixes."""

    def __init__(self, dataset, mask_keys: list[str], cache_dirs: list[str], shard_lru: int) -> None:
        self.dataset = dataset
        self.mask_keys = tuple(mask_keys)
        self.stores = tuple(
            base.VaeLatentCacheStore(base._resolve_path(path, base_path=REPO_ROOT), shard_lru=shard_lru)
            for path in cache_dirs
        )
        self.data = dataset.data
        self.repeat = getattr(dataset, "repeat", 1)
        self.load_from_cache = getattr(dataset, "load_from_cache", False)
        self.is_vae_latent_cache_dataset = True
        self.skip_tokenlight_file_filter = getattr(dataset, "skip_tokenlight_file_filter", False)

    def __len__(self):
        return len(self.dataset)

    def _get(self, index: int):
        item = self.dataset[index]
        for index, (key, store) in enumerate(zip(self.mask_keys, self.stores)):
            mask_path = item.get(key)
            if not isinstance(mask_path, str) or not mask_path:
                raise KeyError(f"Missing extra mask key {key!r} in cached metadata row")
            item[f"_tokenlight_extra_mask_latents_{index}"] = store.get(mask_path)
        return item

    def __getitem__(self, index: int):
        return self._get(int(index))

    def __getitems__(self, indices: list[int]):
        return [self._get(int(index)) for index in indices]


def _load_rejected_scene_ids(value: str | Path | None) -> set[str]:
    if value in (None, "", "None", "none", "null"):
        return set()
    path = base._resolve_path(value, base_path=REPO_ROOT)
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset reject list: {path}")
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _row_is_rejected(row: Mapping[str, Any], rejected: set[str]) -> bool:
    if not rejected:
        return False
    for key in ("scene_id", "scene_folder"):
        value = row.get(key)
        if value is not None and str(value).strip() in rejected:
            return True
    for key in ("video", "input_image"):
        value = row.get(key)
        if isinstance(value, str) and any(part in rejected for part in Path(value).parts):
            return True
    return False


def build_safe_dataset(args):
    # UnifiedDataset only loads declared file keys.  Add the configured mask
    # key when it must be encoded on the fly.
    mask_cache = getattr(args, "tokenlight_mask_latent_cache_dir", None)
    mask_key = str(getattr(args, "tokenlight_mask_image_key", "mask") or "mask")
    extra_mask_keys = [
        key.strip()
        for key in str(getattr(args, "tokenlight_extra_mask_image_keys", "") or "").split(",")
        if key.strip()
    ]
    extra_cache_dirs = [
        path.strip()
        for path in str(getattr(args, "tokenlight_extra_mask_latent_cache_dirs", "") or "").split(",")
        if path.strip()
    ]
    if extra_cache_dirs and len(extra_cache_dirs) != len(extra_mask_keys):
        raise ValueError("tokenlight_extra_mask_latent_cache_dirs must match tokenlight_extra_mask_image_keys")
    keys = [key for key in str(args.data_file_keys).split(",") if key]
    if bool(args.tokenlight_mask_tokens) and not mask_cache and mask_key not in keys:
        keys.append(mask_key)
        args.data_file_keys = ",".join(keys)
    if bool(args.tokenlight_mask_tokens):
        for extra_mask_key in extra_mask_keys:
            if extra_mask_key not in keys:
                keys.append(extra_mask_key)
        args.data_file_keys = ",".join(keys)

    dataset = base.build_dataset(args)
    rejected = _load_rejected_scene_ids(getattr(args, "dataset_reject_metadata_path", None))
    if rejected:
        before = len(dataset.data)
        dataset.data[:] = [row for row in dataset.data if not _row_is_rejected(row, rejected)]
        print(f"Reject-list filter: kept {len(dataset.data)} of {before} rows")
        if not dataset.data:
            raise ValueError("Reject-list filtering removed every training row")

    if bool(args.tokenlight_mask_tokens) and mask_cache:
        if not getattr(dataset, "is_vae_latent_cache_dataset", False):
            raise ValueError("A separate mask latent cache requires vae_latent_cache_specs")
        dataset = _MaskLatentCacheDataset(
            dataset,
            mask_cache,
            mask_key,
            int(getattr(args, "vae_latent_cache_shard_lru", 64)),
        )
    if bool(args.tokenlight_mask_tokens) and extra_cache_dirs:
        if not getattr(dataset, "is_vae_latent_cache_dataset", False):
            raise ValueError("Extra mask latent caches require vae_latent_cache_specs")
        dataset = _ExtraMaskLatentCacheDataset(
            dataset,
            extra_mask_keys,
            extra_cache_dirs,
            int(getattr(args, "vae_latent_cache_shard_lru", 64)),
        )
    return dataset


class TokenLightSafeDecoderTrainingModule(base.TokenLightWanTrainingModule):
    def __init__(
        self,
        *args,
        tokenlight_rgb_latent_loss_weight: float = 1.0,
        tokenlight_rgb_decoder_loss_weight: float = 0.0,
        tokenlight_rgb_decoder_transform: str = "rgb",
        tokenlight_decoder_loss_type: str = "l1",
        tokenlight_decoder_luminance_eps: float = 1e-3,
        tokenlight_decoder_charbonnier_eps: float = 1e-3,
        tokenlight_rgb_decoder_full_loss_weight: float = 1.0,
        tokenlight_rgb_decoder_shadow_loss_weight: float = 0.0,
        tokenlight_rgb_decoder_direct_loss_weight: float = 0.0,
        tokenlight_rgb_decoder_shadow_mask_key: str = "shadow_mask",
        tokenlight_rgb_decoder_direct_mask_key: str = "inf_mask",
        tokenlight_decoder_min_region_fraction: float = 0.1,
        tokenlight_decoder_max_samples_per_batch: int = 1,
        tokenlight_decoder_gradient_checkpointing: bool = True,
        tokenlight_decoder_warmup_steps: int = 1000,
        tokenlight_decoder_ramp_steps: int = 1000,
        tokenlight_mask_image_key: str = "mask",
        tokenlight_extra_mask_image_keys: str = "",
        tokenlight_extra_mask_latent_cache_dirs: str = "",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tokenlight_mask_image_key = str(tokenlight_mask_image_key or "mask")
        self.tokenlight_extra_mask_image_keys = tuple(
            key.strip() for key in str(tokenlight_extra_mask_image_keys or "").split(",") if key.strip()
        )
        self.tokenlight_extra_mask_latent_cache_dirs = tuple(
            path.strip()
            for path in str(tokenlight_extra_mask_latent_cache_dirs or "").split(",")
            if path.strip()
        )
        self.decoder_optimizer_step = 0
        self.decoder_loss_options = {
            "tokenlight_rgb_latent_loss_weight": float(tokenlight_rgb_latent_loss_weight),
            "tokenlight_rgb_decoder_loss_weight": float(tokenlight_rgb_decoder_loss_weight),
            "tokenlight_rgb_decoder_transform": str(tokenlight_rgb_decoder_transform),
            "tokenlight_decoder_loss_type": str(tokenlight_decoder_loss_type),
            "tokenlight_decoder_luminance_eps": float(tokenlight_decoder_luminance_eps),
            "tokenlight_decoder_charbonnier_eps": float(tokenlight_decoder_charbonnier_eps),
            "tokenlight_rgb_decoder_full_loss_weight": float(tokenlight_rgb_decoder_full_loss_weight),
            "tokenlight_decoder_min_region_fraction": float(tokenlight_decoder_min_region_fraction),
            "tokenlight_decoder_max_samples_per_batch": int(tokenlight_decoder_max_samples_per_batch),
            "tokenlight_decoder_gradient_checkpointing": bool(tokenlight_decoder_gradient_checkpointing),
            "tokenlight_decoder_warmup_steps": int(tokenlight_decoder_warmup_steps),
            "tokenlight_decoder_ramp_steps": int(tokenlight_decoder_ramp_steps),
        }
        self.decoder_region_weights = {
            "shadow": float(tokenlight_rgb_decoder_shadow_loss_weight),
            "direct": float(tokenlight_rgb_decoder_direct_loss_weight),
        }
        self.decoder_region_keys = {
            "shadow": str(tokenlight_rgb_decoder_shadow_mask_key),
            "direct": str(tokenlight_rgb_decoder_direct_mask_key),
        }
        self.task_to_loss["sft"] = self._safe_sft_loss
        self.task_to_loss["sft:train"] = self._safe_sft_loss

    def _safe_sft_loss(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        del inputs_nega
        merged = dict(inputs_shared)
        merged.update(inputs_posi)
        merged.update(self.decoder_loss_options)
        merged["tokenlight_optimizer_step"] = int(self.decoder_optimizer_step)
        return TokenLightSafeDecoderFlowMatchLoss(pipe, **merged)

    @staticmethod
    def _mask_batch_values(value, batch: int) -> list[Any]:
        if batch == 1:
            if isinstance(value, list) and len(value) == 1:
                return value
            return [value]
        if not isinstance(value, list) or len(value) != batch:
            raise ValueError(f"Expected {batch} mask values, got {type(value).__name__}")
        return value

    def _load_region_mask_batch(self, value, *, batch: int, name: str) -> torch.Tensor:
        values = self._mask_batch_values(value, batch)
        tensors = []
        for item in values:
            while isinstance(item, (list, tuple)) and len(item) == 1:
                item = item[0]
            if isinstance(item, torch.Tensor):
                tensor = item.detach().float()
                while tensor.ndim > 2 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                if tensor.ndim == 3:
                    tensor = tensor[0]
                if tensor.ndim != 2:
                    raise ValueError(f"Mask {name!r} must be 2D, got {tuple(tensor.shape)}")
                tensors.append(tensor.clamp(0.0, 1.0))
                continue
            if isinstance(item, (str, Path)):
                path = Path(item)
                if not path.is_absolute():
                    path = Path(self.dataset_base_path or ".") / path
                if not path.is_file():
                    raise FileNotFoundError(f"Missing region mask {name!r}: {path}")
                with Image.open(path) as opened:
                    image = opened.convert("L").copy()
            elif isinstance(item, Image.Image):
                image = item.convert("L")
            else:
                raise TypeError(f"Unsupported mask {name!r}: {type(item).__name__}")
            tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            tensors.append(tensor.reshape(image.height, image.width).float().div_(255.0))
        return torch.stack(tensors, dim=0).unsqueeze(1).unsqueeze(2)

    def _prepare_data_aliases(self, data):
        if not isinstance(data, dict):
            return data
        if self.tokenlight_mask_image_key == "mask" or self.tokenlight_mask_image_key not in data:
            return data
        aliased = dict(data)
        aliased["mask"] = data[self.tokenlight_mask_image_key]
        return aliased

    def _attach_decoder_inputs(self, inputs, data):
        inputs_shared, inputs_posi, inputs_nega = inputs
        inputs_shared.update(self.decoder_loss_options)
        enabled = {name: weight for name, weight in self.decoder_region_weights.items() if weight != 0.0}
        if enabled:
            batch = int(inputs_shared["input_latents"].shape[0])
            masks = {}
            for name in enabled:
                key = self.decoder_region_keys[name]
                if key not in data:
                    raise KeyError(f"Decoder region {name!r} requires metadata key {key!r}")
                masks[name] = self._load_region_mask_batch(data[key], batch=batch, name=name)
            inputs_shared["tokenlight_rgb_decoder_region_masks"] = masks
            inputs_shared["tokenlight_rgb_decoder_region_weights"] = enabled
        return inputs_shared, inputs_posi, inputs_nega

    def _attach_extra_mask_inputs(self, inputs, data):
        if not self.tokenlight_mask_tokens or not self.tokenlight_extra_mask_image_keys:
            return inputs
        inputs_shared, inputs_posi, inputs_nega = inputs
        batch = int(inputs_shared["input_latents"].shape[0])
        cached = [
            data.get(f"_tokenlight_extra_mask_latents_{index}")
            for index in range(len(self.tokenlight_extra_mask_image_keys))
        ]
        if all(value is not None for value in cached):
            inputs_shared["tokenlight_extra_mask_latents"] = [
                self._stack_cached_latents(value, batch=batch, name=f"extra_mask_{index}")
                for index, value in enumerate(cached)
            ]
            return inputs_shared, inputs_posi, inputs_nega
        extra_latents = []
        for key in self.tokenlight_extra_mask_image_keys:
            values = data.get(key)
            if not isinstance(values, list) or len(values) != batch:
                raise KeyError(f"Extra VAE mask key {key!r} is missing or has the wrong batch size")
            mask_videos = [base._as_frames(item) for item in values]
            extra_latents.append(self._encode_video_latents_batched(mask_videos, inputs_shared))
        inputs_shared["tokenlight_extra_mask_latents"] = extra_latents
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        data = self._prepare_data_aliases(data)
        if inputs is None and self._has_cached_latents(data):
            prepared = self._attach_extra_mask_inputs(self._cached_batched_inputs(data), data)
            prepared = self._attach_decoder_inputs(prepared, data)
            return self.task_to_loss[self.task](self.pipe, *prepared)
        if inputs is None and self._is_collated_batch(data):
            prepared = self._attach_extra_mask_inputs(self._batched_inputs(data), data)
            prepared = self._attach_decoder_inputs(prepared, data)
            return self.task_to_loss[self.task](self.pipe, *prepared)
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        inputs = self._prepare_tokenlight_inputs(inputs, data)
        inputs = self._attach_extra_mask_inputs(inputs, data)
        inputs = self._attach_decoder_inputs(inputs, data)
        return self.task_to_loss[self.task](self.pipe, *inputs)


def safe_parser(train_mode: str) -> argparse.ArgumentParser:
    parser = base.wan_parser(train_mode)
    parser.add_argument("--train_mode", choices=("single", "zero3"), default=train_mode)
    parser.add_argument("--mixed_precision", choices=("no", "bf16"), default="bf16" if train_mode == "single" else None)
    parser.add_argument(
        "--trainable_parameter_dtype",
        choices=("auto", "fp32", "bf16"),
        default="auto",
        help="auto uses FP32 for plain AdamW and BF16 for DeepSpeed ZeRO-3.",
    )
    parser.add_argument(
        "--tokenlight_extra_mask_latent_cache_dirs",
        default="",
        help="Comma-separated VAE cache directories matching tokenlight_extra_mask_image_keys.",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--tokenlight_rgb_latent_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--tokenlight_rgb_decoder_loss_weight",
        type=float,
        default=0.0,
        help="Outer pixel-loss coefficient alpha; zero preserves the original latent-only path.",
    )
    parser.add_argument("--tokenlight_rgb_decoder_transform", choices=DECODER_TRANSFORMS, default="rgb")
    parser.add_argument(
        "--tokenlight_decoder_loss_type",
        choices=DECODER_LOSS_TYPES,
        default="l1",
        help="Pointwise decoded-flow loss; L1 matches the current-timestep reference objective.",
    )
    parser.add_argument("--tokenlight_decoder_luminance_eps", type=float, default=1e-3)
    parser.add_argument("--tokenlight_decoder_charbonnier_eps", type=float, default=1e-3)
    parser.add_argument("--tokenlight_rgb_decoder_full_loss_weight", type=float, default=1.0)
    parser.add_argument("--tokenlight_rgb_decoder_shadow_loss_weight", type=float, default=0.0)
    parser.add_argument("--tokenlight_rgb_decoder_direct_loss_weight", type=float, default=0.0)
    parser.add_argument("--tokenlight_rgb_decoder_shadow_mask_key", default="shadow_mask")
    parser.add_argument("--tokenlight_rgb_decoder_direct_mask_key", default="inf_mask")
    parser.add_argument("--tokenlight_decoder_min_region_fraction", type=float, default=0.1)
    parser.add_argument("--tokenlight_decoder_max_samples_per_batch", type=int, default=1)
    parser.add_argument(
        "--tokenlight_decoder_gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--tokenlight_decoder_warmup_steps", type=int, default=1000)
    parser.add_argument("--tokenlight_decoder_ramp_steps", type=int, default=1000)
    # Accepted for compatibility with the existing decoder configs.  This
    # implementation uses full decode + activation checkpointing instead.
    parser.add_argument("--tokenlight_decoder_tiled", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--dataset_reject_metadata_path", default=None)
    parser.add_argument("--tokenlight_mask_image_key", default="mask")
    parser.add_argument(
        "--tokenlight_extra_mask_image_keys",
        default="",
        help="Comma-separated additional mask image keys, each encoded as an independent VAE mask prefix.",
    )
    parser.add_argument("--tokenlight_mask_latent_cache_dir", default=None)
    parser.add_argument("--save_full_checkpoints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save_full_checkpoints_final_only", action=argparse.BooleanOptionalAction, default=False)
    return parser


def _validate_safe_args(args, raw_config: Mapping[str, Any], parser: argparse.ArgumentParser) -> None:
    weights = {
        "tokenlight_rgb_latent_loss_weight": args.tokenlight_rgb_latent_loss_weight,
        "tokenlight_rgb_decoder_loss_weight": args.tokenlight_rgb_decoder_loss_weight,
        "tokenlight_rgb_decoder_full_loss_weight": args.tokenlight_rgb_decoder_full_loss_weight,
        "tokenlight_rgb_decoder_shadow_loss_weight": args.tokenlight_rgb_decoder_shadow_loss_weight,
        "tokenlight_rgb_decoder_direct_loss_weight": args.tokenlight_rgb_decoder_direct_loss_weight,
    }
    for name, value in weights.items():
        if not math.isfinite(float(value)) or float(value) < 0:
            parser.error(f"{name} must be finite and non-negative, got {value}")
    if args.tokenlight_rgb_latent_loss_weight == 0 and args.tokenlight_rgb_decoder_loss_weight == 0:
        parser.error("latent and decoder outer weights cannot both be zero")
    if args.tokenlight_rgb_decoder_loss_weight > 0:
        inner = (
            args.tokenlight_rgb_decoder_full_loss_weight
            + args.tokenlight_rgb_decoder_shadow_loss_weight
            + args.tokenlight_rgb_decoder_direct_loss_weight
        )
        if inner <= 0:
            parser.error("decoder is enabled but full/shadow/direct weights are all zero")
        if args.task not in {"sft", "sft:train"}:
            parser.error("decoder loss is supported only for sft training")
        if getattr(args, "enable_model_cpu_offload", False) or args.offload_models:
            parser.error("safe decoder loss currently does not support model/offload_models")
    for name in ("tokenlight_decoder_luminance_eps", "tokenlight_decoder_charbonnier_eps"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be finite and positive")
    if not 0 <= float(args.tokenlight_decoder_min_region_fraction) <= 1:
        parser.error("tokenlight_decoder_min_region_fraction must be in [0, 1]")
    if int(args.tokenlight_decoder_max_samples_per_batch) < 1:
        parser.error("tokenlight_decoder_max_samples_per_batch must be >= 1")
    if int(args.tokenlight_decoder_warmup_steps) < 0 or int(args.tokenlight_decoder_ramp_steps) < 0:
        parser.error("decoder warmup/ramp steps must be non-negative")
    if not math.isfinite(float(args.max_grad_norm)) or float(args.max_grad_norm) < 0:
        parser.error("max_grad_norm must be finite and non-negative")
    if bool(args.tokenlight_decoder_tiled):
        parser.error("tiled decoder backward is disabled; use decoder_gradient_checkpointing instead")

    # JSON defaults bypass argparse type/choice checks, so validate them here.
    if args.tokenlight_rgb_decoder_transform not in DECODER_TRANSFORMS:
        parser.error(f"invalid decoder transform: {args.tokenlight_rgb_decoder_transform}")
    if args.tokenlight_decoder_loss_type not in DECODER_LOSS_TYPES:
        parser.error(f"invalid decoder loss type: {args.tokenlight_decoder_loss_type}")

    known = base._parser_destinations(parser)
    unknown_loss = sorted(key for key in raw_config.get("loss", {}) if key not in known)
    if unknown_loss:
        parser.error(f"unknown key(s) in loss config: {', '.join(unknown_loss)}")
    if args.save_full_checkpoints or args.save_full_checkpoints_final_only:
        print("Warning: this isolated trainer keeps the base inference-weight logger; full-state flags are ignored.")


def parse_safe_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--train_mode", choices=("single", "zero3"), default="single")
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    config_path = pre_args.config or os.environ.get(
        "TOKENLIGHT_TRAIN_CONFIG",
        base._default_config_for_mode(pre_args.train_mode),
    )
    raw_config = base._load_json_config(config_path)
    parser = safe_parser(pre_args.train_mode)
    base._apply_config_defaults(parser, raw_config)
    parser.set_defaults(config=config_path, train_mode=pre_args.train_mode)
    args = parser.parse_args()
    raw_config = base._load_json_config(args.config)
    _validate_safe_args(args, raw_config, parser)
    base._resolve_weight_paths(args)
    base._append_timestamp_to_output_path(args)
    return args, raw_config, raw_config


def _is_deepspeed(accelerator) -> bool:
    return getattr(getattr(accelerator, "state", None), "deepspeed_plugin", None) is not None


def _select_trainable_dtype(args, accelerator) -> torch.dtype:
    requested = str(args.trainable_parameter_dtype)
    if requested == "fp32":
        return torch.float32
    if requested == "bf16":
        if not _is_deepspeed(accelerator):
            raise ValueError(
                "BF16 trainable parameters with plain torch AdamW have no FP32 master weights. "
                "Use --trainable_parameter_dtype fp32 (or auto) for single GPU."
            )
        return torch.bfloat16
    return torch.bfloat16 if _is_deepspeed(accelerator) else torch.float32


def launch_safe_training_task(accelerator, dataset, model, model_logger, args=None, **kwargs):
    del kwargs
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    decoder_enabled = float(args.tokenlight_rgb_decoder_loss_weight) > 0
    if decoder_enabled and accelerator.is_main_process:
        print(
            "Decoder objective: "
            "D(x0 + sigma*v_pred) vs D(x0 + sigma*v_target); "
            f"loss={args.tokenlight_decoder_loss_type}, "
            f"alpha={args.tokenlight_rgb_decoder_loss_weight}, "
            f"full={args.tokenlight_rgb_decoder_full_loss_weight}, "
            f"shadow={args.tokenlight_rgb_decoder_shadow_loss_weight}, "
            f"direct={args.tokenlight_rgb_decoder_direct_loss_weight}, "
            f"warmup={args.tokenlight_decoder_warmup_steps}, "
            f"ramp={args.tokenlight_decoder_ramp_steps}"
        )
    if decoder_enabled and not _is_deepspeed(accelerator) and batch_size > 1:
        print(
            "Warning: decoder loss on a single GPU is safest with batch_size=1; "
            "only decoder_max_samples_per_batch samples are decoded."
        )

    trainable_dtype = _select_trainable_dtype(args, accelerator)
    before = base._coerce_trainable_parameter_dtype(model, trainable_dtype)
    after = base._trainable_dtype_counts(model)
    if accelerator.is_main_process:
        print(f"Trainable dtype: target={trainable_dtype}, before={before}, after={after}")

    optimizer_class = base.get_optimizer_class(args.customized_optimizer)
    trainable_parameters = base._trainable_parameters(model)
    optimizer = optimizer_class(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    task_batch = base._parse_balanced_task_batch(getattr(args, "balanced_task_batch", None))
    dataloader_kwargs = base._dataloader_runtime_kwargs(
        args,
        dataset,
        num_workers=args.dataset_num_workers,
        accelerator=accelerator,
    )
    if task_batch is None:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=base._collate_tokenlight_batch,
            drop_last=batch_size > 1,
            **dataloader_kwargs,
        )
    else:
        if sum(task_batch.values()) != batch_size:
            raise ValueError(f"balanced_task_batch sums to {sum(task_batch.values())}, batch={batch_size}")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=base.BalancedTaskBatchSampler(
                dataset.data,
                task_batch,
                seed=int(getattr(args, "balanced_batch_seed", 0)),
            ),
            collate_fn=base._collate_tokenlight_batch,
            **dataloader_kwargs,
        )

    base._configure_deepspeed_batch_size(accelerator, args, batch_size)
    enable_model_cpu_offload = getattr(args, "enable_model_cpu_offload", False)
    if enable_model_cpu_offload:
        optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
        model.pipe.device = accelerator.device
        offload_manager = base.OffloadTrainingManager(
            model,
            accelerator.device,
            getattr(args, "enable_optimizer_cpu_offload", False),
            getattr(args, "cpu_offload_split_threshold", None),
        )
    else:
        model.to(device=accelerator.device)
        model, optimizer, dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )
        offload_manager = None

    base.save_training_runtime_snapshot(args, accelerator, model)
    tb_metrics = base.TokenLightTensorBoardMetrics(
        args.output_path,
        enabled=args.enable_tensorboard_log,
    )
    base.initialize_deepspeed_gradient_checkpointing(accelerator)
    optimizer_step = int(getattr(model_logger, "num_steps", 0))
    try:
        for epoch_id in range(int(args.start_epoch), int(args.start_epoch) + int(args.num_epochs)):
            iterator = base.tqdm(dataloader, disable=not accelerator.is_local_main_process)
            for data in iterator:
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.decoder_optimizer_step = optimizer_step
                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        loss = model({}, inputs=data) if dataset.load_from_cache else model(data)
                    accelerator.backward(loss)
                    if offload_manager is not None:
                        offload_manager.after_backward()
                    if accelerator.sync_gradients and not _is_deepspeed(accelerator) and args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(trainable_parameters, float(args.max_grad_norm))
                    optimizer.step()
                    scheduler.step()
                    if accelerator.sync_gradients:
                        metrics = base._collect_train_metrics(accelerator, model, optimizer, loss)
                        optimizer_step += 1
                        base._call_compatible_method(
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
                        tb_metrics.log(accelerator, optimizer_step, metrics)
                        if metrics and "train/loss" in metrics:
                            iterator.set_postfix(loss=f"{metrics['train/loss']:.4f}", step=optimizer_step)
                    optimizer.zero_grad(set_to_none=True)
            if args.save_steps is None:
                base._call_compatible_method(
                    model_logger,
                    "on_epoch_end",
                    accelerator,
                    model,
                    epoch_id,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
        base._call_compatible_method(
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


def main() -> None:
    args, raw_config, merged_config = parse_safe_args()
    args.loss_impl_version = LOSS_IMPL_VERSION
    args.decoder_pixel_formula = "D(x0_plus_sigma_v_pred)_vs_D(x0_plus_sigma_v_target)"
    args.decoder_scheduler_weighting = "separate_from_velocity_weight"
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

    dataset = build_safe_dataset(args)
    model = TokenLightSafeDecoderTrainingModule(
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
        tokenlight_rgb_latent_loss_weight=args.tokenlight_rgb_latent_loss_weight,
        tokenlight_rgb_decoder_loss_weight=args.tokenlight_rgb_decoder_loss_weight,
        tokenlight_rgb_decoder_transform=args.tokenlight_rgb_decoder_transform,
        tokenlight_decoder_loss_type=args.tokenlight_decoder_loss_type,
        tokenlight_decoder_luminance_eps=args.tokenlight_decoder_luminance_eps,
        tokenlight_decoder_charbonnier_eps=args.tokenlight_decoder_charbonnier_eps,
        tokenlight_rgb_decoder_full_loss_weight=args.tokenlight_rgb_decoder_full_loss_weight,
        tokenlight_rgb_decoder_shadow_loss_weight=args.tokenlight_rgb_decoder_shadow_loss_weight,
        tokenlight_rgb_decoder_direct_loss_weight=args.tokenlight_rgb_decoder_direct_loss_weight,
        tokenlight_rgb_decoder_shadow_mask_key=args.tokenlight_rgb_decoder_shadow_mask_key,
        tokenlight_rgb_decoder_direct_mask_key=args.tokenlight_rgb_decoder_direct_mask_key,
        tokenlight_decoder_min_region_fraction=args.tokenlight_decoder_min_region_fraction,
        tokenlight_decoder_max_samples_per_batch=args.tokenlight_decoder_max_samples_per_batch,
        tokenlight_decoder_gradient_checkpointing=args.tokenlight_decoder_gradient_checkpointing,
        tokenlight_decoder_warmup_steps=args.tokenlight_decoder_warmup_steps,
        tokenlight_decoder_ramp_steps=args.tokenlight_decoder_ramp_steps,
        tokenlight_mask_image_key=args.tokenlight_mask_image_key,
        tokenlight_extra_mask_image_keys=args.tokenlight_extra_mask_image_keys,
        tokenlight_extra_mask_latent_cache_dirs=args.tokenlight_extra_mask_latent_cache_dirs,
    )
    model_logger = base._make_model_logger(args)
    launcher_map = {
        "sft:data_process": base.launch_data_process_task,
        "direct_distill:data_process": base.launch_data_process_task,
        "sft": launch_safe_training_task,
        "sft:train": launch_safe_training_task,
        "direct_distill": launch_safe_training_task,
        "direct_distill:train": launch_safe_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)


if __name__ == "__main__":
    main()
