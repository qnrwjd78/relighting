from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F


DECODER_TRANSFORMS = ("rgb", "luminance", "illuminance", "log_luminance")
DECODER_LOSS_TYPES = ("mse", "l1", "charbonnier")


def decode_latents(
    pipe,
    latents: torch.Tensor,
    *,
    tiled: bool = False,
    tile_size: Any = (30, 52),
    tile_stride: Any = (15, 26),
) -> torch.Tensor:
    """Decode latents while preserving gradients with respect to ``latents``."""
    pipe.load_models_to_device(["vae"])
    return pipe.vae.decode(
        latents.to(dtype=pipe.torch_dtype, device=pipe.device),
        device=pipe.device,
        tiled=bool(tiled),
        tile_size=tile_size,
        tile_stride=tile_stride,
    )


def transform_decoded_video(video: torch.Tensor, transform: str, eps: float = 1e-3) -> torch.Tensor:
    transform = str(transform).lower().replace("-", "_")
    if transform not in DECODER_TRANSFORMS:
        raise ValueError(f"Unsupported decoder transform {transform!r}; choose from {DECODER_TRANSFORMS}")
    if transform == "rgb":
        return video.float()

    # Wan VAE decoder output is in [-1, 1]. Luminance is defined in linear unit RGB.
    # Do not clamp the prediction: values outside the display range should still
    # receive a useful gradient. Only log-luminance needs a positive floor.
    rgb = (video.float() + 1.0) * 0.5
    weights = rgb.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 3, 1, 1, 1)
    luminance = (rgb[:, :3] * weights).sum(dim=1, keepdim=True)
    if transform in {"luminance", "illuminance"}:
        return luminance
    return torch.log(luminance.clamp_min(float(eps)))


def transform_unit_target_video(video: torch.Tensor, transform: str, eps: float = 1e-3) -> torch.Tensor:
    """Transform a raw PNG target represented in the unit ``[0, 1]`` range."""
    transform = str(transform).lower().replace("-", "_")
    if transform not in DECODER_TRANSFORMS:
        raise ValueError(f"Unsupported decoder transform {transform!r}; choose from {DECODER_TRANSFORMS}")
    unit = video.float()
    if transform == "rgb":
        # Wan VAE predictions decode to [-1, 1], so compare RGB-like maps in
        # that same range. This also preserves the usual signed normal-map
        # interpretation after mapping a normal PNG from [0, 1].
        return unit * 2.0 - 1.0
    weights = unit.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 3, 1, 1, 1)
    luminance = (unit[:, :3] * weights).sum(dim=1, keepdim=True)
    if transform in {"luminance", "illuminance"}:
        return luminance
    return torch.log(luminance.clamp_min(float(eps)))


def pointwise_decoder_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_type: str = "mse",
    charbonnier_eps: float = 1e-3,
) -> torch.Tensor:
    loss_type = str(loss_type).lower()
    if loss_type == "mse":
        return (pred.float() - target.float()).square()
    if loss_type == "l1":
        return (pred.float() - target.float()).abs()
    if loss_type == "charbonnier":
        delta = pred.float() - target.float()
        return (delta.square() + float(charbonnier_eps) ** 2).sqrt()
    raise ValueError(f"Unsupported decoder loss type {loss_type!r}; choose from {DECODER_LOSS_TYPES}")


def decoder_space_loss(
    pipe,
    *,
    pred_x0_latents: torch.Tensor,
    target_latents: torch.Tensor | None = None,
    target_video: torch.Tensor | None = None,
    transform: str = "rgb",
    loss_type: str = "mse",
    sample_weights: Any = None,
    tiled: bool = False,
    tile_size: Any = (30, 52),
    tile_stride: Any = (15, 26),
    luminance_eps: float = 1e-3,
    charbonnier_eps: float = 1e-3,
    full_loss_weight: float = 1.0,
    region_masks: Mapping[str, torch.Tensor] | None = None,
    region_weights: Mapping[str, float] | None = None,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compare decoded x0 against either raw PNG pixels or decoded target latents.

    ``target_video`` must be a ``[B,C,T,H,W]`` tensor in the unit ``[0,1]``
    range. When supplied, the target VAE decode is skipped entirely.
    """
    if (target_latents is None) == (target_video is None):
        raise ValueError("Provide exactly one of target_latents or target_video")
    if target_latents is not None and pred_x0_latents.shape != target_latents.shape:
        raise ValueError(
            "Decoder-space latent shape mismatch: "
            f"pred={tuple(pred_x0_latents.shape)} target={tuple(target_latents.shape)}"
        )
    if target_video is not None:
        if target_video.ndim != 5:
            raise ValueError(f"Raw decoder target must have shape [B,C,T,H,W], got {tuple(target_video.shape)}")
        if int(target_video.shape[0]) != int(pred_x0_latents.shape[0]):
            raise ValueError(
                f"Raw decoder target batch mismatch: {target_video.shape[0]} != {pred_x0_latents.shape[0]}"
            )
        if int(target_video.shape[1]) < 3:
            raise ValueError(f"Raw decoder target must have at least 3 channels, got {target_video.shape[1]}")

    weights = None
    if sample_weights is not None:
        weights = torch.as_tensor(sample_weights, device=pred_x0_latents.device, dtype=torch.float32).flatten()
        if weights.numel() == 1:
            weights = weights.expand(int(pred_x0_latents.shape[0]))
        if weights.numel() != int(pred_x0_latents.shape[0]):
            raise ValueError(f"Expected one decoder sample weight or {pred_x0_latents.shape[0]}, got {weights.numel()}")
        if bool((weights < 0).any()):
            raise ValueError("Decoder sample weights must be non-negative")
        active = weights > 0
        if not bool(active.any()):
            return pred_x0_latents.sum() * 0.0
        pred_x0_latents = pred_x0_latents[active]
        if target_latents is not None:
            target_latents = target_latents[active]
        if target_video is not None:
            target_video = target_video[active.to(device=target_video.device)]
        weights = weights[active]

    pred_video = decode_latents(
        pipe,
        pred_x0_latents,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )
    pred_video = transform_decoded_video(pred_video, transform, luminance_eps)
    # The target branch never needs gradients. Prefer raw PNG pixels when
    # available so VAE reconstruction error is not baked into the ground truth.
    with torch.no_grad():
        if target_video is None:
            target_video = decode_latents(
                pipe,
                target_latents,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            target_video = transform_decoded_video(target_video, transform, luminance_eps)
        else:
            target_video = target_video.to(device=pred_video.device, dtype=torch.float32)
            if tuple(target_video.shape[2:]) != tuple(pred_video.shape[2:]):
                target_video = F.interpolate(
                    target_video,
                    size=pred_video.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )
            target_video = transform_unit_target_video(target_video, transform, luminance_eps)
    error = pointwise_decoder_loss(
        pred_video,
        target_video,
        loss_type=loss_type,
        charbonnier_eps=charbonnier_eps,
    )
    if weights is None:
        full_loss = error.mean()
        sample_weight_map = None
    else:
        sample_weight_map = weights.to(device=error.device).view(-1, *([1] * (error.ndim - 1)))
        denominator = sample_weight_map.sum() * error[0].numel()
        full_loss = (error * sample_weight_map).sum().div(denominator.clamp_min(1.0))

    components = {"full": full_loss.detach()}
    total = float(full_loss_weight) * full_loss
    for name, region_weight in (region_weights or {}).items():
        region_weight = float(region_weight)
        if region_weight == 0.0:
            continue
        if region_masks is None or name not in region_masks:
            raise KeyError(f"Decoder region loss `{name}` is enabled but its mask is missing")
        mask = region_masks[name].to(device=error.device, dtype=torch.float32)
        if mask.ndim == 4:
            mask = mask.unsqueeze(2)
        if mask.ndim != 5 or int(mask.shape[0]) != int(error.shape[0]):
            raise ValueError(
                f"Decoder region mask `{name}` must have shape [B,1,T,H,W], got {tuple(mask.shape)}"
            )
        if tuple(mask.shape[2:]) != tuple(error.shape[2:]):
            mask = F.interpolate(mask, size=error.shape[2:], mode="nearest")
        mask = (mask > 0.5).to(dtype=error.dtype)
        if sample_weight_map is not None:
            mask = mask * sample_weight_map
        denominator = mask.sum() * int(error.shape[1])
        if bool(denominator <= 0):
            region_loss = error.sum() * 0.0
        else:
            region_loss = (error * mask).sum() / denominator
        components[name] = region_loss.detach()
        total = total + region_weight * region_loss

    total = total.to(device=pred_x0_latents.device)
    components["total"] = total.detach()
    return (total, components) if return_components else total
