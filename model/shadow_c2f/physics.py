"""Chunked physics-grounded coarse shadow priors.

This module implements the soft angular prior from arXiv:2512.06174v2 at a
default coarse resolution of 60x60 for TokenLight's 480x480 pipeline.  It also
supports a finite point light by deriving a separate shadow-flow direction for
every occluder point.

All geometry and lights must already be in the same OpenCV camera coordinate
frame and the same physical scale.  Light directions point *toward* the source;
the shadow flow therefore uses their negative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from .coordinates import normalize_vectors


PriorMode = Literal["soft", "hard"]


@dataclass(frozen=True)
class PhysicsPriorConfig:
    """Configuration for directional and point-light shadow priors.

    Attributes:
        coarse_size: Output ``(height, width)``.  ``(60,60)`` preserves an
            integer 8x relation with the repository's 480x480 masks.
        angular_tolerance_degrees: Cone half-angle around the shadow-flow ray.
        temperature: Sigmoid temperature for the differentiable soft prior.
        object_chunk_size: Maximum occluder points compared at once.
        receiver_chunk_size: Maximum receiver points compared at once.
        mask_threshold: Threshold used to select object and receiver pixels.
        gaussian_blur_sigma: Gaussian sigma applied only to the soft prior.
            Set to zero to disable smoothing.
        eps: Numerical tolerance for vector lengths and light validation.
    """

    coarse_size: tuple[int, int] = (60, 60)
    angular_tolerance_degrees: float = 10.0
    temperature: float = 0.05
    object_chunk_size: int = 128
    receiver_chunk_size: int = 1024
    mask_threshold: float = 0.5
    gaussian_blur_sigma: float = 0.8
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if len(self.coarse_size) != 2 or min(int(value) for value in self.coarse_size) <= 0:
            raise ValueError(f"coarse_size must contain two positive integers, got {self.coarse_size}")
        if not 0.0 < self.angular_tolerance_degrees < 90.0:
            raise ValueError("angular_tolerance_degrees must be in (0, 90)")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.object_chunk_size <= 0 or self.receiver_chunk_size <= 0:
            raise ValueError("chunk sizes must be positive")
        if not 0.0 <= self.mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be in [0, 1]")
        if self.gaussian_blur_sigma < 0:
            raise ValueError("gaussian_blur_sigma must be non-negative")
        if self.eps <= 0:
            raise ValueError("eps must be positive")


def gaussian_blur_mask(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply a differentiable Gaussian blur to a ``[B,C,H,W]`` tensor."""

    if mask.ndim != 4:
        raise ValueError(f"mask must have shape [B,C,H,W], got {tuple(mask.shape)}")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    if sigma == 0:
        return mask
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    coordinates = torch.arange(-radius, radius + 1, device=mask.device, dtype=mask.dtype)
    kernel_1d = torch.exp(-0.5 * (coordinates / float(sigma)).square())
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.expand(mask.shape[1], 1, -1, -1)
    padded = F.pad(mask, (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(padded, kernel, groups=mask.shape[1])


def _validate_spatial_inputs(
    point_map: torch.Tensor,
    object_mask: torch.Tensor,
    receiver_mask: torch.Tensor | None,
    point_valid_mask: torch.Tensor | None,
) -> None:
    if point_map.ndim != 4 or point_map.shape[1] != 3:
        raise ValueError(f"point_map must have shape [B,3,H,W], got {tuple(point_map.shape)}")
    if not torch.is_floating_point(point_map):
        raise TypeError(f"point_map must be floating point, got {point_map.dtype}")
    expected_mask_shape = (point_map.shape[0], 1, point_map.shape[2], point_map.shape[3])
    if tuple(object_mask.shape) != expected_mask_shape:
        raise ValueError(f"object_mask must have shape {expected_mask_shape}, got {tuple(object_mask.shape)}")
    if receiver_mask is not None and tuple(receiver_mask.shape) != expected_mask_shape:
        raise ValueError(f"receiver_mask must have shape {expected_mask_shape}, got {tuple(receiver_mask.shape)}")
    if point_valid_mask is not None and tuple(point_valid_mask.shape) != expected_mask_shape:
        raise ValueError(
            f"point_valid_mask must have shape {expected_mask_shape}, got {tuple(point_valid_mask.shape)}"
        )


def _resize_mask(mask: torch.Tensor, size: tuple[int, int], threshold: float) -> torch.Tensor:
    resized = F.interpolate(mask.float(), size=size, mode="nearest")
    return resized >= threshold


def _prepare_coarse_geometry(
    point_map: torch.Tensor,
    object_mask: torch.Tensor,
    receiver_mask: torch.Tensor | None,
    point_valid_mask: torch.Tensor | None,
    config: PhysicsPriorConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_spatial_inputs(point_map, object_mask, receiver_mask, point_valid_mask)
    object_mask = object_mask.to(device=point_map.device)
    if receiver_mask is not None:
        receiver_mask = receiver_mask.to(device=point_map.device)
    if point_valid_mask is not None:
        point_valid_mask = point_valid_mask.to(device=point_map.device)
    size = (int(config.coarse_size[0]), int(config.coarse_size[1]))
    finite = torch.isfinite(point_map).all(dim=1, keepdim=True)
    if point_valid_mask is not None:
        finite = finite & (point_valid_mask >= config.mask_threshold)
    sanitized = torch.where(finite.expand_as(point_map), point_map, torch.zeros_like(point_map))
    if tuple(point_map.shape[-2:]) != size:
        finite_weight = F.interpolate(finite.float(), size=size, mode="bilinear", align_corners=False)
        sanitized = F.interpolate(sanitized, size=size, mode="bilinear", align_corners=False)
        sanitized = sanitized / finite_weight.clamp_min(config.eps)
        finite = finite_weight > config.eps
    objects = _resize_mask(object_mask, size, config.mask_threshold) & finite
    if receiver_mask is None:
        receivers = finite & ~objects
    else:
        receivers = _resize_mask(receiver_mask, size, config.mask_threshold) & finite & ~objects
    return sanitized, objects, receivers


def _validate_light_batch(
    light: torch.Tensor,
    batch: int,
    name: str,
    eps: float,
    *,
    require_nonzero: bool,
) -> torch.Tensor:
    if light.ndim == 1:
        if light.shape[0] != 3:
            raise ValueError(f"{name} must have shape [3] or [B,3], got {tuple(light.shape)}")
        light = light.unsqueeze(0).expand(batch, -1)
    if light.ndim != 2 or light.shape != (batch, 3):
        raise ValueError(f"{name} must have shape [3] or [{batch},3], got {tuple(light.shape)}")
    if not torch.is_floating_point(light):
        raise TypeError(f"{name} must be floating point, got {light.dtype}")
    if not bool(torch.isfinite(light).all()):
        raise ValueError(f"{name} contains non-finite values")
    if require_nonzero and bool((torch.linalg.vector_norm(light, dim=-1) <= eps).any()):
        raise ValueError(f"{name} contains a zero-length vector")
    return light


def _single_prior(
    points_hwc: torch.Tensor,
    object_pixels: torch.Tensor,
    receiver_pixels: torch.Tensor,
    *,
    config: PhysicsPriorConfig,
    mode: PriorMode,
    directional_shadow_flow: torch.Tensor | None,
    point_light_position: torch.Tensor | None,
) -> torch.Tensor:
    height, width, _ = points_hwc.shape
    flattened = points_hwc.reshape(-1, 3)
    object_indices = torch.nonzero(object_pixels.reshape(-1), as_tuple=False).flatten()
    receiver_indices = torch.nonzero(receiver_pixels.reshape(-1), as_tuple=False).flatten()
    output = points_hwc.new_zeros(height * width)
    if object_indices.numel() == 0 or receiver_indices.numel() == 0:
        return output.reshape(1, height, width)

    object_points = flattened.index_select(0, object_indices)
    receiver_points = flattened.index_select(0, receiver_indices)
    if point_light_position is not None:
        object_shadow_flow = normalize_vectors(object_points - point_light_position[None], eps=config.eps)
        valid_object_flow = torch.linalg.vector_norm(object_points - point_light_position[None], dim=-1) > config.eps
    else:
        if directional_shadow_flow is None:
            raise RuntimeError("one light representation must be provided")
        object_shadow_flow = directional_shadow_flow[None].expand(object_points.shape[0], -1)
        valid_object_flow = torch.ones(object_points.shape[0], dtype=torch.bool, device=points_hwc.device)

    cosine_threshold = math.cos(math.radians(config.angular_tolerance_degrees))
    receiver_results: list[torch.Tensor] = []
    for receiver_start in range(0, receiver_points.shape[0], config.receiver_chunk_size):
        receiver_chunk = receiver_points[receiver_start : receiver_start + config.receiver_chunk_size]
        if mode == "hard":
            best: torch.Tensor = torch.zeros(receiver_chunk.shape[0], dtype=torch.bool, device=points_hwc.device)
        else:
            best = points_hwc.new_zeros(receiver_chunk.shape[0])

        for object_start in range(0, object_points.shape[0], config.object_chunk_size):
            object_chunk = object_points[object_start : object_start + config.object_chunk_size]
            flow_chunk = object_shadow_flow[object_start : object_start + config.object_chunk_size]
            flow_valid = valid_object_flow[object_start : object_start + config.object_chunk_size]
            displacement = receiver_chunk[None] - object_chunk[:, None]
            distance = torch.linalg.vector_norm(displacement, dim=-1)
            direction = displacement / distance.clamp_min(config.eps)[..., None]
            alignment = (direction * flow_chunk[:, None]).sum(dim=-1)
            valid_pairs = (distance > config.eps) & flow_valid[:, None]

            if mode == "hard":
                candidate = ((alignment >= cosine_threshold) & valid_pairs).any(dim=0)
                best = best | candidate
            else:
                score = torch.sigmoid((alignment - cosine_threshold) / config.temperature)
                score = torch.where(valid_pairs, score, torch.zeros_like(score))
                candidate = score.max(dim=0).values
                best = torch.maximum(best, candidate)
        receiver_results.append(best)

    receiver_values = torch.cat(receiver_results, dim=0)
    if mode == "hard":
        receiver_values = receiver_values.to(dtype=points_hwc.dtype)
    output = output.scatter(0, receiver_indices, receiver_values)
    return output.reshape(1, height, width)


def _shadow_prior_impl(
    point_map: torch.Tensor,
    object_mask: torch.Tensor,
    *,
    receiver_mask: torch.Tensor | None,
    point_valid_mask: torch.Tensor | None,
    config: PhysicsPriorConfig,
    mode: PriorMode,
    light_direction: torch.Tensor | None,
    light_position: torch.Tensor | None,
) -> torch.Tensor:
    if mode not in ("soft", "hard"):
        raise ValueError(f"mode must be 'soft' or 'hard', got {mode!r}")
    points, objects, receivers = _prepare_coarse_geometry(
        point_map,
        object_mask,
        receiver_mask,
        point_valid_mask,
        config,
    )
    batch = points.shape[0]

    if (light_direction is None) == (light_position is None):
        raise ValueError("provide exactly one of light_direction or light_position")
    if light_direction is not None:
        light_direction = _validate_light_batch(
            light_direction,
            batch,
            "light_direction",
            config.eps,
            require_nonzero=True,
        )
        light_direction = normalize_vectors(
            light_direction.to(device=points.device, dtype=points.dtype),
            eps=config.eps,
        )
        shadow_flow = -light_direction
        point_lights = None
    else:
        assert light_position is not None
        light_position = _validate_light_batch(
            light_position,
            batch,
            "light_position",
            config.eps,
            require_nonzero=False,
        )
        point_lights = light_position.to(device=points.device, dtype=points.dtype)
        shadow_flow = None

    priors = []
    for batch_index in range(batch):
        priors.append(
            _single_prior(
                points[batch_index].permute(1, 2, 0),
                objects[batch_index, 0],
                receivers[batch_index, 0],
                config=config,
                mode=mode,
                directional_shadow_flow=None if shadow_flow is None else shadow_flow[batch_index],
                point_light_position=None if point_lights is None else point_lights[batch_index],
            )
        )
    prior = torch.stack(priors, dim=0)
    if mode == "soft" and config.gaussian_blur_sigma > 0:
        prior = gaussian_blur_mask(prior, config.gaussian_blur_sigma).clamp(0.0, 1.0)
        # Smoothing must not leak shadow support onto the occluder or invalid
        # geometry outside the explicit receiver domain.
        prior = prior * receivers.to(dtype=prior.dtype)
    return prior


def directional_shadow_prior(
    point_map: torch.Tensor,
    object_mask: torch.Tensor,
    light_direction: torch.Tensor,
    *,
    receiver_mask: torch.Tensor | None = None,
    point_valid_mask: torch.Tensor | None = None,
    config: PhysicsPriorConfig | None = None,
    mode: PriorMode = "soft",
) -> torch.Tensor:
    """Construct a directional-light shadow prior.

    Args:
        point_map: ``[B,3,H,W]`` OpenCV camera-space points.
        object_mask: ``[B,1,H,W]`` foreground-occluder mask.
        light_direction: ``[3]`` or ``[B,3]`` vectors pointing toward the
            light source.
        receiver_mask: Optional ``[B,1,H,W]`` valid receiver domain.  If
            omitted, every finite non-object point is a receiver.
        point_valid_mask: Optional ``[B,1,H,W]`` geometry-validity mask.  Pass
            this for MoGe caches that store invalid points as finite zeros.
        config: Prior hyperparameters; defaults to :class:`PhysicsPriorConfig`.
        mode: ``"soft"`` gives a differentiable sigmoid/max prior.  ``"hard"``
            gives a detached binary angular-cone test.

    Returns:
        Prior tensor ``[B,1,Hc,Wc]`` in ``[0,1]``.
    """

    config = config or PhysicsPriorConfig()
    if mode == "hard":
        with torch.no_grad():
            return _shadow_prior_impl(
                point_map,
                object_mask,
                receiver_mask=receiver_mask,
                point_valid_mask=point_valid_mask,
                config=config,
                mode=mode,
                light_direction=light_direction,
                light_position=None,
            ).detach()
    return _shadow_prior_impl(
        point_map,
        object_mask,
        receiver_mask=receiver_mask,
        point_valid_mask=point_valid_mask,
        config=config,
        mode=mode,
        light_direction=light_direction,
        light_position=None,
    )


def point_light_shadow_prior(
    point_map: torch.Tensor,
    object_mask: torch.Tensor,
    light_position: torch.Tensor,
    *,
    receiver_mask: torch.Tensor | None = None,
    point_valid_mask: torch.Tensor | None = None,
    config: PhysicsPriorConfig | None = None,
    mode: PriorMode = "soft",
) -> torch.Tensor:
    """Construct a finite point-light shadow prior.

    ``light_position`` has shape ``[3]`` or ``[B,3]`` and must share the point
    map's OpenCV camera frame and physical scale.  Unlike the paper's single
    directional vector, this variant uses the divergent flow
    ``normalize(object_point - light_position)`` for each occluder point.

    ``point_map``, ``object_mask``, optional ``receiver_mask``, and optional
    ``point_valid_mask`` use the same shapes and semantics as
    :func:`directional_shadow_prior`.  ``mode="soft"`` is differentiable with
    respect to point geometry and light position; ``mode="hard"`` returns a
    detached binary prior.
    """

    config = config or PhysicsPriorConfig()
    if mode == "hard":
        with torch.no_grad():
            return _shadow_prior_impl(
                point_map,
                object_mask,
                receiver_mask=receiver_mask,
                point_valid_mask=point_valid_mask,
                config=config,
                mode=mode,
                light_direction=None,
                light_position=light_position,
            ).detach()
    return _shadow_prior_impl(
        point_map,
        object_mask,
        receiver_mask=receiver_mask,
        point_valid_mask=point_valid_mask,
        config=config,
        mode=mode,
        light_direction=None,
        light_position=light_position,
    )


__all__ = [
    "PhysicsPriorConfig",
    "PriorMode",
    "directional_shadow_prior",
    "gaussian_blur_mask",
    "point_light_shadow_prior",
]
