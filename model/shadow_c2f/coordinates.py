"""Coordinate helpers for the standalone shadow-mask refiner.

The physics-prior functions in :mod:`model.shadow_c2f.physics` consume OpenCV
camera-space geometry: ``x`` points right, ``y`` points down, and ``z`` points
forward.  They deliberately do not parse TokenLight metadata because existing
datasets use renderer-specific camera and light scales.  Convert those values
explicitly before constructing a prior.
"""

from __future__ import annotations

from typing import Tuple

import torch


def _require_last_dimension(tensor: torch.Tensor, size: int, name: str) -> None:
    if tensor.ndim == 0 or tensor.shape[-1] != size:
        raise ValueError(f"{name} must end in dimension {size}, got {tuple(tensor.shape)}")


def _require_transform(transform: torch.Tensor, name: str = "transform") -> None:
    if transform.ndim not in (2, 3) or transform.shape[-2:] != (4, 4):
        raise ValueError(f"{name} must have shape [4,4] or [B,4,4], got {tuple(transform.shape)}")


def normalize_vectors(vectors: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize vectors along their final dimension.

    Zero-length vectors remain zero.  Callers that require a valid direction
    should validate the norm before invoking downstream geometry operations.

    Args:
        vectors: Tensor with shape ``[..., D]``.
        eps: Minimum divisor used for numerical stability.

    Returns:
        Tensor with the same shape, dtype, and device as ``vectors``.
    """

    if not torch.is_floating_point(vectors):
        raise TypeError(f"vectors must be floating point, got {vectors.dtype}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    norm = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    normalized = vectors / norm.clamp_min(eps)
    return torch.where(norm > eps, normalized, torch.zeros_like(normalized))


def blender_camera_to_opencv_matrix(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the homogeneous Blender-camera to OpenCV-camera transform.

    Blender camera coordinates use ``x`` right, ``y`` up, and look along
    ``-z``.  OpenCV camera coordinates use ``x`` right, ``y`` down, and ``z``
    forward, so the conversion is ``diag(1, -1, -1, 1)``.
    """

    return torch.diag(torch.tensor((1.0, -1.0, -1.0, 1.0), dtype=dtype, device=device))


def blender_world_to_opencv_camera_matrix(world_to_blender_camera: torch.Tensor) -> torch.Tensor:
    """Convert a Blender world-to-camera matrix to OpenCV convention.

    Args:
        world_to_blender_camera: ``[4,4]`` or ``[B,4,4]`` affine transform,
            normally the inverse of Blender's camera ``matrix_world``.

    Returns:
        A transform of the same shape mapping world points directly into
        OpenCV camera coordinates.
    """

    _require_transform(world_to_blender_camera, "world_to_blender_camera")
    if not torch.is_floating_point(world_to_blender_camera):
        raise TypeError("world_to_blender_camera must be floating point")
    conversion = blender_camera_to_opencv_matrix(
        dtype=world_to_blender_camera.dtype,
        device=world_to_blender_camera.device,
    )
    return torch.matmul(conversion, world_to_blender_camera)


def transform_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Apply an affine homogeneous transform to 3D points.

    Args:
        points: ``[...,3]`` for an unbatched transform, or ``[B,...,3]`` for
            a batched transform.
        transform: ``[4,4]`` or ``[B,4,4]``.  Batched transforms require the
            first dimension of ``points`` to be the same batch size.

    Returns:
        Transformed points with the same shape as ``points``.
    """

    _require_last_dimension(points, 3, "points")
    _require_transform(transform)
    if not torch.is_floating_point(points) or not torch.is_floating_point(transform):
        raise TypeError("points and transform must be floating point")
    transform = transform.to(device=points.device, dtype=points.dtype)

    if transform.ndim == 2:
        ones = torch.ones_like(points[..., :1])
        homogeneous = torch.cat((points, ones), dim=-1)
        transformed = torch.matmul(homogeneous, transform.transpose(0, 1))
        return transformed[..., :3]

    if points.ndim < 2 or points.shape[0] != transform.shape[0]:
        raise ValueError(
            "batched transform requires points shaped [B,...,3] with matching B; "
            f"got points={tuple(points.shape)}, transform={tuple(transform.shape)}"
        )
    batch = points.shape[0]
    flattened = points.reshape(batch, -1, 3)
    ones = torch.ones_like(flattened[..., :1])
    homogeneous = torch.cat((flattened, ones), dim=-1)
    transformed = torch.bmm(homogeneous, transform.transpose(1, 2))[..., :3]
    return transformed.reshape_as(points)


def transform_directions(
    directions: torch.Tensor,
    transform: torch.Tensor,
    *,
    normalize: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Transform 3D directions using only the linear part of a transform.

    Translation is ignored.  Set ``normalize=True`` for light directions when
    the transform may contain scale.
    """

    _require_last_dimension(directions, 3, "directions")
    _require_transform(transform)
    if not torch.is_floating_point(directions) or not torch.is_floating_point(transform):
        raise TypeError("directions and transform must be floating point")
    transform = transform.to(device=directions.device, dtype=directions.dtype)
    rotation = transform[..., :3, :3]

    if transform.ndim == 2:
        output = torch.matmul(directions, rotation.transpose(0, 1))
    else:
        if directions.ndim < 2 or directions.shape[0] != transform.shape[0]:
            raise ValueError(
                "batched transform requires directions shaped [B,...,3] with matching B; "
                f"got directions={tuple(directions.shape)}, transform={tuple(transform.shape)}"
            )
        batch = directions.shape[0]
        flattened = directions.reshape(batch, -1, 3)
        output = torch.bmm(flattened, rotation.transpose(1, 2)).reshape_as(directions)
    return normalize_vectors(output, eps=eps) if normalize else output


def point_light_direction(
    light_position: torch.Tensor,
    reference_position: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return unit vectors pointing from reference positions to a point light.

    ``light_position`` and ``reference_position`` must be broadcast-compatible
    tensors ending in three coordinates and must use the same coordinate frame
    and physical scale.
    """

    _require_last_dimension(light_position, 3, "light_position")
    _require_last_dimension(reference_position, 3, "reference_position")
    return normalize_vectors(light_position - reference_position, eps=eps)


def world_point_light_to_opencv_camera(
    light_position_world: torch.Tensor,
    reference_position_world: torch.Tensor,
    world_to_blender_camera: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transform a world-space point light and derive its camera-space direction.

    Returns ``(light_position_cv, light_direction_cv)``.  The direction points
    from ``reference_position_world`` toward the source, matching the paper's
    light-vector convention.
    """

    world_to_opencv = blender_world_to_opencv_camera_matrix(world_to_blender_camera)
    light_cv = transform_points(light_position_world, world_to_opencv)
    reference_cv = transform_points(reference_position_world, world_to_opencv)
    return light_cv, point_light_direction(light_cv, reference_cv, eps=eps)


def world_directional_light_to_opencv_camera(
    light_direction_world: torch.Tensor,
    world_to_blender_camera: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Transform source-pointing directional-light vectors into camera space."""

    world_to_opencv = blender_world_to_opencv_camera_matrix(world_to_blender_camera)
    return transform_directions(light_direction_world, world_to_opencv, normalize=True, eps=eps)


def broadcast_light_map(
    light_direction: torch.Tensor,
    spatial_size: tuple[int, int],
    *,
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Broadcast ``[3]`` or ``[B,3]`` light directions to ``[B,3,H,W]``."""

    _require_last_dimension(light_direction, 3, "light_direction")
    if light_direction.ndim == 1:
        light_direction = light_direction.unsqueeze(0)
    if light_direction.ndim != 2:
        raise ValueError(f"light_direction must have shape [3] or [B,3], got {tuple(light_direction.shape)}")
    height, width = (int(spatial_size[0]), int(spatial_size[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"spatial_size must be positive, got {spatial_size}")
    if normalize:
        light_direction = normalize_vectors(light_direction, eps=eps)
    return light_direction[:, :, None, None].expand(-1, -1, height, width)


__all__ = [
    "blender_camera_to_opencv_matrix",
    "blender_world_to_opencv_camera_matrix",
    "broadcast_light_map",
    "normalize_vectors",
    "point_light_direction",
    "transform_directions",
    "transform_points",
    "world_directional_light_to_opencv_camera",
    "world_point_light_to_opencv_camera",
]
