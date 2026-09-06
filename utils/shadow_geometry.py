"""Geometry loading/alignment for the Objaverse shadow C2F pipeline.

All returned dense points use OpenCV camera axes (right, down, forward) in
TokenLight canonical length units.  Target light coordinates from attrs use
TokenLight canonical axes (right, forward, up) and are converted explicitly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


CANONICAL_CAMERA = (0.0, -3.5, 0.0)


def load_binary_mask(path: str | Path, *, size: tuple[int, int] | None = None) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("L")
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.NEAREST)
        array = np.asarray(image, dtype=np.uint8) >= 128
    return torch.from_numpy(array.copy()).unsqueeze(0).float()


def canonical_to_opencv_relative(value: torch.Tensor) -> torch.Tensor:
    """Map canonical absolute XYZ to OpenCV camera coordinates.

    Canonical uses ``(+x right,+y forward,+z up)`` with camera origin
    ``(0,-3.5,0)``.  OpenCV uses ``(+x right,+y down,+z forward)``.
    """

    value = torch.as_tensor(value)
    if value.shape[-1] != 3:
        raise ValueError(f"Expected [...,3], got {tuple(value.shape)}")
    if not torch.is_floating_point(value):
        value = value.to(dtype=torch.float32)
    camera = value.new_tensor(CANONICAL_CAMERA)
    relative = value - camera
    return torch.stack((relative[..., 0], -relative[..., 2], relative[..., 1]), dim=-1)


def opencv_relative_to_canonical(value: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(value)
    if value.shape[-1] != 3:
        raise ValueError(f"Expected [...,3], got {tuple(value.shape)}")
    if not torch.is_floating_point(value):
        value = value.to(dtype=torch.float32)
    camera = value.new_tensor(CANONICAL_CAMERA)
    rotated = torch.stack((value[..., 0], value[..., 2], -value[..., 1]), dim=-1)
    return rotated + camera


def canonical_light_to_opencv(light_position: list[float] | torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(light_position, dtype=torch.float32)
    if tuple(value.shape) != (3,):
        raise ValueError(f"Light position must contain XYZ, got {tuple(value.shape)}")
    return canonical_to_opencv_relative(value)


def _meta_value(meta: Mapping[str, Any], *keys: str) -> Any:
    value: Any = meta
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def reconstruct_oracle_point_map(
    depth_path: str | Path,
    meta_path: str | Path,
    *,
    width: int = 480,
    height: int = 480,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Reconstruct a canonical-scale point map from renderer PBR depth.

    The depth PNG contract is the same one used by ``physical_tasks.data``:
    white encodes the near percentile and black the far percentile.  Renderer
    metric depth is divided by the scene similarity scale before it is applied
    to the normalized canonical camera ray.
    """

    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    with Image.open(depth_path) as source:
        depth_image = source.convert("L")
        if depth_image.size != (width, height):
            depth_image = depth_image.resize((width, height), Image.Resampling.BILINEAR)
        encoded = torch.from_numpy(np.asarray(depth_image, dtype=np.float32).copy()) / 255.0
    depth_range = _meta_value(meta, "common", "pbr_maps", "depth_png_range")
    depth_min = float(depth_range["min_meters"])
    depth_max = float(depth_range["max_meters"])
    canonical_scale = float(_meta_value(meta, "camera", "canonical_scale"))
    if not 0 < canonical_scale < float("inf") or depth_max <= depth_min:
        raise ValueError(f"Invalid depth metadata in {meta_path}")
    depth = (depth_max - encoded * (depth_max - depth_min)) / canonical_scale

    fov = float(_meta_value(meta, "camera", "fov_degrees"))
    xs = ((torch.arange(width, dtype=torch.float32) + 0.5) / width) * 2.0 - 1.0
    ys = ((torch.arange(height, dtype=torch.float32) + 0.5) / height) * 2.0 - 1.0
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    tan_half = math.tan(math.radians(fov) * 0.5)
    ray_can = torch.stack((grid_x * tan_half, torch.ones_like(grid_x), -grid_y * tan_half), dim=-1)
    ray_can = torch.nn.functional.normalize(ray_can, dim=-1, eps=1e-6)
    camera_can = ray_can.new_tensor(CANONICAL_CAMERA)
    point_can = camera_can + depth[..., None] * ray_can
    point_cv = canonical_to_opencv_relative(point_can).permute(2, 0, 1).contiguous()
    valid = torch.isfinite(point_cv).all(dim=0, keepdim=True)
    point_cv = torch.where(valid, point_cv, torch.zeros_like(point_cv))
    return point_cv, valid.float(), {
        "depth_min_meters": depth_min,
        "depth_max_meters": depth_max,
        "canonical_scale": canonical_scale,
        "fov_degrees": fov,
        "moge_scale": 1.0,
    }


def load_and_align_moge_point_map(
    point_map_path: str | Path,
    object_mask: torch.Tensor,
    *,
    canonical_distance: float = 3.5,
    min_valid_depth: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Load MoGe OpenCV points and estimate canonical/MoGe scale per scene.

    The deployable restricted alignment fixes the camera at the known
    TokenLight canonical origin and scales the median foreground depth to the
    canonical camera distance.  It intentionally performs no GT-depth fit.
    """

    array = np.load(point_map_path, allow_pickle=False)
    if array.ndim != 3:
        raise ValueError(f"Expected HWC or CHW point map, got {array.shape}: {point_map_path}")
    if array.shape[-1] == 3:
        points = torch.from_numpy(np.asarray(array, dtype=np.float32).copy()).permute(2, 0, 1)
    elif array.shape[0] == 3:
        points = torch.from_numpy(np.asarray(array, dtype=np.float32).copy())
    else:
        raise ValueError(f"Cannot identify XYZ axis in {array.shape}: {point_map_path}")
    valid = torch.isfinite(points).all(dim=0, keepdim=True)
    # MoGe uses positive OpenCV Z for visible surfaces.  Checking vector norm
    # alone would incorrectly accept finite points behind the camera.
    valid &= points[2:3] > min_valid_depth
    points = torch.where(valid.expand_as(points), points, torch.zeros_like(points))
    target_size = tuple(int(value) for value in object_mask.shape[-2:])
    if tuple(points.shape[-2:]) != target_size:
        # Resize XYZ with validity weights so invalid zero-filled pixels do not
        # pull valid geometry toward the camera at mask boundaries.
        weight = F.interpolate(valid[None].float(), size=target_size, mode="bilinear", align_corners=False)
        points = F.interpolate(points[None], size=target_size, mode="bilinear", align_corners=False)
        points = (points / weight.clamp_min(1e-6))[0]
        valid = F.interpolate(valid[None].float(), size=target_size, mode="nearest")[0] >= 0.5
        points = torch.where(valid.expand_as(points), points, torch.zeros_like(points))
    foreground = (object_mask >= 0.5) & valid
    depths = points[2:3][foreground]
    if depths.numel() < 16:
        raise ValueError(f"Too few valid MoGe foreground points in {point_map_path}")
    median_depth = float(depths.median())
    if not math.isfinite(median_depth) or median_depth <= min_valid_depth:
        raise ValueError(f"Invalid MoGe median object depth {median_depth}: {point_map_path}")
    scale = float(canonical_distance) / median_depth
    points = points * scale
    points = torch.where(valid.expand_as(points), points, torch.zeros_like(points))
    return points.contiguous(), valid.float(), {
        "moge_median_object_depth": median_depth,
        "moge_scale": scale,
        "canonical_distance": float(canonical_distance),
    }


def normalize_point_features(
    points_cv_canonical: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    scale: float = 4.0,
) -> torch.Tensor:
    """Bound canonical camera-space XYZ for CNN input while preserving axes."""

    if points_cv_canonical.ndim != 3 or points_cv_canonical.shape[0] != 3:
        raise ValueError(f"Expected [3,H,W], got {tuple(points_cv_canonical.shape)}")
    if tuple(valid_mask.shape) != (1, *points_cv_canonical.shape[-2:]):
        raise ValueError("valid_mask shape mismatch")
    if scale <= 0:
        raise ValueError("scale must be positive")
    features = (points_cv_canonical / float(scale)).clamp(-2.0, 2.0)
    return torch.where(valid_mask >= 0.5, features, torch.zeros_like(features))


def reference_light_direction(
    points_cv_canonical: torch.Tensor,
    object_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    light_cv_canonical: torch.Tensor,
) -> torch.Tensor:
    if points_cv_canonical.ndim != 3 or points_cv_canonical.shape[0] != 3:
        raise ValueError(f"Expected points [3,H,W], got {tuple(points_cv_canonical.shape)}")
    expected_mask = (1, *points_cv_canonical.shape[-2:])
    if tuple(object_mask.shape) != expected_mask or tuple(valid_mask.shape) != expected_mask:
        raise ValueError(
            f"object_mask and valid_mask must have shape {expected_mask}, got "
            f"{tuple(object_mask.shape)} and {tuple(valid_mask.shape)}"
        )
    light_cv_canonical = torch.as_tensor(
        light_cv_canonical,
        dtype=points_cv_canonical.dtype,
        device=points_cv_canonical.device,
    )
    if tuple(light_cv_canonical.shape) != (3,) or not bool(torch.isfinite(light_cv_canonical).all()):
        raise ValueError("light_cv_canonical must be one finite XYZ vector")
    foreground = (object_mask >= 0.5) & (valid_mask >= 0.5)
    values = points_cv_canonical.permute(1, 2, 0)[foreground[0]]
    if values.shape[0] < 1:
        raise ValueError("No valid foreground point for light direction")
    center = values.median(dim=0).values
    direction = light_cv_canonical - center
    if bool(torch.linalg.vector_norm(direction) <= 1e-6):
        raise ValueError("Light position coincides with the foreground reference point")
    return F.normalize(direction, dim=0, eps=1e-6)
