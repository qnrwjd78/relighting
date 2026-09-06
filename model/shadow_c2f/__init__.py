"""Physics-grounded standalone coarse-to-fine shadow-mask refinement."""

from .coordinates import (
    blender_camera_to_opencv_matrix,
    blender_world_to_opencv_camera_matrix,
    broadcast_light_map,
    normalize_vectors,
    point_light_direction,
    transform_directions,
    transform_points,
    world_directional_light_to_opencv_camera,
    world_point_light_to_opencv_camera,
)
from .losses import ShadowC2FLoss, binary_bce_loss, binary_dice_loss, shadow_c2f_loss
from .metrics import binary_mask_metrics
from .network import ShadowC2FConfig, ShadowCoarseToFine, build_adapter_features
from .physics import (
    PhysicsPriorConfig,
    directional_shadow_prior,
    gaussian_blur_mask,
    point_light_shadow_prior,
)

__all__ = [
    "PhysicsPriorConfig",
    "ShadowC2FConfig",
    "ShadowC2FLoss",
    "ShadowCoarseToFine",
    "binary_bce_loss",
    "binary_dice_loss",
    "binary_mask_metrics",
    "blender_camera_to_opencv_matrix",
    "blender_world_to_opencv_camera_matrix",
    "broadcast_light_map",
    "build_adapter_features",
    "directional_shadow_prior",
    "gaussian_blur_mask",
    "normalize_vectors",
    "point_light_direction",
    "point_light_shadow_prior",
    "shadow_c2f_loss",
    "transform_directions",
    "transform_points",
    "world_directional_light_to_opencv_camera",
    "world_point_light_to_opencv_camera",
]
