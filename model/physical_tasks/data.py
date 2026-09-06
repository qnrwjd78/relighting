"""Dataset and physically motivated feature construction."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_rgb(path: str, size: int, *, nearest: bool = False) -> torch.Tensor:
    image = Image.open(resolve_path(path)).convert("RGB")
    interpolation = Image.Resampling.NEAREST if nearest else Image.Resampling.BICUBIC
    image = image.resize((size, size), interpolation)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def load_gray(path: str, size: int, *, nearest: bool = False) -> torch.Tensor:
    image = Image.open(resolve_path(path)).convert("L")
    interpolation = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    image = image.resize((size, size), interpolation)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


class ManifestDataset(Dataset):
    def __init__(self, manifest: str | Path, *, image_size: int, split: str = "train") -> None:
        self.image_size = int(image_size)
        self.rows: list[dict[str, Any]] = []
        with resolve_path(str(manifest)).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if split != "all" and row.get("split") != split:
                    continue
                self.rows.append(row)
        if not self.rows:
            raise ValueError(f"No rows for split={split!r} in {manifest}")

    def __len__(self) -> int:
        return len(self.rows)

class LightPairDataset(ManifestDataset):
    """LightNet data: read only source RGB, target RGB, and numeric labels."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "sample_id": row["sample_id"],
            "source": load_rgb(row["source_image"], self.image_size),
            "target": load_rgb(row["target_image"], self.image_size),
            "light_position": torch.tensor(row["light_position"], dtype=torch.float32),
            "light_color": torch.tensor(row["light_color"], dtype=torch.float32),
            "light_intensity": torch.tensor(float(row["light_intensity"]), dtype=torch.float32),
            "light_radius": torch.tensor(float(row["light_radius"]), dtype=torch.float32),
            "ambient_scale": torch.tensor(float(row["ambient_scale"]), dtype=torch.float32),
            "camera_location": torch.tensor(row["canonical_camera_location"], dtype=torch.float32),
        }


class Fixed32MaskDataset(ManifestDataset):
    """fixed32 geometry-ray labels plus the common physical inputs."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        normal = load_rgb(row["normal_image"], self.image_size) * 2.0 - 1.0
        normal = torch.nn.functional.normalize(normal, dim=0, eps=1e-6)
        depth_encoded = load_gray(row["depth_image"], self.image_size)
        depth_min = float(row["depth_min_meters"])
        depth_max = float(row["depth_max_meters"])
        depth_meters = depth_max - depth_encoded * (depth_max - depth_min)
        return {
            "sample_id": row["sample_id"],
            "source": load_rgb(row["source_image"], self.image_size),
            "normal_world": normal,
            "mask": (load_gray(row["object_mask"], self.image_size, nearest=True) >= 0.5).float(),
            "receiver_mask": (load_gray(row["receiver_mask"], self.image_size, nearest=True) >= 0.5).float(),
            "visibility": (load_gray(row["visibility_mask"], self.image_size, nearest=True) >= 0.5).float(),
            "shadow": (load_gray(row["shadow_mask"], self.image_size, nearest=True) >= 0.5).float(),
            "depth_meters": depth_meters,
            "light_position": torch.tensor(row["light_position"], dtype=torch.float32),
            "fov_degrees": torch.tensor(float(row["fov_degrees"]), dtype=torch.float32),
            "canonical_scale": torch.tensor(float(row["canonical_scale"]), dtype=torch.float32),
            "camera_location": torch.tensor(row["canonical_camera_location"], dtype=torch.float32),
            "normal_world_to_canonical": torch.tensor(row["normal_world_to_canonical"], dtype=torch.float32),
            "camera_ray_to_geometry": torch.tensor(row["camera_ray_to_geometry"], dtype=torch.float32),
        }


def reconstruct_geometry(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    depth = batch["depth_meters"] / batch["canonical_scale"][:, None, None, None].clamp_min(1e-6)
    batch_size, _, height, width = depth.shape
    dtype, device = depth.dtype, depth.device
    ys = ((torch.arange(height, dtype=dtype, device=device) + 0.5) / height) * 2.0 - 1.0
    xs = ((torch.arange(width, dtype=dtype, device=device) + 0.5) / width) * 2.0 - 1.0
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    tan_half_fov = torch.tan(batch["fov_degrees"] * (math.pi / 360.0))[:, None, None]
    ray = torch.stack(
        [
            grid_x[None].expand(batch_size, -1, -1) * tan_half_fov,
            torch.ones((batch_size, height, width), dtype=dtype, device=device),
            -grid_y[None].expand(batch_size, -1, -1) * tan_half_fov,
        ],
        dim=1,
    )
    ray = torch.nn.functional.normalize(ray, dim=1, eps=1e-6)
    ray = torch.einsum("bij,bjhw->bihw", batch["camera_ray_to_geometry"], ray)
    point = batch["camera_location"][:, :, None, None] + depth * ray

    normal = torch.einsum("bij,bjhw->bihw", batch["normal_world_to_canonical"], batch["normal_world"])
    normal = torch.nn.functional.normalize(normal, dim=1, eps=1e-6)
    to_light = batch["light_position"][:, :, None, None] - point
    distance = torch.linalg.vector_norm(to_light, dim=1, keepdim=True).clamp_min(1e-4)
    direction = to_light / distance
    ndotl = (normal * direction).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    return point, normal, direction, distance, ndotl


def _geometry_features(batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    point, normal, direction, distance, ndotl = reconstruct_geometry(batch)
    return [batch["source"], point / 4.0, normal, direction, distance.log() / 2.0, ndotl, batch["mask"]]


def build_visibility_input(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(_geometry_features(batch), dim=1)


def build_shadow_input(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([*_geometry_features(batch), batch["receiver_mask"]], dim=1)


def lightnet_input(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([batch["source"], batch["target"]], dim=1)
