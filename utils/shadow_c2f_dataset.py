"""Dataset and detector-noise augmentation for standalone shadow C2F."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from utils.shadow_geometry import canonical_light_to_opencv
from utils.shadow_pipeline_io import light_position as manifest_light_position
from utils.shadow_pipeline_io import read_jsonl, sample_key, scene_id


ADAPTER_IMAGE_CACHE_SCHEMA = "adaptershadow_image_v1"
ADAPTER_DELTA_CACHE_SCHEMA = "adaptershadow_positive_delta_v1"
GEOMETRY_CACHE_SCHEMA = "tokenlight_shadow_geometry_cache_v1"
PHYSICS_CACHE_SCHEMA = "tokenlight_shadow_physics_prior_v1"


def _rgb(path: str | Path, size: int) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB")
        if image.size != (size, size):
            image = image.resize((size, size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


def _mask(path: str | Path, size: int) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("L")
        if image.size != (size, size):
            image = image.resize((size, size), Image.Resampling.NEAREST)
        array = np.asarray(image, dtype=np.uint8) >= 128
    return torch.from_numpy(array.copy()).unsqueeze(0).float()


def _scalar(array: np.ndarray, *, path: Path, field: str) -> Any:
    value = np.asarray(array)
    if value.shape != ():
        raise ValueError(f"{path}:{field} must be scalar, got {value.shape}")
    return value.item()


def _probability_map(array: np.ndarray, *, path: Path, field: str) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 2 or not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"{path}:{field} must be a floating [H,W] array, got {value.shape}/{value.dtype}")
    value = value.astype(np.float32, copy=False)
    if not np.isfinite(value).all():
        raise ValueError(f"{path}:{field} contains NaN or infinity")
    if value.size and (float(value.min()) < -1e-4 or float(value.max()) > 1.0001):
        raise ValueError(f"{path}:{field} must be in [0,1]")
    return value.clip(0.0, 1.0)


class _ArrayLru:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(0, int(capacity))
        self.values: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()

    def get(self, path: str | Path, field: str) -> np.ndarray:
        key = (str(path), field)
        if key in self.values:
            value = self.values.pop(key)
            self.values[key] = value
            return value
        with np.load(path, allow_pickle=False) as archive:
            if field not in archive:
                raise KeyError(f"{field!r} is absent from {path}")
            value = np.asarray(archive[field]).copy()
        if self.capacity:
            self.values[key] = value
            while len(self.values) > self.capacity:
                self.values.popitem(last=False)
        return value


class AdapterFeatureAugment:
    """Noise/dropout/morphology used to prevent detector over-reliance."""

    def __init__(
        self,
        *,
        dropout_probability: float = 0.1,
        morphology_probability: float = 0.3,
        noise_std: float = 0.03,
        false_blob_probability: float = 0.15,
    ) -> None:
        self.dropout_probability = float(dropout_probability)
        self.morphology_probability = float(morphology_probability)
        self.noise_std = float(noise_std)
        self.false_blob_probability = float(false_blob_probability)
        for name, value in (
            ("dropout_probability", self.dropout_probability),
            ("morphology_probability", self.morphology_probability),
            ("false_blob_probability", self.false_blob_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[0] != 3:
            raise ValueError(f"Expected [3,H,W] adapter features, got {tuple(features.shape)}")
        value = features.float().clone()
        if torch.rand(()) < self.dropout_probability:
            return torch.zeros_like(value)
        if torch.rand(()) < self.morphology_probability:
            kernel = int(torch.randint(1, 3, ()).item()) * 2 + 1
            if torch.rand(()) < 0.5:
                value = F.max_pool2d(value[None], kernel, stride=1, padding=kernel // 2)[0]
            else:
                value = -F.max_pool2d(-value[None], kernel, stride=1, padding=kernel // 2)[0]
        if self.noise_std:
            value = value + torch.randn_like(value) * self.noise_std
        if torch.rand(()) < self.false_blob_probability:
            low = torch.rand(1, 1, 12, 12)
            blobs = F.interpolate(low, size=value.shape[-2:], mode="bilinear", align_corners=False)[0]
            blobs = (blobs > 0.82).float() * float(torch.empty(()).uniform_(0.1, 0.5))
            value = value + blobs
        value = value.clamp(0.0, 1.0)
        # The delta channel is always defined by the possibly augmented target
        # and source observations, keeping its semantics exact.
        value[2:3] = (value[0:1] - value[1:2]).clamp_min(0.0)
        return value


class ShadowC2FDataset(Dataset):
    """Load one leakage-safe C2F manifest row.

    The default contract requires AdapterShadow caches plus a geometry cache.
    ``coarse_prior_mode='adapter_delta'`` uses the positive target-minus-source
    *coarse* AdapterShadow probability as the C2F starting mask.  A ray-cast
    physics cache is read only for the explicit ``'physics'`` ablation.
    """

    def __init__(
        self,
        manifest: str | Path,
        *,
        image_size: int = 480,
        adapter_mode: str = "required",
        coarse_prior_mode: str = "adapter_delta",
        require_target: bool = True,
        augment_adapter: bool = False,
        array_cache_size: int = 16,
        augment_options: dict[str, float] | None = None,
    ) -> None:
        self.rows = read_jsonl(manifest)
        if not self.rows:
            raise ValueError(f"No rows in {manifest}")
        self.image_size = int(image_size)
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if adapter_mode not in {"required", "optional", "zero"}:
            raise ValueError("adapter_mode must be required, optional, or zero")
        if coarse_prior_mode not in {"adapter_delta", "adapter_target", "physics"}:
            raise ValueError(
                "coarse_prior_mode must be adapter_delta, adapter_target, or physics"
            )
        self.adapter_mode = adapter_mode
        self.coarse_prior_mode = coarse_prior_mode
        self.require_target = bool(require_target)
        self.augment = AdapterFeatureAugment(**(augment_options or {})) if augment_adapter else None
        self.arrays = _ArrayLru(array_cache_size)

    def __len__(self) -> int:
        return len(self.rows)

    def _adapter(self, row: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(3, self.image_size, self.image_size)
        if self.adapter_mode == "zero":
            return zeros, zeros[:1]
        paths = {
            "target": Path(str(row.get("adapter_target_cache", ""))),
            "source": Path(str(row.get("adapter_source_cache", ""))),
            "delta": Path(str(row.get("adapter_delta_cache", ""))),
        }
        if not all(path.is_file() for path in paths.values()):
            if self.adapter_mode == "optional":
                return zeros, zeros[:1]
            missing = [f"{name}={path}" for name, path in paths.items() if not path.is_file()]
            raise FileNotFoundError(f"Missing AdapterShadow cache for {sample_key(row)}: {missing}")
        expected_schemas = {
            "target": ADAPTER_IMAGE_CACHE_SCHEMA,
            "source": ADAPTER_IMAGE_CACHE_SCHEMA,
            "delta": ADAPTER_DELTA_CACHE_SCHEMA,
        }
        cache_ids: dict[str, str] = {}
        output_sizes: dict[str, int] = {}
        for name, path in paths.items():
            schema = str(_scalar(self.arrays.get(path, "schema"), path=path, field="schema"))
            if schema != expected_schemas[name]:
                raise ValueError(f"Unexpected AdapterShadow schema at {path}: {schema!r}")
            cache_ids[name] = str(
                _scalar(self.arrays.get(path, "cache_id"), path=path, field="cache_id")
            )
            output_sizes[name] = int(
                _scalar(self.arrays.get(path, "output_size"), path=path, field="output_size")
            )
        if len(set(cache_ids.values())) != 1:
            raise ValueError(
                f"AdapterShadow cache_id mismatch for {sample_key(row)}: {cache_ids}"
            )
        if len(set(output_sizes.values())) != 1 or next(iter(output_sizes.values())) <= 0:
            raise ValueError(
                f"AdapterShadow output_size mismatch for {sample_key(row)}: {output_sizes}"
            )
        target_array = _probability_map(
            self.arrays.get(paths["target"], "final_prob"),
            path=paths["target"],
            field="final_prob",
        )
        source_array = _probability_map(
            self.arrays.get(paths["source"], "final_prob"),
            path=paths["source"],
            field="final_prob",
        )
        delta_array = _probability_map(
            self.arrays.get(paths["delta"], "positive_delta_final_prob"),
            path=paths["delta"],
            field="positive_delta_final_prob",
        )
        target_coarse = _probability_map(
            self.arrays.get(paths["target"], "coarse_prob"),
            path=paths["target"],
            field="coarse_prob",
        )
        source_coarse = _probability_map(
            self.arrays.get(paths["source"], "coarse_prob"),
            path=paths["source"],
            field="coarse_prob",
        )
        delta_coarse = _probability_map(
            self.arrays.get(paths["delta"], "positive_delta_coarse_prob"),
            path=paths["delta"],
            field="positive_delta_coarse_prob",
        )
        expected_shape = (next(iter(output_sizes.values())),) * 2
        arrays = (
            ("target", target_array),
            ("source", source_array),
            ("delta", delta_array),
            ("target_coarse", target_coarse),
            ("source_coarse", source_coarse),
            ("delta_coarse", delta_coarse),
        )
        for name, value in arrays:
            if value.shape != expected_shape:
                raise ValueError(
                    f"AdapterShadow {name} shape for {sample_key(row)} is {value.shape}; "
                    f"expected {expected_shape}"
                )
        if not np.allclose(
            delta_array,
            np.maximum(target_array - source_array, 0.0),
            atol=2e-3,
            rtol=0.0,
        ):
            raise ValueError(f"AdapterShadow final delta semantics mismatch for {sample_key(row)}")
        if not np.allclose(
            delta_coarse,
            np.maximum(target_coarse - source_coarse, 0.0),
            atol=2e-3,
            rtol=0.0,
        ):
            raise ValueError(f"AdapterShadow coarse delta semantics mismatch for {sample_key(row)}")
        target = torch.from_numpy(target_array)
        source = torch.from_numpy(source_array)
        delta = torch.from_numpy(delta_array)
        features = torch.stack((target, source, delta), dim=0)
        if tuple(features.shape[-2:]) != (self.image_size, self.image_size):
            features = F.interpolate(
                features[None], size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
            )[0]
        coarse_array = delta_coarse if self.coarse_prior_mode == "adapter_delta" else target_coarse
        coarse_prior = torch.from_numpy(coarse_array).unsqueeze(0)
        # Preserve the detector's native coarse branch for validation/inference.
        # During training, augmentation must also affect the primary prior;
        # otherwise detector dropout would be bypassed by an untouched mask.
        if self.augment is not None:
            features = self.augment(features)
            feature_index = 2 if self.coarse_prior_mode == "adapter_delta" else 0
            coarse_prior = features[feature_index : feature_index + 1]
        return features, coarse_prior

    def _geometry(
        self,
        row: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        geometry_path = Path(str(row.get("geometry_cache", "")))
        if not geometry_path.is_file():
            raise FileNotFoundError(
                f"Missing geometry cache for {sample_key(row)}: {geometry_path}"
            )

        geometry_schema = str(
            _scalar(self.arrays.get(geometry_path, "schema"), path=geometry_path, field="schema")
        )
        if geometry_schema != GEOMETRY_CACHE_SCHEMA:
            raise ValueError(f"Unexpected geometry schema at {geometry_path}: {geometry_schema!r}")
        geometry_id = str(
            _scalar(self.arrays.get(geometry_path, "cache_id"), path=geometry_path, field="cache_id")
        )
        expected_geometry_id = row.get("geometry_cache_id")
        if expected_geometry_id is not None and str(expected_geometry_id) != geometry_id:
            raise ValueError(
                f"Manifest geometry_cache_id does not match its cache for {sample_key(row)}"
            )
        feature_scale = float(
            _scalar(
                self.arrays.get(geometry_path, "feature_scale"),
                path=geometry_path,
                field="feature_scale",
            )
        )
        if not np.isfinite(feature_scale) or feature_scale <= 0:
            raise ValueError(f"{geometry_path}:feature_scale must be finite and positive")

        point_array = np.asarray(self.arrays.get(geometry_path, "point_features"))
        valid_array = np.asarray(self.arrays.get(geometry_path, "point_valid"))
        if point_array.ndim != 3 or point_array.shape[0] != 3 or not np.issubdtype(
            point_array.dtype, np.floating
        ):
            raise ValueError(
                f"{geometry_path}:point_features must be floating [3,H,W], "
                f"got {point_array.shape}/{point_array.dtype}"
            )
        if valid_array.shape != (1, *point_array.shape[-2:]):
            raise ValueError(
                f"{geometry_path}:point_valid must be [1,H,W], got {valid_array.shape}"
            )
        point_array = point_array.astype(np.float32, copy=False)
        if not np.isfinite(point_array).all() or not np.isfinite(valid_array).all():
            raise ValueError(f"Non-finite geometry values in {geometry_path}")
        point_features = torch.from_numpy(point_array)
        point_valid = torch.from_numpy((valid_array >= 0.5).astype(np.float32, copy=False))
        if tuple(point_features.shape[-2:]) != (self.image_size, self.image_size):
            weight = F.interpolate(
                point_valid[None],
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            point_features = F.interpolate(
                point_features[None] * point_valid[None],
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ) / weight.clamp_min(1e-6)
            point_valid = F.interpolate(
                point_valid[None], size=(self.image_size, self.image_size), mode="nearest"
            )[0]
            point_features = point_features[0] * point_valid
        else:
            point_features = point_features * point_valid

        canonical_position = manifest_light_position(row)
        light_position = canonical_light_to_opencv(canonical_position) / feature_scale
        if tuple(light_position.shape) != (3,) or not bool(torch.isfinite(light_position).all()):
            raise ValueError(f"Invalid normalized light_position for {sample_key(row)}")

        physics_prior: torch.Tensor | None = None
        if self.coarse_prior_mode == "physics":
            physics_path = Path(str(row.get("physics_cache", "")))
            if not physics_path.is_file():
                raise FileNotFoundError(
                    f"Physics ablation requested but cache is missing for "
                    f"{sample_key(row)}: {physics_path}"
                )
            physics_schema = str(
                _scalar(
                    self.arrays.get(physics_path, "schema"),
                    path=physics_path,
                    field="schema",
                )
            )
            if physics_schema != PHYSICS_CACHE_SCHEMA:
                raise ValueError(f"Unexpected physics schema at {physics_path}: {physics_schema!r}")
            physics_geometry_id = str(
                _scalar(
                    self.arrays.get(physics_path, "geometry_cache_id"),
                    path=physics_path,
                    field="geometry_cache_id",
                )
            )
            if geometry_id != physics_geometry_id:
                raise ValueError(
                    f"Geometry/physics cache mismatch for {sample_key(row)}: "
                    f"{geometry_id!r} != {physics_geometry_id!r}"
                )
            physics_id = str(
                _scalar(
                    self.arrays.get(physics_path, "cache_id"),
                    path=physics_path,
                    field="cache_id",
                )
            )
            expected_physics_id = row.get("physics_cache_id")
            if expected_physics_id is not None and str(expected_physics_id) != physics_id:
                raise ValueError(
                    f"Manifest physics_cache_id does not match its cache for {sample_key(row)}"
                )
            prior_array = _probability_map(
                self.arrays.get(physics_path, "physics_prior"),
                path=physics_path,
                field="physics_prior",
            )
            coarse_size = np.asarray(self.arrays.get(physics_path, "coarse_size"))
            if coarse_size.shape != (2,) or tuple(int(value) for value in coarse_size) != prior_array.shape:
                raise ValueError(
                    f"{physics_path}:coarse_size {coarse_size.tolist()} does not match "
                    f"physics_prior {prior_array.shape}"
                )
            physics_prior = torch.from_numpy(prior_array).unsqueeze(0)
        return (
            point_features.contiguous(),
            light_position.float().contiguous(),
            physics_prior,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        source = _rgb(row["source_image"], self.image_size)
        object_mask = _mask(row["object_mask"], self.image_size)
        receiver = _mask(row["receiver_mask"], self.image_size)
        adapter_features, adapter_prior = self._adapter(row)
        point_features, light_position, physics_prior = self._geometry(row)
        coarse_prior = physics_prior if physics_prior is not None else adapter_prior
        result = {
            "sample_id": sample_key(row),
            "scene_id": str(row["scene_id"]),
            # These lightweight fields let the optional online frozen stack run
            # baseline relighting and FOCUS inside the training process.
            "attrs_json": str(row.get("attrs_json", "")),
            "prompt": str(row.get("prompt", "")),
            "image": source,
            "object_mask": object_mask,
            "receiver_mask": receiver,
            "point_map": point_features,
            "light_position": light_position,
            "coarse_prior": coarse_prior,
            "adapter_features": adapter_features,
        }
        if self.require_target:
            target_value = row.get("gt_shadow_mask")
            if not target_value:
                raise KeyError(f"{sample_key(row)} has no gt_shadow_mask")
            target = _mask(target_value, self.image_size)
            result["target"] = target * receiver * (1.0 - object_mask)
        return result


__all__ = [
    "ADAPTER_DELTA_CACHE_SCHEMA",
    "ADAPTER_IMAGE_CACHE_SCHEMA",
    "GEOMETRY_CACHE_SCHEMA",
    "PHYSICS_CACHE_SCHEMA",
    "AdapterFeatureAugment",
    "ShadowC2FDataset",
]
