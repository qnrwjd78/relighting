from __future__ import annotations

"""TokenLight training with dense physical-map prefixes.

The original ``train_tokenlight.py`` is deliberately not modified.  This
entrypoint reuses its data/pipeline behavior and the FP32-master/BF16-autocast
runtime from ``train_tokenlight_decoder_safe.py``. GT masks retain the original
FlowMatch objective; LGI uses the paper's latent bridge and weighted image loss.
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import train_tokenlight as base  # noqa: E402
from model import train_tokenlight_decoder_safe as safe_runtime  # noqa: E402
from model import tokenlight_physics_runtime as physics_runtime  # noqa: E402
from model.tokenlight_wan_spatial import (  # noqa: E402
    SPATIAL_PREFIX_TYPE_NAMES,
    SpatialConditionEncoder,
    SpatialPrefixTypeEmbedding,
    model_fn_wan_video_tokenlight_spatial,
)


SPATIAL_SCHEMA_VERSION = 1
SPATIAL_KINDS = ("lgi", "gt_masks")
LGI_CHANNEL_ORDER = ("min", "max", "nearest_to_zero")
LGI_SOURCE_FILES = {
    "min": "min.npy",
    "max": "max.npy",
    "nearest_to_zero": "nearest.npy",
}
GT_MASK_CHANNEL_ORDER = ("direct_lit", "cast_shadow")
GT_MASK_VALIDATION_ONLY_INPUTS = ("object",)


def _unwrap_singleton(value: Any) -> Any:
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


def _resolve_from_root(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _scene_id(row: Mapping[str, Any]) -> str:
    for key in ("scene_folder", "scene_id"):
        value = _unwrap_singleton(row.get(key))
        if isinstance(value, str) and value:
            return value
    for key in ("video", "input_image"):
        value = _unwrap_singleton(row.get(key))
        if not isinstance(value, str):
            continue
        for part in Path(value).parts:
            if part.startswith("scene_"):
                return part
    raise KeyError("metadata row has no scene_folder/scene_id and no scene_* path component")


def _sample_name(row: Mapping[str, Any]) -> str:
    value = _unwrap_singleton(row.get("sample_name"))
    if isinstance(value, str) and value:
        return value
    video = _unwrap_singleton(row.get("video"))
    if isinstance(video, str) and video:
        return Path(video).stem
    raise KeyError("metadata row has no sample_name or usable video filename")


class SpatialConditionReader:
    """Strict, model-independent loader for LGI or GT-mask conditions."""

    def __init__(self, args) -> None:
        self.kind = str(args.spatial_condition_kind)
        if self.kind not in SPATIAL_KINDS:
            raise ValueError(f"unsupported spatial condition kind: {self.kind}")
        self.height = int(args.height)
        self.width = int(args.width)
        self.dataset_root = base._resolve_path(args.dataset_base_path or ".", base_path=REPO_ROOT)
        self.allow_missing = bool(args.tokenlight_spatial_allow_missing)

        self.gt_keys = {
            "object": str(args.tokenlight_gt_object_mask_key),
            "direct_lit": str(args.tokenlight_gt_direct_mask_key),
            "cast_shadow": str(args.tokenlight_gt_shadow_mask_key),
        }
        self.gt_mask_threshold = float(args.tokenlight_gt_mask_threshold)
        if not 0.0 <= self.gt_mask_threshold <= 1.0:
            raise ValueError("tokenlight_gt_mask_threshold must be in [0, 1]")
        self.lgi_root = base._resolve_path(args.tokenlight_lgi_root, base_path=REPO_ROOT)
        self.lgi_dir_key = str(args.tokenlight_lgi_dir_key)
        self.lgi_position_prefix = str(args.tokenlight_lgi_position_prefix)
        self.lgi_position_digits = int(args.tokenlight_lgi_position_digits)
        self.lgi_position_offset = int(args.tokenlight_lgi_position_offset)
        self.lgi_zero_triplet_invalid = bool(args.tokenlight_lgi_zero_triplet_invalid)
        if self.lgi_position_digits < 1:
            raise ValueError("tokenlight_lgi_position_digits must be >= 1")

    @property
    def channels(self) -> int:
        return len(LGI_CHANNEL_ORDER) if self.kind == "lgi" else len(GT_MASK_CHANNEL_ORDER)

    @property
    def channel_order(self) -> tuple[str, ...]:
        return LGI_CHANNEL_ORDER if self.kind == "lgi" else GT_MASK_CHANNEL_ORDER

    def schema(self) -> dict[str, Any]:
        schema = {
            "version": SPATIAL_SCHEMA_VERSION,
            "kind": self.kind,
            "shape_chw": [self.channels, self.height, self.width],
            "channel_order": list(self.channel_order),
            "prefix_type_names": list(SPATIAL_PREFIX_TYPE_NAMES),
            "prefix_timestep_semantics": {
                "source_mask_spatial_layout_light": "clean_t0",
                "target": "sampled_flow_timestep",
            },
            "missing_is_strict_error": not self.allow_missing,
        }
        if self.kind == "lgi":
            schema.update(
                {
                    "source_files": dict(LGI_SOURCE_FILES),
                    "equation_8_channel_order": ["min", "max", "nearest_to_zero"],
                    "paper_channels": 3,
                    "local_extra_channels": [],
                    "ordering_invariant": "min <= nearest_to_zero <= max",
                    "validity": (
                        "finite_and_not_all_zero_triplet"
                        if self.lgi_zero_triplet_invalid
                        else "finite"
                    ),
                    "position_mapping": {
                        "input_example": "position_000",
                        "output_prefix": self.lgi_position_prefix,
                        "output_digits": self.lgi_position_digits,
                        "integer_offset": self.lgi_position_offset,
                    },
                }
            )
        else:
            schema.update(
                {
                    "metadata_keys": dict(self.gt_keys),
                    "conditioning_channels": list(GT_MASK_CHANNEL_ORDER),
                    "validation_only_inputs": list(GT_MASK_VALIDATION_ONLY_INPUTS),
                    "binary_range": [0.0, 1.0],
                    "object_mask_threshold": self.gt_mask_threshold,
                    "geometry_masks_require_binary_png": True,
                    "relations": [
                        "direct_lit subset_of object",
                        "cast_shadow disjoint_from object",
                        "direct_lit disjoint_from cast_shadow",
                    ],
                }
            )
        return schema

    def load(self, row: Mapping[str, Any]) -> tuple[torch.Tensor, bool, dict[str, Any]]:
        try:
            if self.kind == "lgi":
                return self._load_lgi(row)
            return self._load_gt_masks(row)
        except (FileNotFoundError, KeyError) as exc:
            if not self.allow_missing:
                raise
            return (
                torch.zeros(self.channels, self.height, self.width, dtype=torch.float32),
                False,
                {"missing": True, "reason": str(exc)},
            )

    def _lgi_directory(self, row: Mapping[str, Any]) -> Path:
        explicit = _unwrap_singleton(row.get(self.lgi_dir_key))
        if isinstance(explicit, str) and explicit:
            return _resolve_from_root(explicit, self.lgi_root)

        sample = _sample_name(row)
        match = re.fullmatch(r"position_(\d+)", sample)
        if match is None:
            raise ValueError(
                f"cannot map LGI position from sample_name={sample!r}; expected position_<integer>"
            )
        position_id = int(match.group(1)) + self.lgi_position_offset
        if position_id < 0:
            raise ValueError(f"mapped LGI position is negative: {position_id}")
        position = f"{self.lgi_position_prefix}{position_id:0{self.lgi_position_digits}d}"
        return self.lgi_root / "scenes" / _scene_id(row) / position

    def _load_lgi(self, row: Mapping[str, Any]) -> tuple[torch.Tensor, bool, dict[str, Any]]:
        directory = self._lgi_directory(row)
        arrays: dict[str, np.ndarray] = {}
        for semantic_name, filename in LGI_SOURCE_FILES.items():
            path = directory / filename
            if not path.is_file():
                raise FileNotFoundError(f"missing LGI file: {path}")
            array = np.load(path, allow_pickle=False)
            if array.shape != (self.height, self.width):
                raise ValueError(
                    f"LGI {path} shape {array.shape} != {(self.height, self.width)}"
                )
            if array.dtype.kind not in "fiu":
                raise TypeError(f"LGI {path} must be numeric, got dtype={array.dtype}")
            array = np.asarray(array, dtype=np.float32)
            if not np.isfinite(array).all():
                bad = int(array.size - np.isfinite(array).sum())
                raise ValueError(f"LGI {path} contains {bad} NaN/Inf values")
            arrays[semantic_name] = array

        minimum = arrays["min"]
        maximum = arrays["max"]
        nearest = arrays["nearest_to_zero"]
        below = minimum > nearest
        above = nearest > maximum
        if below.any() or above.any():
            raise ValueError(
                f"LGI ordering violation in {directory}: "
                f"min>nearest={int(below.sum())}, nearest>max={int(above.sum())}"
            )

        # Eq. 8 order is deliberately min, max, nearest-to-zero.  Do not
        # reorder this stack to match the inequality check above.
        values = np.stack([minimum, maximum, nearest], axis=0)
        valid = np.ones((self.height, self.width), dtype=np.float32)
        if self.lgi_zero_triplet_invalid:
            valid[np.all(values == 0.0, axis=0)] = 0.0
        packed = values
        return (
            torch.from_numpy(np.ascontiguousarray(packed)),
            True,
            {
                "directory": str(directory),
                "valid_fraction": float(valid.mean()),
                "channel_min": [float(channel.min()) for channel in values],
                "channel_max": [float(channel.max()) for channel in values],
            },
        )

    def _mask_path(self, row: Mapping[str, Any], semantic_name: str) -> Path:
        key = self.gt_keys[semantic_name]
        value = _unwrap_singleton(row.get(key))
        if not isinstance(value, str) or not value:
            raise KeyError(f"metadata row is missing GT-mask key {key!r}")
        return _resolve_from_root(value, self.dataset_root)

    def _read_binary_mask(
        self,
        path: Path,
        semantic_name: str,
        *,
        require_binary_source: bool,
    ) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(f"missing {semantic_name} mask: {path}")
        with Image.open(path) as image:
            image.load()
            if image.size != (self.width, self.height):
                raise ValueError(
                    f"{semantic_name} mask {path} size {image.size} != "
                    f"{(self.width, self.height)}"
                )
            array = np.asarray(image.convert("L"), dtype=np.uint8)
        unique = np.unique(array)
        if require_binary_source and not np.isin(unique, np.array([0, 255], dtype=np.uint8)).all():
            raise ValueError(
                f"{semantic_name} mask {path} is not binary 0/255; values={unique[:16].tolist()}"
            )
        return array.astype(np.float32) >= self.gt_mask_threshold * 255.0

    def _load_gt_masks(self, row: Mapping[str, Any]) -> tuple[torch.Tensor, bool, dict[str, Any]]:
        names = (*GT_MASK_VALIDATION_ONLY_INPUTS, *GT_MASK_CHANNEL_ORDER)
        paths = {name: self._mask_path(row, name) for name in names}
        masks = {
            name: self._read_binary_mask(
                path,
                name,
                # The shared object silhouette is antialiased in fixed32.
                # Geometry-ray direct/shadow teachers must remain exact binary.
                require_binary_source=name != "object",
            )
            for name, path in paths.items()
        }
        obj = masks["object"]
        direct = masks["direct_lit"]
        shadow_raw = masks["cast_shadow"]
        # This channel represents cast shadow on the receiver/background, not
        # self-shadow.  Use the supplied object silhouette as the authoritative
        # exclusion mask and remove only overlapping shadow pixels.
        shadow_on_object = shadow_raw & obj
        shadow = shadow_raw & ~obj
        direct_outside = direct & ~obj
        direct_shadow_overlap = direct & shadow
        if direct_outside.any() or direct_shadow_overlap.any():
            raise ValueError(
                f"GT-mask relation failure for {_scene_id(row)}/{_sample_name(row)}: "
                f"direct_outside_object={int(direct_outside.sum())}, "
                f"shadow_on_object_removed={int(shadow_on_object.sum())}, "
                f"direct_shadow_overlap={int(direct_shadow_overlap.sum())}"
            )
        # Only the two user-requested geometric teachers condition the model.
        # The object silhouette is read solely to validate their semantics.
        packed = np.stack([direct, shadow], axis=0).astype(np.float32, copy=False)
        return (
            torch.from_numpy(np.ascontiguousarray(packed)),
            True,
            {
                "paths": {name: str(path) for name, path in paths.items()},
                "positive_fraction": [float(channel.mean()) for channel in packed],
                "shadow_on_object_removed": int(shadow_on_object.sum()),
            },
        )


class SpatialConditionDataset(torch.utils.data.Dataset):
    """Attach a validated dense condition while preserving base cache behavior."""

    def __init__(self, dataset, reader: SpatialConditionReader) -> None:
        self.dataset = dataset
        self.reader = reader
        self.data = dataset.data
        self.repeat = getattr(dataset, "repeat", 1)
        self.load_from_cache = getattr(dataset, "load_from_cache", False)
        self.is_vae_latent_cache_dataset = getattr(dataset, "is_vae_latent_cache_dataset", False)
        self.skip_tokenlight_file_filter = getattr(dataset, "skip_tokenlight_file_filter", False)

    def __len__(self) -> int:
        return len(self.dataset)

    def _get(self, index: int) -> dict[str, Any]:
        item = self.dataset[int(index)]
        # UnifiedDataset may replace declared path strings with PIL objects or
        # omit undeclared file keys.  Always resolve the physical condition
        # from the untouched metadata row, while returning the base item.
        raw_row = self.data[int(index) % len(self.data)]
        spatial_map, present, info = self.reader.load(raw_row)
        item["_tokenlight_spatial_map"] = spatial_map
        item["_tokenlight_spatial_present"] = bool(present)
        item["_tokenlight_spatial_info"] = info
        return item

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._get(int(index))

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        return [self._get(int(index)) for index in indices]


def _stack_spatial_maps(value: Any, *, batch: int) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.unsqueeze(0) if value.ndim == 3 else value
    elif isinstance(value, list):
        if len(value) != batch or not all(isinstance(item, torch.Tensor) for item in value):
            raise TypeError(f"expected {batch} spatial tensors, got {type(value).__name__}")
        tensor = torch.stack(value, dim=0)
    else:
        raise TypeError(f"missing _tokenlight_spatial_map tensor, got {type(value).__name__}")
    if tensor.ndim != 4 or int(tensor.shape[0]) != batch:
        raise ValueError(f"spatial map must be [B,C,H,W], got {tuple(tensor.shape)}")
    return tensor


def _stack_presence(value: Any, *, batch: int) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.bool().reshape(-1)
    elif isinstance(value, list):
        tensor = torch.tensor([bool(item) for item in value], dtype=torch.bool)
    else:
        tensor = torch.full((batch,), bool(value), dtype=torch.bool)
    if tensor.numel() == 1 and batch != 1:
        tensor = tensor.expand(batch)
    if tensor.numel() != batch:
        raise ValueError(f"spatial presence needs {batch} values, got {tensor.numel()}")
    return tensor


def LGIBridgeMatchLoss(pipe, **inputs):
    """LGI paper Eq. (1--5), adapted only at the Wan backbone boundary."""

    target = inputs["input_latents"]
    source = inputs.get("tokenlight_source_latents")
    if not isinstance(source, torch.Tensor) or source.shape != target.shape:
        raise ValueError("LGI bridge matching requires source and target RGB latents with equal shape")

    total_steps = len(pipe.scheduler.timesteps)
    lo = max(0, min(total_steps, int(float(inputs.get("min_timestep_boundary", 0.0)) * total_steps)))
    hi = max(0, min(total_steps, int(float(inputs.get("max_timestep_boundary", 1.0)) * total_steps)))
    if hi <= lo:
        raise ValueError("empty LGI bridge timestep interval")
    step = int(torch.randint(lo, hi, (1,), device="cpu").item())
    model_timestep = pipe.scheduler.timesteps[step].reshape(1).to(
        device=pipe.device, dtype=pipe.torch_dtype
    )
    scheduler_sigma = torch.as_tensor(
        pipe.scheduler.sigmas[step], device=target.device, dtype=torch.float32
    ).reshape(())
    # Wan traverses high sigma -> low sigma; the paper traverses source t=0 -> target t=1.
    t = (1.0 - scheduler_sigma).clamp(1e-4, 1.0 - 1e-4)
    view = (1,) + (1,) * (target.ndim - 1)
    t_b = t.to(dtype=target.dtype).reshape(view)
    bridge_sigma = float(inputs.get("tokenlight_lgi_bridge_sigma", 0.1))
    noise = torch.randn_like(target)
    z_t = (1.0 - t_b) * source + t_b * target
    if bridge_sigma:
        z_t = z_t + bridge_sigma * torch.sqrt(t_b * (1.0 - t_b)) * noise
    drift_target = (target - z_t) / (1.0 - t_b)

    model_inputs = dict(inputs)
    model_inputs["latents"] = z_t
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    drift_pred = pipe.model_fn(**models, **model_inputs, timestep=model_timestep)
    latent_loss = F.mse_loss(drift_pred.float(), drift_target.float())
    total = latent_loss
    metrics = {"train/lgi/bridge_drift_mse": latent_loss.detach(), "train/lgi/t": t.detach()}

    image_weight = float(inputs.get("tokenlight_lgi_image_loss_weight", 10.0))
    if image_weight:
        predicted_target = z_t + (1.0 - t_b) * drift_pred
        with torch.no_grad():
            source_video = (safe_runtime._decode_unclamped(pipe, source.detach(), gradient_checkpointing=False) + 1) * 0.5
            target_video = (safe_runtime._decode_unclamped(pipe, target.detach(), gradient_checkpointing=False) + 1) * 0.5
        predicted_video = (
            safe_runtime._decode_unclamped(pipe, predicted_target, gradient_checkpointing=True) + 1
        ) * 0.5
        threshold = float(inputs.get("tokenlight_lgi_brightness_threshold", 0.01))
        changed = (target_video - source_video).abs().amax(dim=1, keepdim=True) > threshold
        kernel = int(inputs.get("tokenlight_lgi_dilation_kernel", 17))
        b, _, frames, height, width = changed.shape
        changed_2d = changed.permute(0, 2, 1, 3, 4).reshape(b * frames, 1, height, width).float()
        changed_2d = F.max_pool2d(changed_2d, kernel, stride=1, padding=kernel // 2)
        changed = changed_2d.reshape(b, frames, 1, height, width).permute(0, 2, 1, 3, 4)
        error = (predicted_video.float() - target_video.float()).abs()
        denom = changed.sum().clamp_min(1.0) * int(error.shape[1])
        image_l1 = (error * changed).sum() / denom
        total = total + image_weight * image_l1
        metrics["train/lgi/weighted_image_l1"] = image_l1.detach()
        metrics["train/lgi/image_loss"] = (image_weight * image_l1).detach()
    metrics["train/lgi/total_loss"] = total.detach()
    pipe._tokenlight_loss_metrics = metrics
    return total


class TokenLightSpatialTrainingModule(base.TokenLightWanTrainingModule):
    """Base TokenLight trainer plus a trainable dense-map prefix encoder."""

    def __init__(
        self,
        *args,
        spatial_condition_kind: str,
        tokenlight_spatial_input_channels: int,
        tokenlight_spatial_hidden_channels: int = 64,
        tokenlight_spatial_channel_mean: str | Sequence[float] | None = None,
        tokenlight_spatial_channel_std: str | Sequence[float] | None = None,
        tokenlight_spatial_input_clip: float = 0.0,
        tokenlight_spatial_dropout: float = 0.1,
        tokenlight_lgi_bridge_sigma: float = 0.1,
        tokenlight_lgi_image_loss_weight: float = 10.0,
        tokenlight_lgi_brightness_threshold: float = 0.01,
        tokenlight_lgi_dilation_kernel: int = 17,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.spatial_condition_kind = str(spatial_condition_kind)
        self.tokenlight_spatial_dropout = float(tokenlight_spatial_dropout)
        self.tokenlight_lgi_loss_options = {
            "tokenlight_lgi_bridge_sigma": float(tokenlight_lgi_bridge_sigma),
            "tokenlight_lgi_image_loss_weight": float(tokenlight_lgi_image_loss_weight),
            "tokenlight_lgi_brightness_threshold": float(tokenlight_lgi_brightness_threshold),
            "tokenlight_lgi_dilation_kernel": int(tokenlight_lgi_dilation_kernel),
        }
        if not 0.0 <= self.tokenlight_spatial_dropout <= 1.0:
            raise ValueError("tokenlight_spatial_dropout must be in [0, 1]")
        token_dim = int(self.pipe.dit.dim)
        self.tokenlight_spatial_encoder = SpatialConditionEncoder(
            input_channels=int(tokenlight_spatial_input_channels),
            token_dim=token_dim,
            hidden_channels=int(tokenlight_spatial_hidden_channels),
            channel_mean=tokenlight_spatial_channel_mean,
            channel_std=tokenlight_spatial_channel_std,
            input_clip=float(tokenlight_spatial_input_clip),
        )
        self.tokenlight_spatial_type_embedding = SpatialPrefixTypeEmbedding(token_dim)

        checkpoint = kwargs.get("lora_checkpoint") or kwargs.get("resume_from_checkpoint")
        aux_state = base._load_checkpoint_state_dict(checkpoint)
        self._load_spatial_module_state(
            self.tokenlight_spatial_encoder,
            aux_state,
            "tokenlight_spatial_encoder",
        )
        self._load_spatial_module_state(
            self.tokenlight_spatial_type_embedding,
            aux_state,
            "tokenlight_spatial_type_embedding",
        )
        self.pipe.model_fn = self._spatial_model_fn
        if self.spatial_condition_kind == "lgi":
            self.task_to_loss["sft"] = self._lgi_sft_loss
            self.task_to_loss["sft:train"] = self._lgi_sft_loss

    def _lgi_sft_loss(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        del inputs_nega
        merged = dict(inputs_shared)
        merged.update(inputs_posi)
        merged.update(self.tokenlight_lgi_loss_options)
        return LGIBridgeMatchLoss(pipe, **merged)

    @staticmethod
    def _load_spatial_module_state(module, state_dict, prefix: str) -> None:
        module_state = base._extract_module_state(state_dict, prefix)
        if not module_state:
            return
        current = module.state_dict()
        compatible = {
            key: value
            for key, value in module_state.items()
            if key in current and tuple(value.shape) == tuple(current[key].shape)
        }
        skipped = sorted(set(module_state) - set(compatible))
        missing, unexpected = module.load_state_dict(compatible, strict=False)
        print(
            f"Loaded {prefix} from checkpoint: tensors={len(compatible)}, "
            f"shape_skipped={len(skipped)}, missing={len(missing)}, unexpected={len(unexpected)}"
        )

    def _spatial_model_fn(self, **kwargs):
        return model_fn_wan_video_tokenlight_spatial(
            tokenlight_light_encoder=self.light_encoder,
            tokenlight_type_embedding=self.tokenlight_type_embedding,
            tokenlight_spatial_encoder=self.tokenlight_spatial_encoder,
            tokenlight_spatial_type_embedding=self.tokenlight_spatial_type_embedding,
            **kwargs,
        )

    def _attach_spatial_inputs(self, inputs, data):
        inputs_shared, inputs_posi, inputs_nega = inputs
        batch = int(inputs_shared["input_latents"].shape[0])
        spatial_map = _stack_spatial_maps(data.get("_tokenlight_spatial_map"), batch=batch)
        present = _stack_presence(data.get("_tokenlight_spatial_present", True), batch=batch)
        if tuple(spatial_map.shape[-2:]) != (
            int(inputs_shared["height"]),
            int(inputs_shared["width"]),
        ):
            raise ValueError(
                f"spatial/target shape mismatch: {tuple(spatial_map.shape[-2:])} != "
                f"{(int(inputs_shared['height']), int(inputs_shared['width']))}"
            )
        inputs_shared["tokenlight_spatial_map"] = spatial_map.to(
            device=self.pipe.device,
            dtype=torch.float32,
            non_blocking=bool(spatial_map.is_pinned()),
        )
        inputs_shared["tokenlight_spatial_present"] = present.to(device=self.pipe.device)
        if self.training and self.tokenlight_spatial_dropout > 0:
            drop = torch.rand(batch, device=self.pipe.device) < self.tokenlight_spatial_dropout
        else:
            drop = torch.zeros(batch, device=self.pipe.device, dtype=torch.bool)
        inputs_posi["tokenlight_drop_spatial"] = drop
        inputs_nega["tokenlight_drop_spatial"] = True

        layout = data.get("_tokenlight_layout_tokens")
        if layout is not None:
            if isinstance(layout, list):
                layout = torch.stack(layout, dim=0)
            inputs_shared["tokenlight_layout_tokens"] = layout.to(
                device=self.pipe.device,
                dtype=self.pipe.torch_dtype,
            )
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None and self._has_cached_latents(data):
            prepared = self._attach_spatial_inputs(self._cached_batched_inputs(data), data)
            return self.task_to_loss[self.task](self.pipe, *prepared)
        if inputs is None and self._is_collated_batch(data):
            prepared = self._attach_spatial_inputs(self._batched_inputs(data), data)
            return self.task_to_loss[self.task](self.pipe, *prepared)
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        inputs = self._prepare_tokenlight_inputs(inputs, data)
        inputs = self._attach_spatial_inputs(inputs, data)
        return self.task_to_loss[self.task](self.pipe, *inputs)

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        exported = super().export_trainable_state_dict(state_dict, remove_prefix=remove_prefix)
        for module_name in ("tokenlight_spatial_encoder", "tokenlight_spatial_type_embedding"):
            module = getattr(self, module_name)
            for name, _ in module.named_parameters():
                full_key = f"{module_name}.{name}"
                if full_key not in state_dict:
                    continue
                export_key = full_key
                if remove_prefix and export_key.startswith(remove_prefix):
                    export_key = export_key[len(remove_prefix) :]
                exported[export_key] = state_dict[full_key]
            for name, _ in module.named_buffers():
                full_key = f"{module_name}.{name}"
                if full_key not in state_dict:
                    continue
                export_key = full_key
                if remove_prefix and export_key.startswith(remove_prefix):
                    export_key = export_key[len(remove_prefix) :]
                exported[export_key] = state_dict[full_key]
        return exported


def spatial_parser(train_mode: str, condition_kind: str) -> argparse.ArgumentParser:
    parser = safe_runtime.safe_parser(train_mode)
    parser.description = f"TokenLight {condition_kind} spatial-prefix trainer ({train_mode})."
    parser.add_argument("--spatial_condition_kind", choices=SPATIAL_KINDS, default=condition_kind)
    parser.add_argument("--tokenlight_spatial_input_channels", type=int, default=3 if condition_kind == "lgi" else 2)
    parser.add_argument("--tokenlight_spatial_hidden_channels", type=int, default=64)
    parser.add_argument(
        "--tokenlight_spatial_channel_mean",
        default="0,0,0" if condition_kind == "lgi" else "0,0",
    )
    parser.add_argument(
        "--tokenlight_spatial_channel_std",
        default="1,1,1" if condition_kind == "lgi" else "1,1",
    )
    parser.add_argument("--tokenlight_spatial_input_clip", type=float, default=0.0)
    parser.add_argument("--tokenlight_spatial_dropout", type=float, default=0.1)
    parser.add_argument("--tokenlight_lgi_bridge_sigma", type=float, default=0.1)
    parser.add_argument("--tokenlight_lgi_image_loss_weight", type=float, default=10.0)
    parser.add_argument("--tokenlight_lgi_brightness_threshold", type=float, default=0.01)
    parser.add_argument("--tokenlight_lgi_dilation_kernel", type=int, default=17)
    parser.add_argument("--batch_semantics_note", default=None)
    parser.add_argument(
        "--tokenlight_spatial_allow_missing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--tokenlight_gt_object_mask_key", default="mask")
    parser.add_argument("--tokenlight_gt_direct_mask_key", default="inf_mask")
    parser.add_argument("--tokenlight_gt_shadow_mask_key", default="shadow_mask")
    parser.add_argument("--tokenlight_gt_mask_threshold", type=float, default=0.5)

    parser.add_argument("--tokenlight_lgi_root", default="data/objaverse_32_lgimap")
    parser.add_argument("--tokenlight_lgi_dir_key", default="lgi_dir")
    parser.add_argument("--tokenlight_lgi_position_prefix", default="position_")
    parser.add_argument("--tokenlight_lgi_position_digits", type=int, default=2)
    parser.add_argument("--tokenlight_lgi_position_offset", type=int, default=0)
    parser.add_argument("--lgi_paper_channel_order", default="min,max,nearest_to_zero")
    parser.add_argument("--lgi_local_extra_channel", default=None)
    parser.add_argument(
        "--tokenlight_lgi_zero_triplet_invalid",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate spatial files and metadata, print a summary, and exit before model loading.",
    )
    parser.add_argument(
        "--preflight_max_samples",
        type=int,
        default=0,
        help="Maximum rows to validate; 0 validates every selected row.",
    )
    parser.add_argument("--preflight_start_index", type=int, default=0)
    parser.add_argument("--preflight_scene_id", default=None)
    parser.add_argument("--preflight_max_errors", type=int, default=20)
    return parser


def _validate_spatial_args(args, condition_kind: str, raw_config, parser) -> None:
    safe_runtime._validate_safe_args(args, raw_config, parser)
    if args.spatial_condition_kind != condition_kind:
        parser.error(
            f"this entrypoint requires spatial_condition_kind={condition_kind!r}, "
            f"got {args.spatial_condition_kind!r}"
        )
    expected_channels = len(LGI_CHANNEL_ORDER) if condition_kind == "lgi" else len(GT_MASK_CHANNEL_ORDER)
    if int(args.tokenlight_spatial_input_channels) != expected_channels:
        parser.error(
            f"{condition_kind} requires tokenlight_spatial_input_channels={expected_channels} "
            f"for its fixed schema"
        )
    if float(args.tokenlight_rgb_decoder_loss_weight) != 0.0:
        parser.error("use tokenlight_lgi_image_loss_weight for LGI; generic decoder loss must be zero")
    if float(args.tokenlight_rgb_latent_loss_weight) != 1.0:
        parser.error("the preserved base FlowMatch objective requires latent loss weight exactly 1.0")
    if args.task not in {"sft", "sft:train"}:
        parser.error("spatial-prefix training currently supports only sft/sft:train")
    if int(args.tokenlight_spatial_hidden_channels) < 8:
        parser.error("tokenlight_spatial_hidden_channels must be >= 8")
    if not 0 <= float(args.tokenlight_spatial_dropout) <= 1:
        parser.error("tokenlight_spatial_dropout must be in [0, 1]")
    if not 0 <= float(args.tokenlight_gt_mask_threshold) <= 1:
        parser.error("tokenlight_gt_mask_threshold must be in [0, 1]")
    if not math.isfinite(float(args.tokenlight_spatial_input_clip)) or float(args.tokenlight_spatial_input_clip) < 0:
        parser.error("tokenlight_spatial_input_clip must be finite and non-negative")
    if int(args.preflight_max_samples) < 0 or int(args.preflight_start_index) < 0:
        parser.error("preflight sample counts/indices must be non-negative")
    if int(args.preflight_max_errors) < 1:
        parser.error("preflight_max_errors must be >= 1")
    if str(args.lgi_paper_channel_order) != "min,max,nearest_to_zero":
        parser.error("lgi_paper_channel_order is fixed by Eq. 8 to min,max,nearest_to_zero")
    if condition_kind == "lgi" and str(args.lgi_local_extra_channel).lower() not in {"", "none", "null"}:
        parser.error("paper-faithful LGI uses exactly three channels; set lgi_local_extra_channel=null")
    if condition_kind == "lgi":
        if float(args.tokenlight_lgi_bridge_sigma) < 0:
            parser.error("tokenlight_lgi_bridge_sigma must be non-negative")
        if float(args.tokenlight_lgi_image_loss_weight) < 0:
            parser.error("tokenlight_lgi_image_loss_weight must be non-negative")
        if int(args.tokenlight_lgi_dilation_kernel) < 1 or int(args.tokenlight_lgi_dilation_kernel) % 2 == 0:
            parser.error("tokenlight_lgi_dilation_kernel must be a positive odd integer")

    # Construction validates normalization lengths and finite positive stds
    # without loading the large Wan model.
    _ = SpatialConditionEncoder(
        input_channels=expected_channels,
        token_dim=8,
        hidden_channels=max(8, int(args.tokenlight_spatial_hidden_channels)),
        channel_mean=args.tokenlight_spatial_channel_mean,
        channel_std=args.tokenlight_spatial_channel_std,
        input_clip=args.tokenlight_spatial_input_clip,
    )


def parse_spatial_args(condition_kind: str):
    if condition_kind not in SPATIAL_KINDS:
        raise ValueError(f"unknown condition_kind: {condition_kind}")
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--train_mode", choices=("single", "zero3"), default="single")
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    config_path = pre_args.config or os.environ.get(
        "TOKENLIGHT_TRAIN_CONFIG",
        f"configs/train_480/rgb_spatial_{condition_kind}.json",
    )
    raw_config = base._load_json_config(config_path)
    parser = spatial_parser(pre_args.train_mode, condition_kind)
    base._apply_config_defaults(parser, raw_config)
    parser.set_defaults(
        config=config_path,
        train_mode=pre_args.train_mode,
        spatial_condition_kind=condition_kind,
    )
    args = parser.parse_args()
    raw_config = base._load_json_config(args.config)
    _validate_spatial_args(args, condition_kind, raw_config, parser)
    base._resolve_weight_paths(args)
    if not args.preflight:
        base._append_timestamp_to_output_path(args)
    return args, raw_config, raw_config


def _metadata_paths(args) -> list[Path]:
    specs = base._parse_vae_latent_cache_specs(getattr(args, "vae_latent_cache_specs", None))
    if specs:
        paths = [metadata for metadata, _ in specs]
    else:
        paths = [base._resolve_path(args.dataset_metadata_path, base_path=REPO_ROOT)]
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve(strict=False)
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def _preflight_rows(args) -> list[dict[str, Any]]:
    rows = []
    for path in _metadata_paths(args):
        if not path.is_file():
            raise FileNotFoundError(f"missing metadata: {path}")
        rows.extend(base._read_jsonl_rows(path))
    rows = [row for row in rows if row.get("valid") is not False]
    rejected = safe_runtime._load_rejected_scene_ids(args.dataset_reject_metadata_path)
    if rejected:
        rows = [row for row in rows if not safe_runtime._row_is_rejected(row, rejected)]
    if args.preflight_scene_id:
        rows = [row for row in rows if _scene_id(row) == str(args.preflight_scene_id)]
    start = int(args.preflight_start_index)
    rows = rows[start:]
    maximum = int(args.preflight_max_samples)
    return rows[:maximum] if maximum else rows


def run_preflight(args) -> dict[str, Any]:
    reader = SpatialConditionReader(args)
    rows = _preflight_rows(args)
    if not rows:
        raise ValueError("preflight selected zero metadata rows")
    present = 0
    missing = 0
    errors: list[dict[str, Any]] = []
    positive_or_valid_sum = np.zeros(reader.channels, dtype=np.float64)
    attempted = 0
    for index, row in enumerate(rows):
        attempted += 1
        try:
            tensor, is_present, _ = reader.load(row)
            if tensor.shape != (reader.channels, reader.height, reader.width):
                raise ValueError(f"reader returned unexpected shape {tuple(tensor.shape)}")
            if not torch.isfinite(tensor).all():
                raise ValueError("reader returned NaN/Inf")
            if is_present:
                present += 1
            else:
                missing += 1
            positive_or_valid_sum += tensor.double().mean(dim=(1, 2)).numpy()
        except Exception as exc:  # report multiple independent bad rows
            errors.append(
                {
                    "row": index + int(args.preflight_start_index),
                    "scene": row.get("scene_folder", row.get("scene_id")),
                    "sample": row.get("sample_name"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if len(errors) >= int(args.preflight_max_errors):
                break
    checked = attempted
    summary = {
        "ok": not errors,
        "selected_rows": len(rows),
        "checked": checked,
        "present": present,
        "missing_allowed": missing,
        "error_count": len(errors),
        "stopped_at_error_limit": bool(errors) and checked < len(rows),
        "channel_mean_across_images": (positive_or_valid_sum / max(1, checked)).tolist(),
        "schema": reader.schema(),
        "errors": errors,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise RuntimeError(
            f"spatial preflight failed; showing {len(errors)} error(s) from {checked} checked rows"
        )
    return summary


def build_spatial_dataset(args) -> SpatialConditionDataset:
    dataset = safe_runtime.build_safe_dataset(args)
    return SpatialConditionDataset(dataset, SpatialConditionReader(args))


def _make_model(args) -> TokenLightSpatialTrainingModule:
    return TokenLightSpatialTrainingModule(
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
            else args._accelerator_device
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
        spatial_condition_kind=args.spatial_condition_kind,
        tokenlight_spatial_input_channels=args.tokenlight_spatial_input_channels,
        tokenlight_spatial_hidden_channels=args.tokenlight_spatial_hidden_channels,
        tokenlight_spatial_channel_mean=args.tokenlight_spatial_channel_mean,
        tokenlight_spatial_channel_std=args.tokenlight_spatial_channel_std,
        tokenlight_spatial_input_clip=args.tokenlight_spatial_input_clip,
        tokenlight_spatial_dropout=args.tokenlight_spatial_dropout,
        tokenlight_lgi_bridge_sigma=args.tokenlight_lgi_bridge_sigma,
        tokenlight_lgi_image_loss_weight=args.tokenlight_lgi_image_loss_weight,
        tokenlight_lgi_brightness_threshold=args.tokenlight_lgi_brightness_threshold,
        tokenlight_lgi_dilation_kernel=args.tokenlight_lgi_dilation_kernel,
    )


def main(condition_kind: str) -> None:
    args, raw_config, merged_config = parse_spatial_args(condition_kind)
    reader = SpatialConditionReader(args)
    args.spatial_schema_version = SPATIAL_SCHEMA_VERSION
    args.tokenlight_spatial_schema = json.dumps(reader.schema(), sort_keys=True, separators=(",", ":"))
    args.loss_impl_version = (
        "tokenlight_lgi_bridge_weighted_image_v2"
        if condition_kind == "lgi"
        else "tokenlight_latent_flowmatch_spatial_v1"
    )
    if args.preflight:
        run_preflight(args)
        return

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
    args._accelerator_device = accelerator.device
    base.save_training_config_snapshot(args, raw_config, merged_config, accelerator)
    dataset = build_spatial_dataset(args)
    model = _make_model(args)
    model_logger = base._make_model_logger(args)
    physics_runtime.launch_physics_training_task(
        accelerator,
        dataset,
        model,
        model_logger,
        args=args,
    )


__all__ = [
    "GT_MASK_CHANNEL_ORDER",
    "LGI_CHANNEL_ORDER",
    "LGI_SOURCE_FILES",
    "SPATIAL_SCHEMA_VERSION",
    "SpatialConditionDataset",
    "SpatialConditionReader",
    "TokenLightSpatialTrainingModule",
    "build_spatial_dataset",
    "main",
    "parse_spatial_args",
    "run_preflight",
]
