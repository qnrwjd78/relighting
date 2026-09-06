#!/usr/bin/env python3
"""Safe isolated TokenLight/Wan trainer for the fixed32 CoShadow adaptation."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import accelerate
import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import train_tokenlight as base  # noqa: E402
from model import train_tokenlight_decoder_safe as safe  # noqa: E402
from model import tokenlight_physics_runtime as physics_runtime  # noqa: E402
from model.coshadow_box_predictor import (  # noqa: E402
    load_coshadow_box_predictor,
    quantize_boxes_xyxy,
)
from model.lightoken_encoder import attrs_from_batch  # noqa: E402
from model.tokenlight_wan_coshadow import (  # noqa: E402
    CoShadowLatentMaskHead,
    CoShadowLayoutTokenEmbedding,
    CoShadowPromptEmbedding,
    model_fn_wan_video_tokenlight_coshadow,
)


DEFAULT_CONFIG = "configs/train_480/coshadow_diffusion.json"
IMPLEMENTATION_VERSION = "fixed32_coshadow_text_grounded_v2"
_FIXED32_LIGHT_FIELDS = ("x", "y", "z", "r", "g", "b", "lambda", "d")


def _bool_batch(value: Any, *, batch: int, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        result = value.to(device=device, dtype=torch.bool).flatten()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = torch.tensor(list(value), device=device, dtype=torch.bool).flatten()
    else:
        result = torch.tensor([bool(value)], device=device, dtype=torch.bool)
    if result.numel() == 1:
        result = result.expand(batch)
    if result.numel() != batch:
        raise ValueError(f"Expected {batch} boolean values, got {result.numel()}")
    return result


def _bbox_bins_batch(value: Any, *, batch: int, bins: int, device: torch.device) -> torch.Tensor:
    result = torch.as_tensor(value, device=device, dtype=torch.long)
    if result.ndim == 1:
        result = result.unsqueeze(0)
    if tuple(result.shape) != (batch, 4):
        raise ValueError(f"Expected quantized bbox shape {(batch, 4)}, got {tuple(result.shape)}")
    if bool(((result < 0) | (result >= int(bins))).any()):
        raise ValueError(f"Quantized bbox coordinate outside [0,{int(bins) - 1}]")
    return result


def _masked_mean(error: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=error.device, dtype=error.dtype)
    if mask.shape != error.shape:
        mask = mask.expand_as(error)
    denominator = mask.sum()
    if not bool(denominator > 0):
        return error.sum() * 0.0
    return (error * mask).sum() / denominator


def CoShadowFlowMatchLoss(pipe, **inputs):
    timestep_id, model_timestep, _, noisy_latents, velocity_target = safe._sample_flow_state(
        pipe,
        inputs["input_latents"],
        inputs,
    )
    inputs["latents"] = noisy_latents
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    output = pipe.model_fn(**models, **inputs, timestep=model_timestep)
    if isinstance(output, tuple):
        velocity_pred, shadow_mask_logits, alignment_prob = output
    else:
        velocity_pred, shadow_mask_logits, alignment_prob = output, None, None
    if "first_frame_latents" in inputs:
        raise NotImplementedError("CoShadow fixed32 training supports the one-frame cached path only")

    latent_raw = F.mse_loss(velocity_pred.float(), velocity_target.float())
    scheduler_weight = safe._scheduler_training_weight(pipe, timestep_id, latent_raw)
    latent_loss = float(inputs.get("coshadow_latent_loss_weight", 1.0)) * scheduler_weight * latent_raw
    total = latent_loss
    metrics = {
        "train/coshadow/latent_raw_mse": latent_raw.detach(),
        "train/coshadow/latent_loss": latent_loss.detach(),
        "train/coshadow/scheduler_weight": scheduler_weight.detach(),
    }

    shadow_mask = inputs.get("coshadow_shadow_mask")
    mask_weight = float(inputs.get("coshadow_mask_loss_weight", 0.0))
    if mask_weight != 0.0:
        if shadow_mask_logits is None:
            raise RuntimeError("Shadow-mask loss is enabled but the spatial model returned no mask logits")
        if not isinstance(shadow_mask, torch.Tensor):
            raise ValueError("Shadow-mask loss requires coshadow_shadow_mask")
        mask_target = F.interpolate(
            shadow_mask.float().to(device=shadow_mask_logits.device),
            size=shadow_mask_logits.shape[2:],
            mode="area",
        ).clamp(0, 1)
        positive_weight = float(inputs.get("coshadow_mask_positive_weight", 1.0))
        point_weight = 1.0 + (positive_weight - 1.0) * mask_target
        mask_raw = F.binary_cross_entropy_with_logits(
            shadow_mask_logits.float(),
            mask_target,
            weight=point_weight,
        )
        mask_loss = mask_weight * mask_raw
        total = total + mask_loss
        metrics["train/coshadow/mask_bce"] = mask_raw.detach()
        metrics["train/coshadow/mask_loss"] = mask_loss.detach()

    alignment_weight = float(inputs.get("coshadow_attention_alignment_weight", 0.0))
    if alignment_weight != 0.0:
        if alignment_prob is None or not isinstance(shadow_mask, torch.Tensor):
            raise RuntimeError("Attention alignment requires box context tokens and a GT shadow mask")
        token_count = int(alignment_prob.shape[-1])
        side = int(round(token_count ** 0.5))
        if side * side != token_count:
            raise ValueError("CoShadow fixed32 alignment expects a square one-frame token grid")
        target = F.interpolate(
            shadow_mask.float().to(device=alignment_prob.device),
            size=(1, side, side),
            mode="area",
        ).flatten(1)
        target_sum = target.sum(dim=1, keepdim=True)
        valid = target_sum.squeeze(1) > 0
        if bool(valid.any()):
            target = target / target_sum.clamp_min(1e-8)
            # Eq. (6): KL(A_shadow || T_shadow), stabilized for binary masks.
            attention = alignment_prob[:, valid].clamp_min(1e-8)
            target_distribution = target[valid].clamp_min(1e-8)
            align_raw = (
                attention * (attention.log() - target_distribution.unsqueeze(0).log())
            ).sum(dim=-1).mean()
        else:
            align_raw = alignment_prob.sum() * 0.0
        align_loss = alignment_weight * align_raw
        total = total + align_loss
        metrics["train/coshadow/attention_alignment_kl"] = align_raw.detach()
        metrics["train/coshadow/attention_alignment_loss"] = align_loss.detach()

    background_weight = float(inputs.get("coshadow_background_loss_weight", 0.0))
    if background_weight != 0.0:
        if not isinstance(shadow_mask, torch.Tensor):
            raise ValueError("Background loss requires coshadow_shadow_mask")
        sigma = pipe.scheduler.sigmas[timestep_id]
        sigma = torch.as_tensor(sigma, device=velocity_pred.device, dtype=velocity_pred.dtype)
        sigma = sigma.reshape((1,) + (1,) * (velocity_pred.ndim - 1))
        pred_current = inputs["input_latents"] + sigma * velocity_pred
        target_current = noisy_latents.detach()
        latent_shadow = F.interpolate(
            shadow_mask.float().to(device=velocity_pred.device),
            size=velocity_pred.shape[2:],
            mode="area",
        ).clamp(0, 1)
        background = 1.0 - latent_shadow
        background_raw = _masked_mean((pred_current.float() - target_current.float()).abs(), background)
        background_loss = background_weight * background_raw
        total = total + background_loss
        metrics["train/coshadow/background_l1"] = background_raw.detach()
        metrics["train/coshadow/background_loss"] = background_loss.detach()

    metrics["train/coshadow/total_loss"] = total.detach()
    pipe._tokenlight_loss_metrics = metrics
    return total


class _ObjectMaskCacheDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, cache_dir: str, mask_key: str, shard_lru: int) -> None:
        self.dataset = dataset
        self.store = base.VaeLatentCacheStore(
            base._resolve_path(cache_dir, base_path=REPO_ROOT),
            shard_lru=shard_lru,
        )
        self.mask_key = str(mask_key)
        self.data = dataset.data
        self.repeat = getattr(dataset, "repeat", 1)
        self.load_from_cache = getattr(dataset, "load_from_cache", False)
        self.is_vae_latent_cache_dataset = True

    def __len__(self) -> int:
        return len(self.dataset)

    def _get(self, index: int) -> dict[str, Any]:
        item = dict(self.dataset[index])
        value = item.get(self.mask_key)
        if not isinstance(value, str) or not value:
            raise KeyError(f"Missing object-mask key {self.mask_key!r}")
        item["_tokenlight_mask_latents"] = self.store.get(value)
        return item

    def __getitem__(self, index: int):
        return self._get(int(index))

    def __getitems__(self, indices: list[int]):
        return [self._get(int(index)) for index in indices]


class TokenLightCoShadowTrainingModule(base.TokenLightWanTrainingModule):
    def __init__(
        self,
        *args,
        coshadow_bbox_source: str = "gt",
        coshadow_bbox_bins: int = 16,
        coshadow_box_predictor_checkpoint: str | None = None,
        coshadow_box_predictor_image_size: int = 256,
        coshadow_box_predictor_presence_threshold: float = 0.5,
        coshadow_bbox_bins_key: str = "coshadow_bbox_bins",
        coshadow_bbox_valid_key: str = "coshadow_bbox_valid",
        coshadow_object_mask_key: str = "mask",
        coshadow_shadow_mask_key: str = "shadow_mask",
        coshadow_latent_loss_weight: float = 1.0,
        coshadow_mask_loss_weight: float = 0.1,
        coshadow_mask_positive_weight: float = 4.0,
        coshadow_background_loss_weight: float = 0.0,
        coshadow_attention_alignment_weight: float = 0.0,
        coshadow_category_key: str = "coshadow_category",
        coshadow_alignment_query_chunk_size: int = 256,
        dataset_base_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        kwargs["tokenlight_light_tokens"] = True
        kwargs["tokenlight_source_tokens"] = True
        kwargs["tokenlight_mask_tokens"] = True
        super().__init__(*args, **kwargs)
        self.dataset_base_path = dataset_base_path
        self.coshadow_bbox_source = str(coshadow_bbox_source)
        self.coshadow_bbox_bins = int(coshadow_bbox_bins)
        self.coshadow_box_predictor_image_size = int(coshadow_box_predictor_image_size)
        self.coshadow_box_predictor_presence_threshold = float(coshadow_box_predictor_presence_threshold)
        self.coshadow_bbox_bins_key = str(coshadow_bbox_bins_key)
        self.coshadow_bbox_valid_key = str(coshadow_bbox_valid_key)
        self.coshadow_object_mask_key = str(coshadow_object_mask_key)
        self.coshadow_shadow_mask_key = str(coshadow_shadow_mask_key)
        self.coshadow_category_key = str(coshadow_category_key)
        self.coshadow_alignment_query_chunk_size = int(coshadow_alignment_query_chunk_size)
        self.coshadow_loss_options = {
            "coshadow_latent_loss_weight": float(coshadow_latent_loss_weight),
            "coshadow_mask_loss_weight": float(coshadow_mask_loss_weight),
            "coshadow_mask_positive_weight": float(coshadow_mask_positive_weight),
            "coshadow_background_loss_weight": float(coshadow_background_loss_weight),
            "coshadow_attention_alignment_weight": float(coshadow_attention_alignment_weight),
        }
        token_dim = int(kwargs.get("tokenlight_token_dim", 0) or self.pipe.dit.dim)
        self.coshadow_layout_embedding = CoShadowLayoutTokenEmbedding(token_dim, bins=self.coshadow_bbox_bins)
        self.coshadow_prompt_embedding = None
        self.coshadow_alignment_head = None
        self.coshadow_mask_head = (
            CoShadowLatentMaskHead(token_dim)
            if float(coshadow_mask_loss_weight) != 0.0
            else None
        )
        self.coshadow_box_predictor = None
        if self.coshadow_bbox_source == "predicted":
            if coshadow_box_predictor_checkpoint in (None, "", "None", "none", "null"):
                raise ValueError("predicted bbox mode requires coshadow_box_predictor_checkpoint")
            self.coshadow_box_predictor = load_coshadow_box_predictor(
                coshadow_box_predictor_checkpoint,
                device="cpu",
                dtype=torch.float32,
            )
            self.coshadow_box_predictor.eval()
            for parameter in self.coshadow_box_predictor.parameters():
                parameter.requires_grad_(False)

        checkpoint_state = base._load_checkpoint_state_dict(
            kwargs.get("lora_checkpoint") or kwargs.get("resume_from_checkpoint")
        )
        base._load_module_state_from_checkpoint(
            self.coshadow_layout_embedding,
            checkpoint_state,
            "coshadow_layout_embedding",
        )
        base._load_module_state_from_checkpoint(
            self.coshadow_mask_head,
            checkpoint_state,
            "coshadow_mask_head",
        )
        base._load_module_state_from_checkpoint(
            self.coshadow_prompt_embedding, checkpoint_state, "coshadow_prompt_embedding"
        )
        base._load_module_state_from_checkpoint(
            self.coshadow_alignment_head, checkpoint_state, "coshadow_alignment_head"
        )
        self.pipe.model_fn = self._coshadow_model_fn
        self.task_to_loss["sft"] = self._coshadow_sft_loss
        self.task_to_loss["sft:train"] = self._coshadow_sft_loss

    def train(self, mode: bool = True):
        result = super().train(mode)
        if self.coshadow_box_predictor is not None:
            self.coshadow_box_predictor.eval()
        return result

    def _coshadow_model_fn(self, **kwargs):
        return model_fn_wan_video_tokenlight_coshadow(
            tokenlight_light_encoder=self.light_encoder,
            tokenlight_type_embedding=self.tokenlight_type_embedding,
            coshadow_layout_embedding=self.coshadow_layout_embedding,
            coshadow_prompt_embedding=None,
            coshadow_collect_actual_alignment=(
                self.coshadow_loss_options["coshadow_attention_alignment_weight"] != 0.0
            ),
            coshadow_alignment_query_chunk_size=self.coshadow_alignment_query_chunk_size,
            coshadow_mask_head=self.coshadow_mask_head,
            **kwargs,
        )

    def _coshadow_sft_loss(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        del inputs_nega
        merged = dict(inputs_shared)
        merged.update(inputs_posi)
        merged.update(self.coshadow_loss_options)
        return CoShadowFlowMatchLoss(pipe, **merged)

    @staticmethod
    def _batch_values(value: Any, batch: int, name: str) -> list[Any]:
        if batch == 1:
            if isinstance(value, list) and len(value) == 1 and not isinstance(value[0], (int, float)):
                return value
            return [value]
        if not isinstance(value, list) or len(value) != batch:
            raise ValueError(f"Expected {batch} values for {name!r}, got {type(value).__name__}")
        return value

    def _load_images(self, value: Any, *, batch: int, name: str, mode: str) -> list[Image.Image]:
        values = self._batch_values(value, batch, name)
        images = []
        for item in values:
            while isinstance(item, (list, tuple)) and len(item) == 1:
                item = item[0]
            if isinstance(item, Image.Image):
                image = item.convert(mode)
            elif isinstance(item, (str, Path)):
                path = Path(item)
                if not path.is_absolute():
                    path = Path(self.dataset_base_path or ".") / path
                if not path.is_file():
                    raise FileNotFoundError(f"Missing {name}: {path}")
                with Image.open(path) as opened:
                    image = opened.convert(mode).copy()
            else:
                raise TypeError(f"Unsupported {name} value: {type(item).__name__}")
            images.append(image)
        return images

    @staticmethod
    def _mask_tensor(images: list[Image.Image], *, height: int, width: int) -> torch.Tensor:
        tensors = []
        for image in images:
            image = image.resize((width, height), Image.Resampling.NEAREST).convert("L")
            tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            tensors.append(tensor.reshape(height, width).float().div_(255.0))
        return torch.stack(tensors).unsqueeze(1).unsqueeze(2)

    @staticmethod
    def _rgb_tensor(images: list[Image.Image], *, size: int) -> torch.Tensor:
        tensors = []
        for image in images:
            image = image.resize((size, size), Image.Resampling.BICUBIC).convert("RGB")
            tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            tensors.append(
                tensor.reshape(size, size, 3).permute(2, 0, 1).float().div_(255.0).mul_(2.0).sub_(1.0)
            )
        return torch.stack(tensors)

    def _layout_from_data(
        self,
        data: Mapping[str, Any],
        *,
        batch: int,
        device: torch.device,
        object_images: list[Image.Image],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.coshadow_bbox_source == "gt":
            if self.coshadow_bbox_bins_key not in data or self.coshadow_bbox_valid_key not in data:
                raise KeyError("GT bbox mode requires CoShadow builder bbox fields")
            return (
                _bbox_bins_batch(
                    data[self.coshadow_bbox_bins_key],
                    batch=batch,
                    bins=self.coshadow_bbox_bins,
                    device=device,
                ),
                _bool_batch(data[self.coshadow_bbox_valid_key], batch=batch, device=device),
            )

        if self.coshadow_box_predictor is None:
            raise RuntimeError("Predicted bbox mode has no loaded box predictor")
        source_images = self._load_images(data.get("input_image"), batch=batch, name="input_image", mode="RGB")
        source = self._rgb_tensor(source_images, size=self.coshadow_box_predictor_image_size).to(device=device)
        masks = self._mask_tensor(
            object_images,
            height=self.coshadow_box_predictor_image_size,
            width=self.coshadow_box_predictor_image_size,
        ).squeeze(2).to(device=device)
        attrs = attrs_from_batch(data, key=self.tokenlight_attrs_key)
        self.coshadow_box_predictor.eval()
        with torch.no_grad():
            outputs = self.coshadow_box_predictor(source, masks, attrs)
        valid = outputs["presence_logits"].sigmoid() >= self.coshadow_box_predictor_presence_threshold
        bins = quantize_boxes_xyxy(outputs["boxes"], bins=self.coshadow_bbox_bins)
        bins = torch.where(valid[:, None], bins, torch.zeros_like(bins))
        return bins, valid

    def _attach_coshadow(self, inputs, data, *, cached: bool):
        inputs_shared, inputs_posi, inputs_nega = inputs
        batch = int(inputs_shared["input_latents"].shape[0])
        categories = self._batch_values(data.get(self.coshadow_category_key, "object"), batch, self.coshadow_category_key)
        category_prompts = [f"{str(category).strip() or 'object'} casting shadow" for category in categories]
        inputs_posi["prompt"] = category_prompts
        inputs_posi["context"] = self._encode_prompts(category_prompts)
        height, width = int(inputs_shared["height"]), int(inputs_shared["width"])
        object_images = self._load_images(
            data.get(self.coshadow_object_mask_key),
            batch=batch,
            name=self.coshadow_object_mask_key,
            mode="L",
        )
        if cached and data.get("_tokenlight_mask_latents") is None:
            inputs_shared["tokenlight_mask_latents"] = self._encode_video_latents_batched(
                [[image.convert("RGB")] for image in object_images],
                inputs_shared,
            )
        layout_bins, bbox_valid = self._layout_from_data(
            data,
            batch=batch,
            device=inputs_shared["input_latents"].device,
            object_images=object_images,
        )
        inputs_shared["coshadow_layout_bins"] = layout_bins
        inputs_shared["coshadow_bbox_valid"] = bbox_valid
        if (
            self.coshadow_loss_options["coshadow_mask_loss_weight"] != 0.0
            or self.coshadow_loss_options["coshadow_background_loss_weight"] != 0.0
        ):
            shadow_images = self._load_images(
                data.get(self.coshadow_shadow_mask_key),
                batch=batch,
                name=self.coshadow_shadow_mask_key,
                mode="L",
            )
            inputs_shared["coshadow_shadow_mask"] = self._mask_tensor(
                shadow_images,
                height=height,
                width=width,
            )
        inputs_shared.update(self.coshadow_loss_options)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None and self._has_cached_latents(data):
            return self.task_to_loss[self.task](
                self.pipe,
                *self._attach_coshadow(self._cached_batched_inputs(data), data, cached=True),
            )
        if inputs is None and self._is_collated_batch(data):
            return self.task_to_loss[self.task](
                self.pipe,
                *self._attach_coshadow(self._batched_inputs(data), data, cached=False),
            )
        raise NotImplementedError("CoShadow trainer requires batched raw data or fixed32 cached latents")

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        exported = super().export_trainable_state_dict(state_dict, remove_prefix=remove_prefix)
        for prefix in (
            "coshadow_layout_embedding.",
            "coshadow_prompt_embedding.",
            "coshadow_alignment_head.",
            "coshadow_mask_head.",
        ):
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    exported[key] = value
        return exported


def coshadow_parser(train_mode: str) -> argparse.ArgumentParser:
    parser = base.wan_parser(train_mode)
    parser.add_argument("--train_mode", choices=("single", "zero3"), default=train_mode)
    parser.add_argument("--mixed_precision", choices=("no", "bf16"), default="bf16" if train_mode == "single" else None)
    parser.add_argument("--trainable_parameter_dtype", choices=("auto", "fp32", "bf16"), default="auto")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--tokenlight_rgb_decoder_loss_weight", type=float, default=0.0)
    parser.add_argument("--coshadow_bbox_source", choices=("gt", "predicted"), default="gt")
    parser.add_argument("--coshadow_bbox_bins", type=int, default=16)
    parser.add_argument("--coshadow_box_predictor_checkpoint", default=None)
    parser.add_argument("--coshadow_box_predictor_image_size", type=int, default=256)
    parser.add_argument("--coshadow_box_predictor_presence_threshold", type=float, default=0.5)
    parser.add_argument("--coshadow_bbox_bins_key", default="coshadow_bbox_bins")
    parser.add_argument("--coshadow_bbox_valid_key", default="coshadow_bbox_valid")
    parser.add_argument("--coshadow_object_mask_key", default="mask")
    parser.add_argument("--coshadow_shadow_mask_key", default="shadow_mask")
    parser.add_argument("--coshadow_object_mask_latent_cache_dir", default=None)
    parser.add_argument("--coshadow_latent_loss_weight", type=float, default=1.0)
    parser.add_argument("--coshadow_mask_loss_weight", type=float, default=0.1)
    parser.add_argument("--coshadow_mask_positive_weight", type=float, default=4.0)
    parser.add_argument("--coshadow_background_loss_weight", type=float, default=0.0)
    parser.add_argument("--coshadow_attention_alignment_weight", type=float, default=0.0)
    parser.add_argument("--coshadow_category_key", default="coshadow_category")
    parser.add_argument("--coshadow_alignment_query_chunk_size", type=int, default=256)
    parser.add_argument("--validate_dataset_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--validate_dataset_samples",
        type=int,
        default=128,
        help=(
            "Number of rows to fully decode and latent-cache samples to load in model-free "
            "validation; 0 means all rows."
        ),
    )
    return parser


def parse_coshadow_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--train_mode", choices=("single", "zero3"), default="single")
    pre.add_argument("--config", default=os.environ.get("COSHADOW_DIFFUSION_CONFIG", DEFAULT_CONFIG))
    pre_args, _ = pre.parse_known_args()
    raw = base._load_json_config(pre_args.config)
    parser = coshadow_parser(pre_args.train_mode)
    flat = base._flatten_train_config(raw)
    known = base._parser_destinations(parser)
    unknown = sorted(set(flat) - known)
    if unknown:
        parser.error(f"Unknown config key(s): {', '.join(unknown)}")
    base._apply_config_defaults(parser, raw)
    parser.set_defaults(config=pre_args.config, train_mode=pre_args.train_mode)
    args = parser.parse_args()
    if args.task not in {"sft", "sft:train"}:
        parser.error("CoShadow supports sft training only")
    if args.train_mode not in {"single", "zero3"}:
        parser.error(f"invalid train_mode: {args.train_mode}")
    if args.coshadow_bbox_source not in {"gt", "predicted"}:
        parser.error(f"invalid coshadow_bbox_source: {args.coshadow_bbox_source}")
    if args.trainable_parameter_dtype not in {"auto", "fp32", "bf16"}:
        parser.error(f"invalid trainable_parameter_dtype: {args.trainable_parameter_dtype}")
    if args.train_mode == "single" and args.mixed_precision not in {"no", "bf16"}:
        parser.error(f"invalid mixed_precision: {args.mixed_precision}")
    for name in (
        "coshadow_latent_loss_weight",
        "coshadow_mask_loss_weight",
        "coshadow_mask_positive_weight",
        "coshadow_background_loss_weight",
        "coshadow_attention_alignment_weight",
        "max_grad_norm",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            parser.error(f"{name} must be finite and non-negative")
    if args.coshadow_latent_loss_weight == 0 and args.coshadow_mask_loss_weight == 0 and args.coshadow_background_loss_weight == 0:
        parser.error("At least one CoShadow loss weight must be positive")
    if args.coshadow_bbox_bins < 2:
        parser.error("coshadow_bbox_bins must be at least 2")
    if int(args.coshadow_box_predictor_image_size) < 1:
        parser.error("coshadow_box_predictor_image_size must be positive")
    if int(args.coshadow_alignment_query_chunk_size) < 1:
        parser.error("coshadow_alignment_query_chunk_size must be positive")
    for name in ("batch_size", "gradient_accumulation_steps", "num_epochs"):
        if int(getattr(args, name)) < 1:
            parser.error(f"{name} must be positive")
    for name in ("learning_rate", "weight_decay"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            parser.error(f"{name} must be finite and non-negative")
    if not 0 <= args.coshadow_box_predictor_presence_threshold <= 1:
        parser.error("coshadow_box_predictor_presence_threshold must be in [0,1]")
    if (
        not args.validate_dataset_only
        and args.coshadow_bbox_source == "predicted"
        and args.coshadow_box_predictor_checkpoint in (None, "", "none", "null")
    ):
        parser.error("predicted bbox mode requires coshadow_box_predictor_checkpoint")
    if int(args.validate_dataset_samples) < 0:
        parser.error("validate_dataset_samples must be non-negative")
    if not args.tokenlight_source_tokens or not args.tokenlight_mask_tokens or not args.tokenlight_light_tokens:
        parser.error("CoShadow requires source, object-mask, and Lightoken conditioning")
    base._resolve_weight_paths(args)
    base._append_timestamp_to_output_path(args)
    return args, raw, raw


def _metadata_paths_for_validation(args) -> list[Path]:
    paths: list[Path] = []
    if args.dataset_metadata_path not in (None, "", "None", "none", "null"):
        paths.append(base._resolve_path(args.dataset_metadata_path, base_path=REPO_ROOT))
    paths.extend(path for path, _ in base._parse_vae_latent_cache_specs(args.vae_latent_cache_specs))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve(strict=False)
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    if not unique:
        raise ValueError("Dataset validation requires dataset_metadata_path or vae_latent_cache_specs")
    return unique


def _validation_png(root: Path, value: Any, *, key: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing non-empty PNG path {key!r}")
    path = Path(value)
    path = path if path.is_absolute() else root / path
    root_resolved = root.resolve(strict=True)
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"Path for {key!r} escapes dataset_base_path: {value!r}") from exc
    if resolved.suffix.lower() != ".png":
        raise ValueError(f"Expected PNG path for {key!r}, got {value!r}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _fully_decode_png(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError(f"File is not a PNG: {path}")
        image.verify()
    with Image.open(path) as image:
        image.load()
        return tuple(int(value) for value in image.size)


def _validated_light_attrs(value: Any, *, where: str) -> None:
    try:
        attrs = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where}: attrs_json is invalid JSON") from exc
    if not isinstance(attrs, Mapping):
        raise ValueError(f"{where}: attrs_json must be an object")
    lights = attrs.get("lights")
    if not isinstance(lights, list) or not lights or not isinstance(lights[0], Mapping):
        raise ValueError(f"{where}: attrs_json must contain lights[0]")
    for name in _FIXED32_LIGHT_FIELDS:
        try:
            number = float(lights[0].get(name))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}: lights[0].{name} must be numeric") from exc
        if not math.isfinite(number):
            raise ValueError(f"{where}: lights[0].{name} must be finite")


def _validated_bbox_fields(row: Mapping[str, Any], args, *, where: str) -> bool:
    bbox = row.get("coshadow_bbox_xyxy")
    valid = row.get(args.coshadow_bbox_valid_key)
    indices = row.get(args.coshadow_bbox_bins_key)
    if not isinstance(valid, bool):
        raise ValueError(f"{where}: {args.coshadow_bbox_valid_key} must be boolean")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"{where}: coshadow_bbox_xyxy must contain four values")
    try:
        coordinates = [float(value) for value in bbox]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: bbox coordinates must be numeric") from exc
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in coordinates):
        raise ValueError(f"{where}: bbox coordinates must be finite and normalized")
    if valid and not (coordinates[0] < coordinates[2] and coordinates[1] < coordinates[3]):
        raise ValueError(f"{where}: valid bbox must have positive width and height")
    if not valid and coordinates != [0.0, 0.0, 0.0, 0.0]:
        raise ValueError(f"{where}: invalid/empty bbox must use zero coordinates")
    if not isinstance(indices, list) or len(indices) != 4:
        raise ValueError(f"{where}: {args.coshadow_bbox_bins_key} must contain four indices")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in indices):
        raise ValueError(f"{where}: bbox bins must be integer indices")
    bins = int(args.coshadow_bbox_bins)
    if any(value < 0 or value >= bins for value in indices):
        raise ValueError(f"{where}: bbox bin outside [0,{bins - 1}]")
    expected = (
        [int(math.floor(value * (bins - 1) + 0.5)) for value in coordinates]
        if valid
        else [0, 0, 0, 0]
    )
    if indices != expected:
        raise ValueError(f"{where}: bbox bins {indices} do not match nearest-bin values {expected}")
    if int(row.get("coshadow_bbox_num_bins", -1)) != bins:
        raise ValueError(f"{where}: coshadow_bbox_num_bins must equal configured {bins}")
    return valid


def _assert_box_predictor_checkpoint_ready(args) -> None:
    if args.coshadow_bbox_source != "predicted":
        return
    checkpoint = base._resolve_path(args.coshadow_box_predictor_checkpoint, base_path=REPO_ROOT)
    if checkpoint.is_dir():
        config_path = checkpoint / "config.json"
        model_paths = (checkpoint / "model.safetensors", checkpoint / "model.pt")
        if not config_path.is_file():
            raise FileNotFoundError(f"Box predictor checkpoint is missing {config_path}")
        if not any(path.is_file() for path in model_paths):
            raise FileNotFoundError(
                f"Box predictor checkpoint has no model.safetensors or model.pt: {checkpoint}"
            )
    elif not checkpoint.is_file():
        raise FileNotFoundError(f"Box predictor checkpoint does not exist: {checkpoint}")
    args.coshadow_box_predictor_checkpoint = str(checkpoint)


def validate_coshadow_dataset(args) -> dict[str, Any]:
    """Validate metadata, PNGs, and cache access without loading either model."""

    root = base._resolve_path(args.dataset_base_path, base_path=REPO_ROOT).resolve(strict=True)
    attrs_key = str(args.tokenlight_attrs_key)
    image_keys = (
        "video",
        "input_image",
        str(args.coshadow_object_mask_key),
        str(args.coshadow_shadow_mask_key),
    )
    sample_limit = int(args.validate_dataset_samples)
    decoded_sizes: dict[Path, tuple[int, int]] = {}
    metadata_summaries: list[dict[str, Any]] = []
    for metadata_path in _metadata_paths_for_validation(args):
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        rows = base._read_jsonl_rows(metadata_path)
        if not rows:
            raise ValueError(f"No rows in {metadata_path}")
        expected_split = metadata_path.stem if metadata_path.stem in {"train", "val", "test"} else None
        decoded_rows = len(rows) if sample_limit == 0 else min(len(rows), sample_limit)
        valid_boxes = 0
        scenes: set[str] = set()
        for index, row in enumerate(rows):
            where = f"{metadata_path}:{index + 1}"
            if not isinstance(row, Mapping):
                raise TypeError(f"{where}: metadata row must be an object")
            if row.get("valid") is False:
                raise ValueError(f"{where}: source row is marked valid=false")
            if expected_split is not None and row.get("coshadow_split") != expected_split:
                raise ValueError(
                    f"{where}: coshadow_split={row.get('coshadow_split')!r}, expected {expected_split!r}"
                )
            scene = row.get("scene_id", row.get("scene_folder"))
            if not isinstance(scene, str) or not scene:
                raise ValueError(f"{where}: missing scene_id/scene_folder")
            scenes.add(scene)
            valid_boxes += int(_validated_bbox_fields(row, args, where=where))
            _validated_light_attrs(row.get(attrs_key), where=where)
            if index >= decoded_rows:
                continue
            row_sizes: dict[str, tuple[int, int]] = {}
            for key in image_keys:
                path = _validation_png(root, row.get(key), key=key)
                if path not in decoded_sizes:
                    decoded_sizes[path] = _fully_decode_png(path)
                row_sizes[key] = decoded_sizes[path]
            if len(set(row_sizes.values())) != 1:
                raise ValueError(f"{where}: source/target/object/shadow size mismatch: {row_sizes}")
            expected_size = (int(args.width), int(args.height))
            if next(iter(row_sizes.values())) != expected_size:
                raise ValueError(f"{where}: expected PNG size {expected_size}, got {row_sizes}")
        metadata_summaries.append(
            {
                "path": str(metadata_path),
                "rows": len(rows),
                "scenes": len(scenes),
                "valid_boxes": valid_boxes,
                "empty_boxes": len(rows) - valid_boxes,
                "fully_decoded_rows": decoded_rows,
            }
        )

    dataset = build_coshadow_dataset(args)
    cache_mode = bool(getattr(dataset, "is_vae_latent_cache_dataset", False))
    loaded_cache_samples = 0
    if cache_mode:
        count = len(dataset) if sample_limit == 0 else min(len(dataset), sample_limit)
        for index in range(count):
            item = dataset[index]
            for key in ("_tokenlight_input_latents", "_tokenlight_source_latents"):
                value = item.get(key)
                if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                    raise ValueError(f"Dataset sample {index} has invalid cached tensor {key!r}")
            loaded_cache_samples += 1
    return {
        "schema": "fixed32_coshadow_diffusion_validation_v1",
        "model_loaded": False,
        "box_predictor_loaded": False,
        "metadata": metadata_summaries,
        "unique_pngs_fully_decoded": len(decoded_sizes),
        "dataset_rows": len(dataset),
        "latent_cache_mode": cache_mode,
        "latent_cache_samples_loaded": loaded_cache_samples,
        "bbox_source_requested": str(args.coshadow_bbox_source),
    }


def build_coshadow_dataset(args):
    dataset = base.build_dataset(args)
    required = {
        "input_image",
        args.coshadow_object_mask_key,
        args.coshadow_shadow_mask_key,
        args.tokenlight_attrs_key,
        "coshadow_bbox_xyxy",
        args.coshadow_bbox_bins_key,
        args.coshadow_bbox_valid_key,
        "coshadow_bbox_num_bins",
    }
    missing = [
        (index, sorted(key for key in required if key not in row))
        for index, row in enumerate(dataset.data)
        if any(key not in row for key in required)
    ]
    if missing:
        raise ValueError(f"CoShadow metadata contract failed: first missing row={missing[0]}")
    for index, row in enumerate(dataset.data):
        where = f"training metadata row {index}"
        if row.get("coshadow_split") != "train":
            raise ValueError(f"{where}: expected coshadow_split='train'")
        _validated_bbox_fields(row, args, where=where)
        _validated_light_attrs(row.get(args.tokenlight_attrs_key), where=where)
    cache_dir = args.coshadow_object_mask_latent_cache_dir
    if cache_dir not in (None, "", "None", "none", "null"):
        if not getattr(dataset, "is_vae_latent_cache_dataset", False):
            raise ValueError("Object-mask latent cache requires vae_latent_cache_specs")
        dataset = _ObjectMaskCacheDataset(
            dataset,
            cache_dir,
            args.coshadow_object_mask_key,
            int(args.vae_latent_cache_shard_lru),
        )
    return dataset


def main() -> None:
    args, raw, merged = parse_coshadow_args()
    if args.validate_dataset_only:
        print(json.dumps(validate_coshadow_dataset(args), indent=2, sort_keys=True))
        return
    # Check the small frozen predictor before constructing the 5B Wan model.
    _assert_box_predictor_checkpoint_ready(args)
    args.coshadow_implementation_version = IMPLEMENTATION_VERSION
    args.coshadow_layout_injection = "semantic_plus_four_layout_cross_attention_context_tokens"
    args.coshadow_attention_alignment_status = "actual_wan_cross_attention_probability_kl"
    accelerator_kwargs = {
        "kwargs_handlers": [
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)
        ]
    }
    if args.train_mode == "single":
        accelerator_kwargs["gradient_accumulation_steps"] = args.gradient_accumulation_steps
        accelerator_kwargs["mixed_precision"] = args.mixed_precision
    accelerator = accelerate.Accelerator(**accelerator_kwargs)
    is_deepspeed = getattr(getattr(accelerator, "state", None), "deepspeed_plugin", None) is not None
    if args.train_mode == "zero3" and not is_deepspeed:
        raise RuntimeError("train_mode=zero3 requires an Accelerate DeepSpeed plugin/config")
    if args.train_mode == "single" and is_deepspeed:
        raise RuntimeError("train_mode=single cannot run with an active DeepSpeed plugin")
    base.save_training_config_snapshot(args, raw, merged, accelerator)
    dataset = build_coshadow_dataset(args)
    model = TokenLightCoShadowTrainingModule(
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
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
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
        dataset_base_path=args.dataset_base_path,
        coshadow_bbox_source=args.coshadow_bbox_source,
        coshadow_bbox_bins=args.coshadow_bbox_bins,
        coshadow_box_predictor_checkpoint=args.coshadow_box_predictor_checkpoint,
        coshadow_box_predictor_image_size=args.coshadow_box_predictor_image_size,
        coshadow_box_predictor_presence_threshold=args.coshadow_box_predictor_presence_threshold,
        coshadow_bbox_bins_key=args.coshadow_bbox_bins_key,
        coshadow_bbox_valid_key=args.coshadow_bbox_valid_key,
        coshadow_object_mask_key=args.coshadow_object_mask_key,
        coshadow_shadow_mask_key=args.coshadow_shadow_mask_key,
        coshadow_latent_loss_weight=args.coshadow_latent_loss_weight,
        coshadow_mask_loss_weight=args.coshadow_mask_loss_weight,
        coshadow_mask_positive_weight=args.coshadow_mask_positive_weight,
        coshadow_background_loss_weight=args.coshadow_background_loss_weight,
        coshadow_attention_alignment_weight=args.coshadow_attention_alignment_weight,
        coshadow_category_key=args.coshadow_category_key,
        coshadow_alignment_query_chunk_size=args.coshadow_alignment_query_chunk_size,
    )
    logger = base._make_model_logger(args)
    physics_runtime.launch_physics_training_task(
        accelerator, dataset, model, logger, args=args
    )


if __name__ == "__main__":
    main()
