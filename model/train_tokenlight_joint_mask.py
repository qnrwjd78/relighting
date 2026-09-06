#!/usr/bin/env python3
from __future__ import annotations

"""Joint RGB flow matching and clean-condition shadow-mask prediction.

This isolated trainer reuses the exp_1 RGB scene-latent cache.  A new mask
predictor consumes only the clean source latent and exact light attributes,
predicts a full-resolution shadow mask, and supplies a learned additive spatial
condition to Wan.  GT masks are used only by BCE + Dice supervision.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import accelerate
import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import train_tokenlight as base  # noqa: E402
from model import train_tokenlight_decoder_safe as safe  # noqa: E402
from model import train_tokenlight_scene_cache_v2 as scene_cache  # noqa: E402
from model.tokenlight_joint_mask import (  # noqa: E402
    LightConditionedMaskPredictor,
    model_fn_wan_video_tokenlight_joint_mask,
)
from model.train_tokenlight_scene_cache_shadow_safe_retained import (  # noqa: E402
    DistributedSceneQuotaBatchSampler,
)
from model.train_tokenlight_scene_cache_v2_retained import (  # noqa: E402
    OneBasedRetainedModelLogger,
)


LOSS_IMPL_VERSION = "tokenlight_clean_source_light_joint_mask_v1"


def _resolve_mask_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    workspace_path = REPO_ROOT / path
    if workspace_path.is_file():
        return workspace_path
    return path


def _read_mask(path: Path, *, height: int, width: int) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Missing joint shadow-mask target: {path}")
    with Image.open(path) as opened:
        image = opened.convert("L")
        if image.size != (int(width), int(height)):
            raise ValueError(
                f"Joint shadow mask {path} has size {image.size}, expected {(width, height)}"
            )
        image = image.copy()
    tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = tensor.reshape(int(height), int(width)).float().div_(255.0)
    return tensor.unsqueeze(0).unsqueeze(0).contiguous()


class JointMaskSceneCacheDataset(torch.utils.data.Dataset):
    """Attach a full-resolution GT target while retaining cached RGB latents."""

    def __init__(self, dataset, *, mask_key: str, height: int, width: int) -> None:
        self.dataset = dataset
        self.mask_key = str(mask_key)
        self.height = int(height)
        self.width = int(width)
        self.data = dataset.data
        self.repeat = getattr(dataset, "repeat", 1)
        self.load_from_cache = getattr(dataset, "load_from_cache", False)
        self.is_vae_latent_cache_dataset = getattr(dataset, "is_vae_latent_cache_dataset", True)
        self.is_scene_latent_cache_dataset = getattr(dataset, "is_scene_latent_cache_dataset", True)
        self.skip_tokenlight_file_filter = getattr(dataset, "skip_tokenlight_file_filter", True)

    def __len__(self) -> int:
        return len(self.dataset)

    def _attach(self, item: dict[str, Any]) -> dict[str, Any]:
        value = item.get(self.mask_key)
        if not isinstance(value, (str, Path)) or not str(value):
            raise KeyError(f"Metadata row is missing joint mask key {self.mask_key!r}")
        item["_tokenlight_joint_shadow_mask"] = _read_mask(
            _resolve_mask_path(value),
            height=self.height,
            width=self.width,
        )
        return item

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._attach(self.dataset[int(index)])

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        getter = getattr(self.dataset, "__getitems__", None)
        items = getter(indices) if callable(getter) else [self.dataset[int(index)] for index in indices]
        return [self._attach(item) for item in items]


def _stack_mask_targets(value: Any, *, batch: int) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(0)
    elif isinstance(value, list):
        if len(value) != batch or not all(isinstance(item, torch.Tensor) for item in value):
            raise TypeError(f"Expected {batch} joint-mask tensors")
        tensor = torch.stack(value, dim=0)
    else:
        raise TypeError(f"Missing joint-mask tensor, got {type(value).__name__}")
    if tensor.ndim != 5 or tuple(tensor.shape[:3]) != (batch, 1, 1):
        raise ValueError(f"Joint-mask target must be [B,1,1,H,W], got {tuple(tensor.shape)}")
    return tensor.float().clamp_(0.0, 1.0)


def _soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probability = logits.sigmoid()
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + float(eps)) / (denominator + float(eps))).mean()


def JointRGBShadowMaskFlowMatchLoss(pipe, **inputs):
    timestep_id, model_timestep, _, noisy_latents, velocity_target = safe._sample_flow_state(
        pipe,
        inputs["input_latents"],
        inputs,
    )
    model_inputs = dict(inputs)
    model_inputs["latents"] = noisy_latents
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    output = pipe.model_fn(**models, **model_inputs, timestep=model_timestep)
    if not isinstance(output, tuple) or len(output) != 2:
        raise RuntimeError("Joint RGB/mask model must return (velocity, mask_logits)")
    velocity_pred, mask_logits = output

    latent_raw = F.mse_loss(velocity_pred.float(), velocity_target.float())
    scheduler_weight = safe._scheduler_training_weight(pipe, timestep_id, latent_raw)
    latent_weight = float(inputs.get("tokenlight_rgb_latent_loss_weight", 1.0))
    latent_loss = latent_weight * scheduler_weight * latent_raw

    target = inputs.get("tokenlight_joint_shadow_mask")
    if not isinstance(target, torch.Tensor):
        raise ValueError("Joint mask loss requires tokenlight_joint_shadow_mask")
    target = target.to(device=mask_logits.device, dtype=torch.float32)
    if tuple(target.shape[2:]) != tuple(mask_logits.shape[2:]):
        target = F.interpolate(target, size=mask_logits.shape[2:], mode="nearest")

    bce = F.binary_cross_entropy_with_logits(mask_logits.float(), target)
    dice = _soft_dice_loss(mask_logits.float(), target)
    bce_weight = float(inputs.get("tokenlight_joint_mask_bce_weight", 1.0))
    dice_weight = float(inputs.get("tokenlight_joint_mask_dice_weight", 0.1))
    mask_outer_weight = float(inputs.get("tokenlight_joint_mask_loss_weight", 0.1))
    mask_raw = bce_weight * bce + dice_weight * dice
    mask_loss = mask_outer_weight * mask_raw
    total = latent_loss + mask_loss

    with torch.no_grad():
        prediction_mean = mask_logits.float().sigmoid().mean()
        target_mean = target.mean()
    pipe._tokenlight_loss_metrics = {
        "train/joint_mask/latent_raw_mse": latent_raw.detach(),
        "train/joint_mask/latent_loss": latent_loss.detach(),
        "train/joint_mask/scheduler_weight": scheduler_weight.detach(),
        "train/joint_mask/bce": bce.detach(),
        "train/joint_mask/dice": dice.detach(),
        "train/joint_mask/mask_raw": mask_raw.detach(),
        "train/joint_mask/mask_loss": mask_loss.detach(),
        "train/joint_mask/predicted_fraction": prediction_mean.detach(),
        "train/joint_mask/target_fraction": target_mean.detach(),
        "train/joint_mask/total_loss": total.detach(),
    }
    return total


class TokenLightJointMaskTrainingModule(base.TokenLightWanTrainingModule):
    def __init__(
        self,
        *args,
        tokenlight_joint_mask_loss_weight: float = 0.1,
        tokenlight_joint_mask_bce_weight: float = 1.0,
        tokenlight_joint_mask_dice_weight: float = 0.1,
        tokenlight_joint_mask_source_channels: int = 48,
        tokenlight_joint_mask_base_channels: int = 96,
        tokenlight_joint_mask_condition_dim: int = 128,
        tokenlight_joint_mask_output_height: int = 480,
        tokenlight_joint_mask_output_width: int = 480,
        tokenlight_predicted_mask_condition_scale: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.light_encoder is None or not self.tokenlight_source_tokens:
            raise ValueError("Joint mask training requires light and clean source conditioning")
        if self.tokenlight_mask_tokens:
            raise ValueError("Joint mask training forbids external/GT tokenlight_mask_tokens")
        self.joint_mask_options = {
            "tokenlight_rgb_latent_loss_weight": 1.0,
            "tokenlight_joint_mask_loss_weight": float(tokenlight_joint_mask_loss_weight),
            "tokenlight_joint_mask_bce_weight": float(tokenlight_joint_mask_bce_weight),
            "tokenlight_joint_mask_dice_weight": float(tokenlight_joint_mask_dice_weight),
            "tokenlight_predicted_mask_condition_scale": float(
                tokenlight_predicted_mask_condition_scale
            ),
        }
        self.joint_mask_predictor = LightConditionedMaskPredictor(
            source_channels=int(tokenlight_joint_mask_source_channels),
            base_channels=int(tokenlight_joint_mask_base_channels),
            condition_dim=int(tokenlight_joint_mask_condition_dim),
            output_height=int(tokenlight_joint_mask_output_height),
            output_width=int(tokenlight_joint_mask_output_width),
        )
        checkpoint = kwargs.get("lora_checkpoint") or kwargs.get("resume_from_checkpoint")
        state = base._load_checkpoint_state_dict(checkpoint)
        base._load_module_state_from_checkpoint(
            self.joint_mask_predictor,
            state,
            "joint_mask_predictor",
        )
        self.pipe.model_fn = self._joint_mask_model_fn
        self.task_to_loss["sft"] = self._joint_mask_sft_loss
        self.task_to_loss["sft:train"] = self._joint_mask_sft_loss

    def _joint_mask_model_fn(self, **kwargs):
        call_kwargs = dict(kwargs)
        condition_scale = float(
            call_kwargs.pop(
                "tokenlight_predicted_mask_condition_scale",
                self.joint_mask_options["tokenlight_predicted_mask_condition_scale"],
            )
        )
        return model_fn_wan_video_tokenlight_joint_mask(
            joint_mask_predictor=self.joint_mask_predictor,
            tokenlight_light_encoder=self.light_encoder,
            tokenlight_type_embedding=self.tokenlight_type_embedding,
            tokenlight_predicted_mask_condition_scale=condition_scale,
            **call_kwargs,
        )

    def _joint_mask_sft_loss(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        del inputs_nega
        merged = dict(inputs_shared)
        merged.update(inputs_posi)
        merged.update(self.joint_mask_options)
        return JointRGBShadowMaskFlowMatchLoss(pipe, **merged)

    def _attach_joint_mask(self, inputs, data):
        inputs_shared, inputs_posi, inputs_nega = inputs
        batch = int(inputs_shared["input_latents"].shape[0])
        target = _stack_mask_targets(data.get("_tokenlight_joint_shadow_mask"), batch=batch)
        if tuple(target.shape[-2:]) != (
            int(inputs_shared["height"]),
            int(inputs_shared["width"]),
        ):
            raise ValueError(
                f"Joint mask/target size mismatch: {tuple(target.shape[-2:])} != "
                f"{(int(inputs_shared['height']), int(inputs_shared['width']))}"
            )
        inputs_shared["tokenlight_joint_shadow_mask"] = target.to(
            device=self.pipe.device,
            dtype=torch.float32,
            non_blocking=bool(target.is_pinned()),
        )
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None and self._has_cached_latents(data):
            prepared = self._attach_joint_mask(self._cached_batched_inputs(data), data)
            return self.task_to_loss[self.task](self.pipe, *prepared)
        if inputs is None and self._is_collated_batch(data):
            prepared = self._attach_joint_mask(self._batched_inputs(data), data)
            return self.task_to_loss[self.task](self.pipe, *prepared)
        raise NotImplementedError("Joint mask training requires batched data with cached RGB latents")

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        exported = super().export_trainable_state_dict(state_dict, remove_prefix=remove_prefix)
        for key, value in state_dict.items():
            if key.startswith("joint_mask_predictor."):
                exported[key] = value
        return exported


def joint_mask_parser(train_mode: str) -> argparse.ArgumentParser:
    parser = safe.safe_parser(train_mode)
    parser.description = "TokenLight clean source/light joint RGB + shadow-mask trainer."
    parser.add_argument("--tokenlight_joint_shadow_mask_key", default="shadow_mask")
    parser.add_argument("--tokenlight_joint_mask_loss_weight", type=float, default=0.1)
    parser.add_argument("--tokenlight_joint_mask_bce_weight", type=float, default=1.0)
    parser.add_argument("--tokenlight_joint_mask_dice_weight", type=float, default=0.1)
    parser.add_argument("--tokenlight_joint_mask_source_channels", type=int, default=48)
    parser.add_argument("--tokenlight_joint_mask_base_channels", type=int, default=96)
    parser.add_argument("--tokenlight_joint_mask_condition_dim", type=int, default=128)
    parser.add_argument("--tokenlight_joint_mask_output_height", type=int, default=480)
    parser.add_argument("--tokenlight_joint_mask_output_width", type=int, default=480)
    parser.add_argument("--tokenlight_predicted_mask_condition_scale", type=float, default=1.0)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight_max_samples", type=int, default=128)
    return parser


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--train_mode", choices=("single", "zero3"), default="single")
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    config_path = pre_args.config or os.environ.get(
        "TOKENLIGHT_TRAIN_CONFIG",
        "configs/train_480/exp2_7x7x5_power06_rgb_joint_shadow_mask_clean_20ep_b5x4_ga2_gb40.json",
    )
    raw_config = base._load_json_config(config_path)
    parser = joint_mask_parser(pre_args.train_mode)
    base._apply_config_defaults(parser, raw_config)
    parser.set_defaults(config=config_path, train_mode=pre_args.train_mode)
    args = parser.parse_args()
    raw_config = base._load_json_config(args.config)
    safe._validate_safe_args(args, raw_config, parser)

    finite_nonnegative = (
        "tokenlight_joint_mask_loss_weight",
        "tokenlight_joint_mask_bce_weight",
        "tokenlight_joint_mask_dice_weight",
        "tokenlight_predicted_mask_condition_scale",
    )
    for name in finite_nonnegative:
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            parser.error(f"{name} must be finite and non-negative")
    if float(args.tokenlight_joint_mask_loss_weight) <= 0:
        parser.error("joint mask loss must be nonzero on every DDP rank")
    if float(args.tokenlight_joint_mask_bce_weight) + float(args.tokenlight_joint_mask_dice_weight) <= 0:
        parser.error("joint mask BCE and Dice weights cannot both be zero")
    if not args.tokenlight_light_tokens or not args.tokenlight_source_tokens:
        parser.error("joint mask training requires tokenlight_light_tokens and tokenlight_source_tokens")
    if args.tokenlight_mask_tokens:
        parser.error("tokenlight_mask_tokens must be false; GT masks are output supervision only")
    if float(args.tokenlight_rgb_latent_loss_weight) != 1.0:
        parser.error("joint trainer preserves RGB latent flow loss weight at exactly 1.0")
    if float(args.tokenlight_rgb_decoder_loss_weight) != 0.0:
        parser.error("joint mask trainer does not run the Wan RGB VAE decoder")
    if int(args.tokenlight_joint_mask_source_channels) != 48:
        parser.error("current Wan2.2 scene cache requires 48 source latent channels")
    if (int(args.tokenlight_joint_mask_output_height), int(args.tokenlight_joint_mask_output_width)) != (
        int(args.height),
        int(args.width),
    ):
        parser.error("joint mask output size must equal the RGB training resolution")
    if int(args.preflight_max_samples) < 1:
        parser.error("preflight_max_samples must be positive")

    base._resolve_weight_paths(args)
    if not args.preflight:
        base._append_timestamp_to_output_path(args)
    return args, raw_config, raw_config


def build_dataset(args) -> JointMaskSceneCacheDataset:
    cached = scene_cache.build_scene_cache_dataset(args)
    return JointMaskSceneCacheDataset(
        cached,
        mask_key=args.tokenlight_joint_shadow_mask_key,
        height=args.height,
        width=args.width,
    )


def run_preflight(args) -> None:
    dataset = build_dataset(args)
    maximum = min(len(dataset), int(args.preflight_max_samples))
    if maximum <= 0:
        raise ValueError("Joint-mask preflight selected no samples")
    indices = [round(index * (len(dataset) - 1) / max(1, maximum - 1)) for index in range(maximum)]
    fractions = []
    latent_shapes = set()
    scenes = set()
    for index in indices:
        item = dataset[index]
        target = item["_tokenlight_joint_shadow_mask"]
        fractions.append(float(target.mean()))
        latent_shapes.add(tuple(item["_tokenlight_source_latents"].shape))
        scenes.add(str(item.get("scene_id") or item.get("scene_folder")))
    summary = {
        "ok": True,
        "metadata_rows": len(dataset),
        "checked": maximum,
        "checked_scenes": len(scenes),
        "source_latent_shapes": sorted(latent_shapes),
        "mask_fraction_min": min(fractions),
        "mask_fraction_mean": sum(fractions) / len(fractions),
        "mask_fraction_max": max(fractions),
        "mask_is_input": False,
        "mask_loss": "BCE + 0.1 Dice",
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _make_model(args) -> TokenLightJointMaskTrainingModule:
    return TokenLightJointMaskTrainingModule(
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
        device=("cpu" if args.initialize_model_on_cpu else args._accelerator_device),
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
        tokenlight_joint_mask_loss_weight=args.tokenlight_joint_mask_loss_weight,
        tokenlight_joint_mask_bce_weight=args.tokenlight_joint_mask_bce_weight,
        tokenlight_joint_mask_dice_weight=args.tokenlight_joint_mask_dice_weight,
        tokenlight_joint_mask_source_channels=args.tokenlight_joint_mask_source_channels,
        tokenlight_joint_mask_base_channels=args.tokenlight_joint_mask_base_channels,
        tokenlight_joint_mask_condition_dim=args.tokenlight_joint_mask_condition_dim,
        tokenlight_joint_mask_output_height=args.tokenlight_joint_mask_output_height,
        tokenlight_joint_mask_output_width=args.tokenlight_joint_mask_output_width,
        tokenlight_predicted_mask_condition_scale=args.tokenlight_predicted_mask_condition_scale,
    )


def main() -> None:
    args, raw_config, merged_config = parse_args()
    args.loss_impl_version = LOSS_IMPL_VERSION
    args.mask_target_usage = "output_supervision_only"
    args.mask_condition_source = "clean_source_vae_latent_plus_exact_light_attrs"
    args.mask_condition_injection = "zero_init_additive_target_tokens_no_sequence_growth"
    if args.preflight:
        run_preflight(args)
        return

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            )
        ],
    )
    args._accelerator_device = accelerator.device
    base.save_training_config_snapshot(args, raw_config, merged_config, accelerator)

    dataset = build_dataset(args)
    model = _make_model(args)
    base.BalancedTaskBatchSampler = DistributedSceneQuotaBatchSampler
    base.ModelLogger = OneBasedRetainedModelLogger
    model_logger = base._make_model_logger(args)
    safe.launch_safe_training_task(
        accelerator,
        dataset,
        model,
        model_logger,
        args=args,
    )


if __name__ == "__main__":
    main()
