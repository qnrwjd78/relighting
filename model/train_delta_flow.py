from __future__ import annotations

"""Same-scene dual-light delta-flow training for TokenLight.

This is an opt-in training entrypoint for the aligned VAE-cache pipeline used
by the RGB baseline.  It does not modify the base TokenLight model or
checkpoint format.  A structured batch contains same-scene light pairs
followed by ordinary singleton samples.  Pair members share the flow noise and
CFG light-drop decision, and a velocity-difference objective is added to the
existing absolute flow-matching objective.

The default batch-8 layout is::

    [A/light_i, A/light_j, B/light_i, B/light_j, C, D, E, F]

Inference remains unchanged because no new trainable module is introduced.
"""

import argparse
import heapq
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import accelerate
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import train as base  # noqa: E402
from model import train_decoder_safe as safe  # noqa: E402
from model.light_encoder import parse_attrs_json  # noqa: E402


LOSS_IMPL_VERSION = "tokenlight_same_scene_delta_flow_v1"
DEFAULT_DELTA_CONFIG_PATH = "configs/train_480/rgb_baseline_15ep_b8_ga40.json"
DELTA_LOSS_TYPES = ("mse", "charbonnier")


@dataclass(frozen=True)
class _PairRowInfo:
    group_key: tuple[str, ...]
    scene_id: str
    source_path: str
    position: tuple[float, float, float]


def _finite_position(light: Mapping[str, Any]) -> tuple[float, float, float] | None:
    if all(key in light for key in ("x", "y", "z")):
        raw = (light.get("x"), light.get("y"), light.get("z"))
    else:
        position = light.get("position")
        if not isinstance(position, Sequence) or isinstance(position, (str, bytes)) or len(position) < 3:
            return None
        raw = (position[0], position[1], position[2])
    try:
        result = tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in result):
        return None
    return result  # type: ignore[return-value]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _pair_row_info(row: Mapping[str, Any], *, attrs_key: str) -> _PairRowInfo | None:
    scene_id = str(row.get("scene_id") or row.get("scene_folder") or "").strip()
    source_path = str(row.get("input_image") or "").strip()
    if not scene_id or not source_path:
        return None

    try:
        attrs = parse_attrs_json(row.get(attrs_key, row.get("attrs")))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    lights = attrs.get("lights")
    if not isinstance(lights, Sequence) or isinstance(lights, (str, bytes)) or len(lights) != 1:
        return None
    light = lights[0]
    if not isinstance(light, Mapping):
        return None
    position = _finite_position(light)
    if position is None:
        return None

    # Pair only rows for which the complete lighting condition is identical
    # except for XYZ.  This keeps intensity, color, radius, ambient, task, and
    # any future metadata attributes fixed without maintaining a fragile list.
    static_light = {
        str(key): value
        for key, value in light.items()
        if str(key) not in {"x", "y", "z", "position"}
    }
    static_attrs = {str(key): value for key, value in attrs.items() if str(key) != "lights"}
    try:
        static_condition = _canonical_json({"attrs": static_attrs, "light": static_light})
    except (TypeError, ValueError):
        return None

    group_key = (
        str(row.get("_vae_latent_cache_source", "")),
        scene_id,
        source_path,
        str(row.get("prompt") or ""),
        str(row.get("task") or ""),
        static_condition,
    )
    return _PairRowInfo(
        group_key=group_key,
        scene_id=scene_id,
        source_path=source_path,
        position=position,
    )


def _distinct_position_pairs(
    virtual_indices: Sequence[int],
    *,
    base_size: int,
    infos: Sequence[_PairRowInfo | None],
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Return a maximum-size random matching that never pairs equal XYZ."""

    by_position: dict[tuple[float, float, float], list[int]] = defaultdict(list)
    for virtual_index in virtual_indices:
        info = infos[int(virtual_index) % base_size]
        if info is not None:
            by_position[info.position].append(int(virtual_index))
    heap: list[tuple[int, float, tuple[float, float, float], list[int]]] = []
    for position, indices in by_position.items():
        rng.shuffle(indices)
        heapq.heappush(heap, (-len(indices), rng.random(), position, indices))

    pairs: list[tuple[int, int]] = []
    while len(heap) >= 2:
        neg_a, _, position_a, indices_a = heapq.heappop(heap)
        neg_b, _, position_b, indices_b = heapq.heappop(heap)
        left = indices_a.pop()
        right = indices_b.pop()
        if rng.random() < 0.5:
            left, right = right, left
        pairs.append((left, right))
        if indices_a:
            heapq.heappush(heap, (neg_a + 1, rng.random(), position_a, indices_a))
        if indices_b:
            heapq.heappush(heap, (neg_b + 1, rng.random(), position_b, indices_b))
    return pairs


class MixedSameScenePairBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Pack disjoint strict light-position pairs and singletons into each batch.

    Every virtual dataset index is used at most once per epoch.  The sampler
    emits only full batches, matching the previous ``drop_last=True`` exposure.
    Pair slots always occupy the first ``2 * pairs_per_batch`` positions.
    """

    drop_last = False

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        dataset_length: int,
        batch_size: int,
        pairs_per_batch: int,
        attrs_key: str = "attrs_json",
        seed: int = 480,
    ) -> None:
        self.rows = rows
        self.base_size = len(rows)
        self.dataset_length = int(dataset_length)
        self.batch_size = int(batch_size)
        self.pairs_per_batch = int(pairs_per_batch)
        self.attrs_key = str(attrs_key)
        self.seed = int(seed)
        self.epoch = 0

        if self.base_size <= 0:
            raise ValueError("Delta pair sampler requires non-empty metadata rows")
        if self.dataset_length <= 0 or self.dataset_length % self.base_size != 0:
            raise ValueError(
                "Dataset length must be a positive integer multiple of metadata rows: "
                f"length={self.dataset_length}, rows={self.base_size}"
            )
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.pairs_per_batch <= 0:
            raise ValueError("pairs_per_batch must be positive")
        if 2 * self.pairs_per_batch > self.batch_size:
            raise ValueError(
                f"{self.pairs_per_batch} pairs do not fit in batch size {self.batch_size}"
            )
        self.repeat = self.dataset_length // self.base_size
        self._length = self.dataset_length // self.batch_size
        if self._length <= 0:
            raise ValueError("Dataset is smaller than one full batch")

        self.infos = [_pair_row_info(row, attrs_key=self.attrs_key) for row in self.rows]
        groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for index, info in enumerate(self.infos):
            if info is not None:
                groups[info.group_key].append(index)
        self.groups = {key: tuple(indices) for key, indices in groups.items() if len(indices) >= 2}
        self.pairable_rows = sum(len(indices) for indices in self.groups.values())
        if not self.groups:
            raise ValueError("No same-scene, same-condition, different-position groups were found")

        # Fail before model initialization if the requested epoch schedule is
        # impossible without replacement.
        dry_pairs = self._candidate_pairs(random.Random(self.seed))
        required_pairs = self._length * self.pairs_per_batch
        if len(dry_pairs) < required_pairs:
            raise ValueError(
                "Not enough disjoint same-scene pairs for one epoch: "
                f"required={required_pairs}, available={len(dry_pairs)}, "
                f"rows={self.dataset_length}"
            )

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _candidate_pairs(self, rng: random.Random) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        for repeat_id in range(self.repeat):
            offset = repeat_id * self.base_size
            group_items = list(self.groups.items())
            rng.shuffle(group_items)
            for _, base_indices in group_items:
                virtual = [offset + int(index) for index in base_indices]
                result.extend(
                    _distinct_position_pairs(
                        virtual,
                        base_size=self.base_size,
                        infos=self.infos,
                        rng=rng,
                    )
                )
        return result

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        candidate_pairs = self._candidate_pairs(rng)
        rng.shuffle(candidate_pairs)
        required_pairs = self._length * self.pairs_per_batch
        selected_pairs = candidate_pairs[:required_pairs]
        selected_indices = {index for pair in selected_pairs for index in pair}
        if len(selected_indices) != 2 * len(selected_pairs):
            raise RuntimeError("Internal error: pair schedule contains duplicate dataset indices")

        singleton_count = self.batch_size - 2 * self.pairs_per_batch
        required_singletons = self._length * singleton_count
        singletons = [index for index in range(self.dataset_length) if index not in selected_indices]
        rng.shuffle(singletons)
        if len(singletons) < required_singletons:
            raise RuntimeError(
                f"Not enough singleton rows: required={required_singletons}, available={len(singletons)}"
            )
        singletons = singletons[:required_singletons]
        rng.shuffle(selected_pairs)

        pair_cursor = 0
        singleton_cursor = 0
        for _ in range(self._length):
            batch: list[int] = []
            for _ in range(self.pairs_per_batch):
                left, right = selected_pairs[pair_cursor]
                pair_cursor += 1
                batch.extend((left, right))
            batch.extend(singletons[singleton_cursor : singleton_cursor + singleton_count])
            singleton_cursor += singleton_count
            if len(batch) != self.batch_size or len(set(batch)) != self.batch_size:
                raise RuntimeError("Internal error: malformed delta-flow batch")
            yield batch


def collate_delta_flow_batch(batch: list[dict], *, pairs_per_batch: int):
    result = base._collate_tokenlight_batch(batch)
    if not isinstance(result, dict):
        raise TypeError("TokenLight collate must return a dictionary")
    pair_slots = 2 * int(pairs_per_batch)
    if len(batch) < pair_slots:
        raise ValueError(f"Batch of {len(batch)} cannot contain {pairs_per_batch} pairs")
    result["_tokenlight_delta_pair_indices"] = torch.arange(
        pair_slots, dtype=torch.long
    ).reshape(int(pairs_per_batch), 2)
    return result


def _delta_ramp(inputs: Mapping[str, Any]) -> float:
    step = max(0, int(inputs.get("tokenlight_optimizer_step", 0)))
    warmup = max(0, int(inputs.get("tokenlight_delta_warmup_steps", 0)))
    ramp_steps = max(0, int(inputs.get("tokenlight_delta_ramp_steps", 0)))
    if step < warmup:
        return 0.0
    if ramp_steps == 0:
        return 1.0
    return min(1.0, max(0.0, (step - warmup + 1) / float(ramp_steps)))


def _normalized_sigma_weight(pipe, sigma: torch.Tensor, gamma: float) -> torch.Tensor:
    if gamma == 0.0:
        return torch.ones_like(sigma, dtype=torch.float32)
    sigmas = torch.as_tensor(pipe.scheduler.sigmas, dtype=torch.float32)
    weights = getattr(pipe.scheduler, "linear_timesteps_weights", None)
    if weights is None:
        weights_f = torch.ones_like(sigmas)
    else:
        weights_f = torch.as_tensor(weights, dtype=torch.float32).reshape_as(sigmas)
    normalizer = (weights_f * sigmas.clamp_min(0).pow(gamma)).sum() / weights_f.sum().clamp_min(1e-12)
    return sigma.float().clamp_min(0).pow(gamma) / normalizer.to(
        device=sigma.device, dtype=torch.float32
    ).clamp_min(1e-12)


def _sample_delta_flow_state(
    pipe,
    clean_latents: torch.Tensor,
    inputs: Mapping[str, Any],
    pair_indices: torch.Tensor,
):
    total = len(pipe.scheduler.timesteps)
    max_boundary = int(float(inputs.get("max_timestep_boundary", 1.0)) * total)
    min_boundary = int(float(inputs.get("min_timestep_boundary", 0.0)) * total)
    min_boundary = max(0, min(total, min_boundary))
    max_boundary = max(0, min(total, max_boundary))
    if max_boundary <= min_boundary:
        raise ValueError(
            f"Empty timestep interval: min={min_boundary}, max={max_boundary}, total={total}"
        )

    timestep_id = int(torch.randint(min_boundary, max_boundary, (1,), device="cpu").item())
    scheduler_timestep = pipe.scheduler.timesteps[timestep_id]
    model_timestep = scheduler_timestep.reshape(1).to(dtype=pipe.torch_dtype, device=pipe.device)
    sigma_scalar = safe._as_scalar_tensor(pipe.scheduler.sigmas[timestep_id], clean_latents)
    sigma = sigma_scalar.to(dtype=clean_latents.dtype).reshape(
        (1,) + (1,) * (clean_latents.ndim - 1)
    )

    noise = torch.randn_like(clean_latents) * float(inputs.get("noise_scale", 1.0))
    pair_indices = pair_indices.to(device=noise.device, dtype=torch.long)
    left = pair_indices[:, 0]
    right = pair_indices[:, 1]
    noise = noise.clone()
    noise.index_copy_(0, right, noise.index_select(0, left))
    noisy_latents = (1.0 - sigma) * clean_latents + sigma * noise
    velocity_target = noise - clean_latents
    return timestep_id, model_timestep, sigma_scalar, noisy_latents, velocity_target, noise


def _pointwise_delta_loss(error: torch.Tensor, *, loss_type: str, charbonnier_eps: float):
    if loss_type == "mse":
        return error.square().mean()
    if loss_type == "charbonnier":
        eps = float(charbonnier_eps)
        return (torch.sqrt(error.square() + eps**2) - eps).mean()
    raise ValueError(f"Unsupported delta loss type: {loss_type}")


def TokenLightDeltaFlowMatchLoss(pipe, **inputs):
    """Absolute flow matching plus same-scene relative velocity matching."""

    latent_weight = float(inputs.get("tokenlight_rgb_latent_loss_weight", 1.0))
    decoder_weight = float(inputs.get("tokenlight_rgb_decoder_loss_weight", 0.0))
    delta_weight = float(inputs.get("tokenlight_delta_loss_weight", 0.05))
    pair_indices = inputs.get("tokenlight_delta_pair_indices")
    if not isinstance(pair_indices, torch.Tensor) or pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
        raise ValueError("Delta-flow loss requires pair_indices with shape [P,2]")

    if "lora" in inputs:
        pipe.clear_lora(verbose=0)
        pipe.load_lora(pipe.dit, state_dict=inputs["lora"], hotload=True, verbose=0)

    clean_latents = inputs["input_latents"]
    pair_indices = pair_indices.to(device=clean_latents.device, dtype=torch.long)
    if pair_indices.numel() == 0:
        raise ValueError("Delta-flow loss requires at least one pair")
    if int(pair_indices.min()) < 0 or int(pair_indices.max()) >= int(clean_latents.shape[0]):
        raise IndexError("Delta pair index is outside the latent batch")
    if int(torch.unique(pair_indices).numel()) != int(pair_indices.numel()):
        raise ValueError("Delta pairs must not reuse a batch slot")

    timestep_id, timestep, sigma, noisy_latents, velocity_target, _ = _sample_delta_flow_state(
        pipe, clean_latents, inputs, pair_indices
    )
    model_latents = noisy_latents
    if "first_frame_latents" in inputs:
        model_latents = noisy_latents.clone()
        model_latents[:, :, 0:1] = inputs["first_frame_latents"]
    inputs["latents"] = model_latents

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    velocity_pred_full = pipe.model_fn(**models, **inputs, timestep=timestep)
    velocity_pred = velocity_pred_full
    latent_target = velocity_target
    if "first_frame_latents" in inputs:
        velocity_pred = velocity_pred[:, :, 1:]
        latent_target = latent_target[:, :, 1:]

    pred_f = velocity_pred.float()
    target_f = latent_target.float()
    clean_target_f = clean_latents.float()
    if "first_frame_latents" in inputs:
        clean_target_f = clean_target_f[:, :, 1:]
    absolute_mse = F.mse_loss(pred_f, target_f)
    scheduler_weight = safe._scheduler_training_weight(pipe, timestep_id, absolute_mse)
    absolute_term = latent_weight * scheduler_weight * absolute_mse

    left = pair_indices[:, 0]
    right = pair_indices[:, 1]
    delta_pred = pred_f.index_select(0, right) - pred_f.index_select(0, left)
    # With the shared-noise convention used above,
    # (noise-x_right) - (noise-x_left) = x_left-x_right.  Subtract the clean
    # BF16 latents only after promotion to FP32; subtracting two BF16 velocity
    # targets first loses a measurable amount of close-light delta signal.
    delta_target = (
        clean_target_f.index_select(0, left) - clean_target_f.index_select(0, right)
    ).detach()
    active = inputs.get("tokenlight_delta_pair_active")
    if active is None:
        active = torch.ones(pair_indices.shape[0], dtype=torch.bool, device=pred_f.device)
    else:
        active = torch.as_tensor(active, dtype=torch.bool, device=pred_f.device).reshape(-1)
    if active.numel() != pair_indices.shape[0]:
        raise ValueError("pair_active length does not match pair_indices")

    if bool(active.any()):
        delta_pred_valid = delta_pred[active]
        delta_target_valid = delta_target[active]
        delta_error = delta_pred_valid - delta_target_valid
        delta_raw = _pointwise_delta_loss(
            delta_error,
            loss_type=str(inputs.get("tokenlight_delta_loss_type", "mse")),
            charbonnier_eps=float(inputs.get("tokenlight_delta_charbonnier_eps", 1e-3)),
        )
        pred_delta_rms = delta_pred_valid.square().mean().sqrt()
        target_delta_rms = delta_target_valid.square().mean().sqrt()
        pred_flat = delta_pred_valid.flatten(1)
        target_flat = delta_target_valid.flatten(1)
        delta_cosine = F.cosine_similarity(pred_flat, target_flat, dim=1, eps=1e-8).mean()
    else:
        delta_raw = pred_f.sum() * 0.0
        pred_delta_rms = pred_f.new_zeros(())
        target_delta_rms = pred_f.new_zeros(())
        delta_cosine = pred_f.new_zeros(())

    ramp = _delta_ramp(inputs)
    sigma_factor = _normalized_sigma_weight(
        pipe, sigma, float(inputs.get("tokenlight_delta_sigma_gamma", 0.0))
    )
    effective_delta_weight = delta_raw.new_tensor(delta_weight * ramp) * sigma_factor
    delta_term = latent_weight * scheduler_weight * effective_delta_weight * delta_raw
    total = absolute_term + delta_term

    metrics = {
        "train/delta/absolute_mse": absolute_mse.detach(),
        "train/delta/absolute_term": absolute_term.detach(),
        "train/delta/delta_raw": delta_raw.detach(),
        "train/delta/delta_term": delta_term.detach(),
        "train/delta/ramp": absolute_mse.new_tensor(float(ramp)),
        "train/delta/effective_weight": effective_delta_weight.detach(),
        "train/delta/sigma_factor": sigma_factor.detach(),
        "train/delta/pairs_total": absolute_mse.new_tensor(float(pair_indices.shape[0])),
        "train/delta/pairs_valid": active.float().sum().detach(),
        "train/delta/pairs_dropped": (~active).float().sum().detach(),
        "train/delta/pred_rms": pred_delta_rms.detach(),
        "train/delta/target_rms": target_delta_rms.detach(),
        "train/delta/cosine": delta_cosine.detach(),
        "train/delta/sigma": sigma.detach(),
        "train/delta/scheduler_weight": scheduler_weight.detach(),
        "train/rgb/latent_mse": absolute_mse.detach(),
        "train/rgb/latent_term": absolute_term.detach(),
    }

    # Preserve the opt-in safe decoder objective when a decoder config is used.
    decoder_ramp = safe._decoder_ramp(inputs)
    if decoder_weight != 0.0 and decoder_ramp > 0.0:
        sigma_view = sigma.to(device=noisy_latents.device, dtype=noisy_latents.dtype).reshape(
            (1,) + (1,) * (noisy_latents.ndim - 1)
        )
        pred_current = clean_latents + sigma_view * velocity_pred_full
        target_current = model_latents.detach()
        exclude_first = "first_frame_latents" in inputs
        if exclude_first:
            pred_current = pred_current.clone()
            pred_current[:, :, 0:1] = model_latents[:, :, 0:1].detach()
        decoder_raw, decoder_components = safe.safe_decoder_space_loss(
            pipe,
            pred_current_latents=pred_current,
            target_current_latents=target_current,
            region_masks=inputs.get("tokenlight_rgb_decoder_region_masks"),
            region_weights=inputs.get("tokenlight_rgb_decoder_region_weights", {}),
            full_loss_weight=float(inputs.get("tokenlight_rgb_decoder_full_loss_weight", 1.0)),
            transform=str(inputs.get("tokenlight_rgb_decoder_transform", "rgb")),
            loss_type=str(inputs.get("tokenlight_decoder_loss_type", "l1")),
            luminance_eps=float(inputs.get("tokenlight_decoder_luminance_eps", 1e-3)),
            charbonnier_eps=float(inputs.get("tokenlight_decoder_charbonnier_eps", 1e-3)),
            min_region_fraction=float(inputs.get("tokenlight_decoder_min_region_fraction", 0.1)),
            max_samples=int(inputs.get("tokenlight_decoder_max_samples_per_batch", 1)),
            gradient_checkpointing=bool(inputs.get("tokenlight_decoder_gradient_checkpointing", True)),
            exclude_first_frame=exclude_first,
        )
        decoder_term = decoder_weight * decoder_ramp * decoder_raw
        total = total + decoder_term
        metrics["train/rgb/decoder_raw"] = decoder_raw.detach()
        metrics["train/rgb/decoder_term"] = decoder_term.detach()
        for name, value in decoder_components.items():
            metrics[f"train/rgb/decoder_{name}"] = value.detach()
    metrics["train/rgb/decoder_ramp"] = absolute_mse.new_tensor(float(decoder_ramp))
    metrics["train/rgb/decoder_effective_weight"] = absolute_mse.new_tensor(
        float(decoder_weight) * float(decoder_ramp)
    )
    metrics["train/rgb/total_loss"] = total.detach()
    pipe._tokenlight_loss_metrics = metrics
    return total


def _data_value_at(data: Mapping[str, Any], key: str, index: int, batch: int):
    value = data.get(key)
    if isinstance(value, list) and len(value) == batch:
        return value[index]
    if isinstance(value, tuple) and len(value) == batch:
        return value[index]
    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch:
        item = value[index]
        return item.item() if item.numel() == 1 else item
    return value


class TokenLightDeltaFlowTrainingModule(safe.TokenLightSafeDecoderTrainingModule):
    def __init__(
        self,
        *args,
        tokenlight_delta_pairs_per_batch: int = 2,
        tokenlight_delta_loss_weight: float = 0.05,
        tokenlight_delta_loss_type: str = "mse",
        tokenlight_delta_charbonnier_eps: float = 1e-3,
        tokenlight_delta_warmup_steps: int = 0,
        tokenlight_delta_ramp_steps: int = 1000,
        tokenlight_delta_sigma_gamma: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tokenlight_delta_pairs_per_batch = int(tokenlight_delta_pairs_per_batch)
        self.delta_loss_options = {
            "tokenlight_delta_loss_weight": float(tokenlight_delta_loss_weight),
            "tokenlight_delta_loss_type": str(tokenlight_delta_loss_type),
            "tokenlight_delta_charbonnier_eps": float(tokenlight_delta_charbonnier_eps),
            "tokenlight_delta_warmup_steps": int(tokenlight_delta_warmup_steps),
            "tokenlight_delta_ramp_steps": int(tokenlight_delta_ramp_steps),
            "tokenlight_delta_sigma_gamma": float(tokenlight_delta_sigma_gamma),
        }
        self.task_to_loss["sft"] = self._delta_sft_loss
        self.task_to_loss["sft:train"] = self._delta_sft_loss

    def _delta_sft_loss(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        del inputs_nega
        merged = dict(inputs_shared)
        merged.update(inputs_posi)
        merged.update(self.decoder_loss_options)
        merged.update(self.delta_loss_options)
        merged["tokenlight_optimizer_step"] = int(self.decoder_optimizer_step)
        return TokenLightDeltaFlowMatchLoss(pipe, **merged)

    def _attach_delta_inputs(self, inputs, data):
        inputs_shared, inputs_posi, inputs_nega = inputs
        batch = int(inputs_shared["input_latents"].shape[0])
        raw_pair_indices = data.get("_tokenlight_delta_pair_indices")
        if not isinstance(raw_pair_indices, torch.Tensor):
            raise TypeError("Structured delta batch is missing explicit pair indices")
        pair_indices = raw_pair_indices.to(
            device=inputs_shared["input_latents"].device, dtype=torch.long
        )
        expected = self.tokenlight_delta_pairs_per_batch
        if tuple(pair_indices.shape) != (expected, 2):
            raise ValueError(
                f"Expected pair indices [{expected},2], got {tuple(pair_indices.shape)}"
            )
        if int(torch.unique(pair_indices).numel()) != int(pair_indices.numel()):
            raise ValueError("Pair indices reuse a batch slot")

        # Verify the sampler contract before any expensive DiT computation.
        for left, right in pair_indices.detach().cpu().tolist():
            rows = []
            for index in (int(left), int(right)):
                row = {
                    key: _data_value_at(data, key, index, batch)
                    for key in (
                        "_vae_latent_cache_source",
                        "scene_id",
                        "scene_folder",
                        "input_image",
                        "prompt",
                        "task",
                        self.tokenlight_attrs_key,
                    )
                }
                rows.append(row)
            left_info = _pair_row_info(rows[0], attrs_key=self.tokenlight_attrs_key)
            right_info = _pair_row_info(rows[1], attrs_key=self.tokenlight_attrs_key)
            if left_info is None or right_info is None:
                raise ValueError("Malformed light metadata in a structured delta pair")
            if left_info.group_key != right_info.group_key:
                raise ValueError("Delta pair differs in scene/source/non-position lighting attributes")
            if left_info.position == right_info.position:
                raise ValueError("Delta pair uses the same light position twice")

        device = inputs_posi["context"].device
        if self.training and self.tokenlight_cfg_drop_prob > 0:
            pair_drop = torch.rand(expected, device=device) < self.tokenlight_cfg_drop_prob
            singleton_count = batch - 2 * expected
            singleton_drop = torch.rand(singleton_count, device=device) < self.tokenlight_cfg_drop_prob
            drop_light = torch.cat((pair_drop.repeat_interleave(2), singleton_drop), dim=0)
        else:
            pair_drop = torch.zeros(expected, dtype=torch.bool, device=device)
            drop_light = torch.zeros(batch, dtype=torch.bool, device=device)
        inputs_posi["tokenlight_drop_light"] = drop_light
        inputs_nega["tokenlight_drop_light"] = True
        inputs_shared["tokenlight_delta_pair_indices"] = pair_indices
        inputs_shared["tokenlight_delta_pair_active"] = ~pair_drop
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        # Preserve compatibility with launchers that pass cached rows through
        # the ``inputs`` argument instead of ``data``.
        if isinstance(inputs, dict) and "_tokenlight_delta_pair_indices" in inputs:
            data, inputs = inputs, None
        data = self._prepare_data_aliases(data)
        if inputs is None and self._has_cached_latents(data):
            prepared = self._cached_batched_inputs(data)
            prepared = self._attach_delta_inputs(prepared, data)
            prepared = self._attach_extra_mask_inputs(prepared, data)
            prepared = self._attach_decoder_inputs(prepared, data)
            return self.task_to_loss[self.task](self.pipe, *prepared)
        if inputs is None and self._is_collated_batch(data):
            prepared = self._batched_inputs(data)
            prepared = self._attach_delta_inputs(prepared, data)
            prepared = self._attach_extra_mask_inputs(prepared, data)
            prepared = self._attach_decoder_inputs(prepared, data)
            return self.task_to_loss[self.task](self.pipe, *prepared)
        raise ValueError("Delta-flow training requires a collated structured batch")


def delta_parser(train_mode: str) -> argparse.ArgumentParser:
    parser = safe.safe_parser(train_mode)
    parser.add_argument("--tokenlight_delta_pairs_per_batch", type=int, default=2)
    parser.add_argument("--tokenlight_delta_loss_weight", type=float, default=0.05)
    parser.add_argument(
        "--tokenlight_delta_loss_type", choices=DELTA_LOSS_TYPES, default="mse"
    )
    parser.add_argument("--tokenlight_delta_charbonnier_eps", type=float, default=1e-3)
    parser.add_argument("--tokenlight_delta_warmup_steps", type=int, default=0)
    parser.add_argument("--tokenlight_delta_ramp_steps", type=int, default=1000)
    parser.add_argument("--tokenlight_delta_sigma_gamma", type=float, default=0.0)
    parser.add_argument("--tokenlight_delta_pair_seed", type=int, default=480)
    return parser


def _validate_delta_args(
    args,
    raw_config: Mapping[str, Any],
    parser: argparse.ArgumentParser,
) -> None:
    safe._validate_safe_args(args, raw_config, parser)
    if args.task not in {"sft", "sft:train"}:
        parser.error("delta-flow trainer supports only sft and sft:train")
    if int(args.tokenlight_delta_pairs_per_batch) <= 0:
        parser.error("tokenlight_delta_pairs_per_batch must be positive")
    if 2 * int(args.tokenlight_delta_pairs_per_batch) > int(args.batch_size):
        parser.error("delta pairs do not fit in the configured batch size")
    if not math.isfinite(float(args.tokenlight_delta_loss_weight)) or args.tokenlight_delta_loss_weight < 0:
        parser.error("tokenlight_delta_loss_weight must be finite and non-negative")
    if args.tokenlight_delta_loss_type not in DELTA_LOSS_TYPES:
        parser.error(f"invalid delta loss type: {args.tokenlight_delta_loss_type}")
    if not math.isfinite(float(args.tokenlight_delta_charbonnier_eps)) or args.tokenlight_delta_charbonnier_eps <= 0:
        parser.error("tokenlight_delta_charbonnier_eps must be finite and positive")
    if int(args.tokenlight_delta_warmup_steps) < 0 or int(args.tokenlight_delta_ramp_steps) < 0:
        parser.error("delta warmup/ramp steps must be non-negative")
    if not math.isfinite(float(args.tokenlight_delta_sigma_gamma)) or args.tokenlight_delta_sigma_gamma < 0:
        parser.error("tokenlight_delta_sigma_gamma must be finite and non-negative")
    if float(args.tokenlight_rgb_latent_loss_weight) <= 0:
        parser.error("absolute latent flow loss must remain enabled for delta-flow training")
    if not bool(args.tokenlight_light_tokens) or not bool(args.tokenlight_source_tokens):
        parser.error("delta-flow training requires both light and source tokens")
    if float(args.tokenlight_light_dropout) != 0.0:
        parser.error("V1 requires tokenlight_light_dropout=0 so pair dropout remains shared")
    if base._parse_balanced_task_batch(getattr(args, "balanced_task_batch", None)) is not None:
        parser.error("balanced_task_batch cannot be combined with the structured delta sampler")


def parse_delta_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--train_mode", choices=("single", "zero3"), default="single")
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    config_path = pre_args.config or os.environ.get(
        "TOKENLIGHT_TRAIN_CONFIG",
        DEFAULT_DELTA_CONFIG_PATH,
    )
    raw_config = base._load_json_config(config_path)
    parser = delta_parser(pre_args.train_mode)
    base._apply_config_defaults(parser, raw_config)
    parser.set_defaults(config=config_path, train_mode=pre_args.train_mode)
    args = parser.parse_args()
    raw_config = base._load_json_config(args.config)
    _validate_delta_args(args, raw_config, parser)
    base._resolve_weight_paths(args)
    base._append_timestamp_to_output_path(args)
    return args, raw_config, raw_config


def launch_delta_training_task(accelerator, dataset, model, model_logger, args=None, **kwargs):
    del kwargs
    if not bool(getattr(dataset, "is_vae_latent_cache_dataset", False)):
        raise ValueError(
            "Delta-flow V1 requires the aligned VAE latent-cache dataset used by the baseline. "
            "Uncached image transforms are intentionally not supported in this isolated trainer."
        )
    batch_size = int(args.batch_size)
    pairs_per_batch = int(args.tokenlight_delta_pairs_per_batch)
    pair_sampler = MixedSameScenePairBatchSampler(
        dataset.data,
        dataset_length=len(dataset),
        batch_size=batch_size,
        pairs_per_batch=pairs_per_batch,
        attrs_key=args.tokenlight_attrs_key,
        seed=int(args.tokenlight_delta_pair_seed),
    )
    if accelerator.is_main_process:
        print(
            "Delta-flow objective: "
            f"pairs_per_batch={pairs_per_batch}, "
            f"singletons={batch_size - 2 * pairs_per_batch}, "
            f"lambda={args.tokenlight_delta_loss_weight}, "
            f"loss={args.tokenlight_delta_loss_type}, "
            f"warmup={args.tokenlight_delta_warmup_steps}, "
            f"ramp={args.tokenlight_delta_ramp_steps}, "
            f"sigma_gamma={args.tokenlight_delta_sigma_gamma}, "
            f"pairable_rows={pair_sampler.pairable_rows}/{len(dataset.data)}, "
            f"batches={len(pair_sampler)}"
        )

    decoder_enabled = float(args.tokenlight_rgb_decoder_loss_weight) > 0
    if decoder_enabled and not safe._is_deepspeed(accelerator) and batch_size > 1:
        print(
            "Warning: decoder loss on a single GPU is safest with batch_size=1; "
            "only decoder_max_samples_per_batch samples are decoded."
        )

    trainable_dtype = safe._select_trainable_dtype(args, accelerator)
    before = base._coerce_trainable_parameter_dtype(model, trainable_dtype)
    after = base._trainable_dtype_counts(model)
    if accelerator.is_main_process:
        print(f"Trainable dtype: target={trainable_dtype}, before={before}, after={after}")

    optimizer_class = base.get_optimizer_class(args.customized_optimizer)
    trainable_parameters = base._trainable_parameters(model)
    optimizer = optimizer_class(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader_kwargs = base._dataloader_runtime_kwargs(
        args,
        dataset,
        num_workers=args.dataset_num_workers,
        accelerator=accelerator,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=pair_sampler,
        collate_fn=partial(
            collate_delta_flow_batch,
            pairs_per_batch=pairs_per_batch,
        ),
        **dataloader_kwargs,
    )

    base._configure_deepspeed_batch_size(accelerator, args, batch_size)
    enable_model_cpu_offload = getattr(args, "enable_model_cpu_offload", False)
    if enable_model_cpu_offload:
        optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
        model.pipe.device = accelerator.device
        offload_manager = base.OffloadTrainingManager(
            model,
            accelerator.device,
            getattr(args, "enable_optimizer_cpu_offload", False),
            getattr(args, "cpu_offload_split_threshold", None),
        )
    else:
        model.to(device=accelerator.device)
        model, optimizer, dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )
        offload_manager = None

    base.save_training_runtime_snapshot(args, accelerator, model)
    tb_metrics = base.TokenLightTensorBoardMetrics(
        args.output_path,
        enabled=args.enable_tensorboard_log,
    )
    base.initialize_deepspeed_gradient_checkpointing(accelerator)
    optimizer_step = int(getattr(model_logger, "num_steps", 0))
    try:
        for epoch_id in range(int(args.start_epoch), int(args.start_epoch) + int(args.num_epochs)):
            # Accelerate's DataLoaderShard owns its own iteration counter and
            # calls set_epoch() again inside __iter__.  Set both layers so a
            # resumed start_epoch cannot silently reset the pair schedule to 0.
            if hasattr(dataloader, "set_epoch"):
                dataloader.set_epoch(epoch_id)
            pair_sampler.set_epoch(epoch_id)
            iterator = base.tqdm(dataloader, disable=not accelerator.is_local_main_process)
            for data in iterator:
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.decoder_optimizer_step = optimizer_step
                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        loss = model({}, inputs=data) if dataset.load_from_cache else model(data)
                    accelerator.backward(loss)
                    if offload_manager is not None:
                        offload_manager.after_backward()
                    if (
                        accelerator.sync_gradients
                        and not safe._is_deepspeed(accelerator)
                        and args.max_grad_norm > 0
                    ):
                        accelerator.clip_grad_norm_(trainable_parameters, float(args.max_grad_norm))
                    optimizer.step()
                    scheduler.step()
                    if accelerator.sync_gradients:
                        metrics = base._collect_train_metrics(accelerator, model, optimizer, loss)
                        optimizer_step += 1
                        base._call_compatible_method(
                            model_logger,
                            "on_step_end",
                            accelerator,
                            model,
                            args.save_steps,
                            loss=loss,
                            epoch=epoch_id,
                            optimizer=optimizer,
                            scheduler=scheduler,
                        )
                        tb_metrics.log(accelerator, optimizer_step, metrics)
                        if metrics and "train/loss" in metrics:
                            iterator.set_postfix(
                                loss=f"{metrics['train/loss']:.4f}", step=optimizer_step
                            )
                    optimizer.zero_grad(set_to_none=True)
            if args.save_steps is None:
                base._call_compatible_method(
                    model_logger,
                    "on_epoch_end",
                    accelerator,
                    model,
                    epoch_id,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
        base._call_compatible_method(
            model_logger,
            "on_training_end",
            accelerator,
            model,
            args.save_steps,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    finally:
        tb_metrics.close(accelerator)


def _build_delta_model(args, accelerator):
    return TokenLightDeltaFlowTrainingModule(
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
            else accelerator.device
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
        tokenlight_rgb_latent_loss_weight=args.tokenlight_rgb_latent_loss_weight,
        tokenlight_rgb_decoder_loss_weight=args.tokenlight_rgb_decoder_loss_weight,
        tokenlight_rgb_decoder_transform=args.tokenlight_rgb_decoder_transform,
        tokenlight_decoder_loss_type=args.tokenlight_decoder_loss_type,
        tokenlight_decoder_luminance_eps=args.tokenlight_decoder_luminance_eps,
        tokenlight_decoder_charbonnier_eps=args.tokenlight_decoder_charbonnier_eps,
        tokenlight_rgb_decoder_full_loss_weight=args.tokenlight_rgb_decoder_full_loss_weight,
        tokenlight_rgb_decoder_shadow_loss_weight=args.tokenlight_rgb_decoder_shadow_loss_weight,
        tokenlight_rgb_decoder_direct_loss_weight=args.tokenlight_rgb_decoder_direct_loss_weight,
        tokenlight_rgb_decoder_shadow_mask_key=args.tokenlight_rgb_decoder_shadow_mask_key,
        tokenlight_rgb_decoder_direct_mask_key=args.tokenlight_rgb_decoder_direct_mask_key,
        tokenlight_decoder_min_region_fraction=args.tokenlight_decoder_min_region_fraction,
        tokenlight_decoder_max_samples_per_batch=args.tokenlight_decoder_max_samples_per_batch,
        tokenlight_decoder_gradient_checkpointing=args.tokenlight_decoder_gradient_checkpointing,
        tokenlight_decoder_warmup_steps=args.tokenlight_decoder_warmup_steps,
        tokenlight_decoder_ramp_steps=args.tokenlight_decoder_ramp_steps,
        tokenlight_mask_image_key=args.tokenlight_mask_image_key,
        tokenlight_extra_mask_image_keys=args.tokenlight_extra_mask_image_keys,
        tokenlight_extra_mask_latent_cache_dirs=args.tokenlight_extra_mask_latent_cache_dirs,
        tokenlight_delta_pairs_per_batch=args.tokenlight_delta_pairs_per_batch,
        tokenlight_delta_loss_weight=args.tokenlight_delta_loss_weight,
        tokenlight_delta_loss_type=args.tokenlight_delta_loss_type,
        tokenlight_delta_charbonnier_eps=args.tokenlight_delta_charbonnier_eps,
        tokenlight_delta_warmup_steps=args.tokenlight_delta_warmup_steps,
        tokenlight_delta_ramp_steps=args.tokenlight_delta_ramp_steps,
        tokenlight_delta_sigma_gamma=args.tokenlight_delta_sigma_gamma,
    )


def main() -> None:
    args, raw_config, merged_config = parse_delta_args()
    args.loss_impl_version = LOSS_IMPL_VERSION
    args.delta_flow_formula = "MSE((v_pred_j-v_pred_i)-(v_target_j-v_target_i))"
    args.delta_pair_layout = "pairs_first_then_singletons"
    args.delta_noise_sharing = "within_pair"

    accelerator_kwargs = {
        "kwargs_handlers": [
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            )
        ],
        "dataloader_config": accelerate.DataLoaderConfiguration(
            split_batches=False,
            even_batches=True,
        ),
    }
    if args.train_mode == "single":
        accelerator_kwargs["gradient_accumulation_steps"] = args.gradient_accumulation_steps
        accelerator_kwargs["mixed_precision"] = args.mixed_precision
    accelerator = accelerate.Accelerator(**accelerator_kwargs)
    base.save_training_config_snapshot(args, raw_config, merged_config, accelerator)

    dataset = safe.build_safe_dataset(args)
    if not bool(getattr(dataset, "is_vae_latent_cache_dataset", False)):
        raise ValueError(
            "Delta-flow V1 requires the aligned VAE latent-cache dataset used by the baseline."
        )
    model = _build_delta_model(args, accelerator)
    model_logger = base._make_model_logger(args)
    launch_delta_training_task(
        accelerator,
        dataset,
        model,
        model_logger,
        args=args,
    )


if __name__ == "__main__":
    main()
