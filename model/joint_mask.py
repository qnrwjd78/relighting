from __future__ import annotations

"""Clean-condition shadow-mask prediction for TokenLight Wan.

The predictor consumes only the clean source VAE latent and exact numeric light
attributes.  It produces both a full-resolution shadow-mask logit and a compact
latent condition that is prepended to Wan through the existing mask-token path.
The rendered GT shadow mask is never an input to the generator.  The compact
condition is added to the existing target-token grid, so it does not increase
Wan's self-attention sequence length.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from model.light_encoder import LIGHTOKEN_NAMES, compact_attrs, parse_attrs_json
from model.wan import (
    TokenLightTypeEmbedding,
    _repeat_to_batch,
    model_fn_wan_video_tokenlight,
)


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2):
        if int(channels) % groups == 0:
            return groups
    return 1


def numeric_light_features(
    attrs: torch.Tensor | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    *,
    batch: int,
    device: torch.device,
) -> torch.Tensor:
    """Return exact values plus validity bits in ``LIGHTOKEN_NAMES`` order."""

    if isinstance(attrs, torch.Tensor):
        values = attrs.to(device=device, dtype=torch.float32)
        values = values.unsqueeze(0) if values.ndim == 1 else values
        if values.shape[0] == 1 and batch > 1:
            values = values.expand(batch, -1)
        if tuple(values.shape) != (batch, len(LIGHTOKEN_NAMES)):
            raise ValueError(
                f"Numeric joint-mask attrs must be {(batch, len(LIGHTOKEN_NAMES))}, "
                f"got {tuple(values.shape)}"
            )
        valid = torch.isfinite(values)
        values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.cat([values, valid.float()], dim=1)

    if attrs is None:
        rows: list[dict[str, Any]] = [{}]
    elif isinstance(attrs, Mapping) or isinstance(attrs, (str, bytes)):
        rows = [parse_attrs_json(attrs)]
    else:
        rows = [parse_attrs_json(item) for item in attrs]
    if len(rows) == 1 and batch > 1:
        rows = rows * batch
    if len(rows) != batch:
        raise ValueError(f"Got {len(rows)} light records for joint-mask batch {batch}")

    values = torch.zeros(batch, len(LIGHTOKEN_NAMES), device=device, dtype=torch.float32)
    valid = torch.zeros_like(values, dtype=torch.bool)
    for row_index, row in enumerate(rows):
        compact = compact_attrs(row)
        for column, name in enumerate(LIGHTOKEN_NAMES):
            value = compact.get(name)
            if value is None:
                continue
            number = float(value)
            if not torch.isfinite(torch.tensor(number)):
                continue
            values[row_index, column] = number
            valid[row_index, column] = True
    return torch.cat([values, valid.float()], dim=1)


class FiLMResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int) -> None:
        super().__init__()
        self.in_projection = (
            nn.Identity()
            if int(in_channels) == int(out_channels)
            else nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1)
        )
        self.norm1 = nn.GroupNorm(_group_count(out_channels), int(out_channels))
        self.conv1 = nn.Conv2d(int(out_channels), int(out_channels), kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), int(out_channels))
        self.conv2 = nn.Conv2d(int(out_channels), int(out_channels), kernel_size=3, padding=1)
        self.condition = nn.Linear(int(condition_dim), 2 * int(out_channels))
        self.activation = nn.SiLU()

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        residual = self.in_projection(value)
        hidden = self.norm1(residual)
        scale, shift = self.condition(condition).chunk(2, dim=1)
        hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv1(self.activation(hidden))
        hidden = self.conv2(self.activation(self.norm2(hidden)))
        return residual + hidden


class LightConditionedMaskPredictor(nn.Module):
    """Predict a 480px mask and a Wan-compatible clean mask condition."""

    def __init__(
        self,
        *,
        source_channels: int = 48,
        base_channels: int = 96,
        condition_dim: int = 128,
        output_height: int = 480,
        output_width: int = 480,
    ) -> None:
        super().__init__()
        self.source_channels = int(source_channels)
        self.output_height = int(output_height)
        self.output_width = int(output_width)
        if self.source_channels <= 0 or int(base_channels) < 16 or int(condition_dim) < 16:
            raise ValueError("Malformed joint-mask predictor dimensions")

        light_features = 2 * len(LIGHTOKEN_NAMES)
        self.light_mlp = nn.Sequential(
            nn.Linear(light_features, int(condition_dim)),
            nn.SiLU(),
            nn.Linear(int(condition_dim), int(condition_dim)),
        )
        self.source_stem = nn.Conv2d(self.source_channels, int(base_channels), kernel_size=3, padding=1)
        self.bottleneck = nn.ModuleList(
            [
                FiLMResidualBlock(base_channels, base_channels, condition_dim),
                FiLMResidualBlock(base_channels, base_channels, condition_dim),
            ]
        )

        # Zero initialization makes the new Wan condition a no-op at step 0,
        # preserving the pretrained baseline while the predictor starts learning.
        self.condition_head = nn.Conv2d(int(base_channels), self.source_channels, kernel_size=1)
        nn.init.zeros_(self.condition_head.weight)
        nn.init.zeros_(self.condition_head.bias)

        stage_channels = (int(base_channels), 64, 32, 16)
        blocks = []
        current = int(base_channels)
        for channels in stage_channels:
            blocks.append(FiLMResidualBlock(current, channels, condition_dim))
            current = channels
        self.upsample_blocks = nn.ModuleList(blocks)
        self.output_norm = nn.GroupNorm(_group_count(current), current)
        self.output_head = nn.Conv2d(current, 1, kernel_size=3, padding=1)
        nn.init.normal_(self.output_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.output_head.bias, -2.0)
        self.activation = nn.SiLU()

    def forward(
        self,
        source_latents: torch.Tensor,
        attrs: torch.Tensor | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
        *,
        output_size: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if source_latents.ndim != 5 or int(source_latents.shape[2]) != 1:
            raise ValueError(
                "Joint-mask predictor requires one-frame source latents [B,C,1,H,W], "
                f"got {tuple(source_latents.shape)}"
            )
        if int(source_latents.shape[1]) != self.source_channels:
            raise ValueError(
                f"Joint-mask source channels {source_latents.shape[1]} != {self.source_channels}"
            )
        batch = int(source_latents.shape[0])
        exact_light = numeric_light_features(attrs, batch=batch, device=source_latents.device)
        condition = self.light_mlp(exact_light.to(dtype=self.light_mlp[0].weight.dtype))
        hidden = self.source_stem(source_latents[:, :, 0].to(dtype=self.source_stem.weight.dtype))
        for block in self.bottleneck:
            hidden = block(hidden, condition)

        mask_condition = self.condition_head(hidden).unsqueeze(2)
        decoded = hidden
        for block in self.upsample_blocks:
            decoded = F.interpolate(decoded, scale_factor=2.0, mode="bilinear", align_corners=False)
            decoded = block(decoded, condition)
        logits = self.output_head(self.activation(self.output_norm(decoded)))
        target_size = output_size or (self.output_height, self.output_width)
        if tuple(logits.shape[-2:]) != tuple(target_size):
            logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
        return logits.unsqueeze(2), mask_condition


def _drop_mask(
    value: bool | torch.Tensor | Sequence[bool],
    *,
    batch: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        result = value.to(device=device, dtype=torch.bool).reshape(-1)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = torch.as_tensor(list(value), device=device, dtype=torch.bool).reshape(-1)
    else:
        result = torch.full((batch,), bool(value), device=device, dtype=torch.bool)
    if result.numel() == 1 and batch > 1:
        result = result.expand(batch)
    if result.numel() != batch:
        raise ValueError(f"Got {result.numel()} drop flags for joint-mask batch {batch}")
    return result


def model_fn_wan_video_tokenlight_joint_mask(
    *,
    joint_mask_predictor: LightConditionedMaskPredictor,
    tokenlight_light_encoder,
    tokenlight_type_embedding: TokenLightTypeEmbedding | None,
    tokenlight_predicted_mask_condition_scale: float = 1.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a clean mask prediction once per training call and condition Wan."""

    call = dict(kwargs)
    source = call.get("tokenlight_source_latents")
    if not isinstance(source, torch.Tensor):
        raise ValueError("Joint RGB/mask model requires tokenlight_source_latents")
    latents = call.get("latents")
    if not isinstance(latents, torch.Tensor):
        raise ValueError("Joint RGB/mask model requires target latents")
    batch = int(latents.shape[0])
    source = _repeat_to_batch(source, batch)
    attrs = call.get("tokenlight_attrs")
    height = int(call.get("height", joint_mask_predictor.output_height))
    width = int(call.get("width", joint_mask_predictor.output_width))
    mask_logits, predicted_condition = joint_mask_predictor(
        source,
        attrs,
        output_size=(height, width),
    )

    dropped = _drop_mask(
        call.get("tokenlight_drop_light", False),
        batch=batch,
        device=predicted_condition.device,
    )
    keep = (~dropped).to(dtype=predicted_condition.dtype).view(batch, 1, 1, 1, 1)
    predicted_condition = (
        predicted_condition * keep * float(tokenlight_predicted_mask_condition_scale)
    )
    # Never honor an externally supplied/GT mask latent on this path.  Add the
    # learned clean condition on the existing target grid instead of prepending
    # another 225-token prefix at 480px.
    call.pop("tokenlight_mask_latents", None)
    velocity = model_fn_wan_video_tokenlight(
        tokenlight_light_encoder=tokenlight_light_encoder,
        tokenlight_type_embedding=tokenlight_type_embedding,
        tokenlight_additive_condition_latents=predicted_condition,
        **call,
    )
    return velocity, mask_logits


__all__ = [
    "LightConditionedMaskPredictor",
    "model_fn_wan_video_tokenlight_joint_mask",
    "numeric_light_features",
]
