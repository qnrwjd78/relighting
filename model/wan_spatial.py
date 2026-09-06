from __future__ import annotations

"""Spatial-prefix extensions for TokenLight's Wan model function.

This module is intentionally isolated from :mod:`model.wan`.  The
original source/mask/light/target prefix behavior is preserved, while a small
CNN converts camera-aligned physical maps into one token per Wan spatial patch.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from model.light_encoder import LightokenEncoder
from model.wan import TokenLightTypeEmbedding
from model import wan as legacy


SPATIAL_PREFIX_TYPE_CONDITION = 0
SPATIAL_PREFIX_TYPE_LAYOUT = 1
SPATIAL_PREFIX_TYPE_NAMES = ("spatial_condition", "layout")


class SpatialPrefixTypeEmbedding(nn.Module):
    """Type embeddings dedicated to spatial maps and future layout tokens."""

    def __init__(self, token_dim: int, num_types: int = len(SPATIAL_PREFIX_TYPE_NAMES)) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(num_types), int(token_dim))
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, type_id: int) -> torch.Tensor:
        embedding = self.embedding.weight[int(type_id)].to(
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return tokens + embedding.view(1, 1, -1)


def _parse_channel_values(
    values: str | Sequence[float],
    *,
    channels: int,
    name: str,
) -> torch.Tensor:
    if isinstance(values, str):
        parsed = [float(value.strip()) for value in values.split(",") if value.strip()]
    else:
        parsed = [float(value) for value in values]
    if len(parsed) != int(channels):
        raise ValueError(f"{name} needs {channels} values, got {len(parsed)}: {parsed}")
    tensor = torch.tensor(parsed, dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    return tensor


class SpatialConditionEncoder(nn.Module):
    """Encode a dense physical map into spatially registered Wan prefix tokens.

    Missing/dropped conditions use learned null tokens.  A valid all-zero map
    remains distinguishable because present and missing maps receive different
    learned presence embeddings.
    """

    def __init__(
        self,
        *,
        input_channels: int,
        token_dim: int,
        hidden_channels: int = 64,
        channel_mean: str | Sequence[float] | None = None,
        channel_std: str | Sequence[float] | None = None,
        input_clip: float = 0.0,
    ) -> None:
        super().__init__()
        input_channels = int(input_channels)
        token_dim = int(token_dim)
        hidden_channels = int(hidden_channels)
        if input_channels < 1 or token_dim < 1 or hidden_channels < 8:
            raise ValueError("input_channels/token_dim must be positive and hidden_channels >= 8")
        mean = _parse_channel_values(
            [0.0] * input_channels if channel_mean is None else channel_mean,
            channels=input_channels,
            name="tokenlight_spatial_channel_mean",
        )
        std = _parse_channel_values(
            [1.0] * input_channels if channel_std is None else channel_std,
            channels=input_channels,
            name="tokenlight_spatial_channel_std",
        )
        if (std <= 0).any():
            raise ValueError("tokenlight_spatial_channel_std values must be positive")
        if not torch.isfinite(torch.tensor(float(input_clip))) or float(input_clip) < 0:
            raise ValueError("tokenlight_spatial_input_clip must be finite and non-negative")

        self.input_channels = input_channels
        self.token_dim = token_dim
        self.input_clip = float(input_clip)
        self.register_buffer("channel_mean", mean.view(1, input_channels, 1, 1), persistent=True)
        self.register_buffer("channel_std", std.view(1, input_channels, 1, 1), persistent=True)

        middle = hidden_channels * 2
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=7, stride=4, padding=3),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, middle, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(middle), middle),
            nn.SiLU(),
            nn.Conv2d(middle, middle, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(middle), middle),
            nn.SiLU(),
        )
        self.projection = nn.Conv2d(middle, token_dim, kernel_size=1)
        self.presence_embedding = nn.Embedding(2, token_dim)
        self.null_token = nn.Parameter(torch.empty(1, 1, token_dim))
        # Keep the new prefix small at initialization so adding the condition
        # does not abruptly dominate pretrained Wan/source/light activations.
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.projection.bias)
        nn.init.normal_(self.presence_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.null_token, mean=0.0, std=0.02)

    def forward(
        self,
        spatial_map: torch.Tensor,
        *,
        output_size: tuple[int, int],
        present: bool | torch.Tensor | Sequence[bool] = True,
        drop_condition: bool | torch.Tensor | Sequence[bool] = False,
    ) -> torch.Tensor:
        if spatial_map.ndim == 3:
            spatial_map = spatial_map.unsqueeze(0)
        if spatial_map.ndim != 4:
            raise ValueError(
                "tokenlight_spatial_map must have shape [B,C,H,W], "
                f"got {tuple(spatial_map.shape)}"
            )
        if spatial_map.shape[1] != self.input_channels:
            raise ValueError(
                f"spatial channel mismatch: {spatial_map.shape[1]} != {self.input_channels}"
            )
        out_h, out_w = (int(output_size[0]), int(output_size[1]))
        if out_h < 1 or out_w < 1:
            raise ValueError(f"invalid spatial token grid: {(out_h, out_w)}")
        if not torch.isfinite(spatial_map).all():
            raise ValueError("tokenlight_spatial_map contains NaN or Inf")

        batch = int(spatial_map.shape[0])
        present_tensor = _batch_bool(present, batch=batch, device=spatial_map.device, name="present")
        drop_tensor = _batch_bool(
            drop_condition,
            batch=batch,
            device=spatial_map.device,
            name="drop_condition",
        )
        effective_present = present_tensor & ~drop_tensor

        try:
            parameter = next(self.stem.parameters())
        except StopIteration:
            parameter = None
        compute_dtype = parameter.dtype if parameter is not None else spatial_map.dtype
        value = spatial_map.to(dtype=compute_dtype)
        value = (
            value - self.channel_mean.to(device=value.device, dtype=compute_dtype)
        ) / self.channel_std.to(device=value.device, dtype=compute_dtype)
        if self.input_clip > 0:
            value = value.clamp(min=-self.input_clip, max=self.input_clip)
        # Single GPU keeps FP32 master weights and autocast selects BF16
        # kernels.  ZeRO-3 may partition BF16 parameters, hence compute_dtype.
        features = self.stem(value)
        features = torch.nn.functional.adaptive_avg_pool2d(features, (out_h, out_w))
        tokens = self.projection(features).flatten(2).transpose(1, 2).contiguous()

        null = self.null_token.to(device=tokens.device, dtype=tokens.dtype).expand(
            batch,
            out_h * out_w,
            -1,
        )
        tokens = torch.where(effective_present.view(batch, 1, 1), tokens, null)
        presence = self.presence_embedding(effective_present.long()).to(dtype=tokens.dtype)
        return tokens + presence.unsqueeze(1)


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _batch_bool(
    value: bool | torch.Tensor | Sequence[bool],
    *,
    batch: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=torch.bool).reshape(-1)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        tensor = torch.tensor(list(value), device=device, dtype=torch.bool).reshape(-1)
    else:
        tensor = torch.full((batch,), bool(value), device=device, dtype=torch.bool)
    if tensor.numel() == 1 and batch != 1:
        tensor = tensor.expand(batch)
    if tensor.numel() != batch:
        raise ValueError(f"{name} needs {batch} values, got {tensor.numel()}")
    return tensor


def _add_spatial_type(
    tokens: torch.Tensor,
    embedding: SpatialPrefixTypeEmbedding | None,
    type_id: int,
) -> torch.Tensor:
    return tokens if embedding is None else embedding(tokens, type_id)


def _prepare_layout_tokens(
    tokens: torch.Tensor,
    *,
    batch: int,
    token_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if tokens.ndim == 2:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 3:
        raise ValueError(f"tokenlight_layout_tokens must be [B,L,D], got {tuple(tokens.shape)}")
    tokens = legacy._repeat_to_batch(tokens, batch)
    if int(tokens.shape[-1]) != int(token_dim):
        raise ValueError(f"layout token dim mismatch: {tokens.shape[-1]} != {token_dim}")
    return tokens.to(device=device, dtype=dtype)


def _promote_clean_prefix_t_mod(
    dit: nn.Module,
    t_mod: torch.Tensor,
    *,
    prefix_len: int,
    target_len: int,
    batch: int,
) -> torch.Tensor:
    """Build per-token modulation: clean prefix, sampled-timestep target.

    One-frame Wan normally produces ``[B,6,D]`` modulation.  Leaving that
    tensor untouched would incorrectly label clean physical/source/light
    prefixes with the sampled noisy target timestep.  DiTBlock explicitly
    supports ``[B,L,6,D]``, so promote the target modulation and prepend t=0.
    """

    if t_mod.ndim == 3:
        target_t_mod = t_mod.unsqueeze(1)
    elif t_mod.ndim == 4:
        target_t_mod = t_mod
    else:
        raise ValueError(f"unexpected Wan t_mod shape: {tuple(t_mod.shape)}")

    if target_t_mod.shape[0] == 1 and batch != 1:
        target_t_mod = target_t_mod.expand(batch, *target_t_mod.shape[1:])
    elif int(target_t_mod.shape[0]) != int(batch):
        raise ValueError(f"t_mod batch mismatch: {target_t_mod.shape[0]} != {batch}")

    if target_t_mod.shape[1] == 1 and target_len != 1:
        target_t_mod = target_t_mod.expand(batch, target_len, *target_t_mod.shape[2:])
    elif int(target_t_mod.shape[1]) != int(target_len):
        raise ValueError(
            f"target t_mod length mismatch: {target_t_mod.shape[1]} != {target_len}"
        )
    clean_t_mod = legacy._clean_prefix_t_mod(
        dit,
        int(prefix_len),
        int(batch),
        target_t_mod.dtype,
        target_t_mod.device,
    )
    return torch.cat([clean_t_mod, target_t_mod], dim=1).contiguous()


def model_fn_wan_video_tokenlight_spatial(
    *,
    dit: nn.Module,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    clip_feature: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
    control_camera_latents_input=None,
    fuse_vae_embedding_in_latents: bool = False,
    motion_controller: nn.Module | None = None,
    motion_bucket_id: torch.Tensor | None = None,
    tokenlight_light_encoder: LightokenEncoder | None = None,
    tokenlight_type_embedding: TokenLightTypeEmbedding | None = None,
    tokenlight_spatial_encoder: SpatialConditionEncoder | None = None,
    tokenlight_spatial_type_embedding: SpatialPrefixTypeEmbedding | None = None,
    tokenlight_attrs: Mapping[str, Any] | Sequence[Mapping[str, Any]] | torch.Tensor | None = None,
    tokenlight_drop_light: bool | torch.Tensor | Sequence[bool] = False,
    tokenlight_source_latents: torch.Tensor | None = None,
    tokenlight_mask_latents: torch.Tensor | None = None,
    tokenlight_spatial_map: torch.Tensor | None = None,
    tokenlight_spatial_present: bool | torch.Tensor | Sequence[bool] = True,
    tokenlight_drop_spatial: bool | torch.Tensor | Sequence[bool] = False,
    tokenlight_layout_tokens: torch.Tensor | None = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Wan model function with dense-map and future CoShadow layout prefixes."""

    del kwargs
    from einops import rearrange
    from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d

    if getattr(dit, "seperated_timestep", False) and fuse_vae_embedding_in_latents:
        timestep = torch.concat(
            [
                torch.zeros(
                    (1, latents.shape[3] * latents.shape[4] // 4),
                    dtype=latents.dtype,
                    device=latents.device,
                ),
                torch.ones(
                    (latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4),
                    dtype=latents.dtype,
                    device=latents.device,
                )
                * timestep,
            ]
        ).flatten()
        t_head = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        t_mod = dit.time_projection(t_head).unflatten(2, (6, dit.dim))
    else:
        t_head = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t_head).unflatten(1, (6, dit.dim))

    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))

    context = dit.text_embedding(context)
    batch = int(context.shape[0])
    x = latents if latents.shape[0] == batch else torch.cat([latents] * batch, dim=0)
    if y is not None and getattr(dit, "require_vae_embedding", True):
        x = torch.cat([x, legacy._repeat_to_batch(y, batch)], dim=1)
    if clip_feature is not None and getattr(dit, "require_clip_embedding", True):
        context = torch.cat([dit.img_emb(legacy._repeat_to_batch(clip_feature, batch)), context], dim=1)

    patches = dit.patchify(x, control_camera_latents_input)
    target_grid = patches.shape[2:]
    target_tokens = rearrange(patches, "b c f h w -> b (f h w) c").contiguous()
    target_tokens = legacy._add_type_embedding(
        target_tokens,
        tokenlight_type_embedding,
        legacy.TOKENLIGHT_TYPE_TARGET,
    )
    target_freqs = legacy._freqs_for_grid(dit, target_grid, target_tokens.device)

    prefix_tokens: list[torch.Tensor] = []
    prefix_freqs: list[torch.Tensor] = []
    if tokenlight_source_latents is not None:
        source_tokens, source_grid = legacy._patch_to_tokens(dit, tokenlight_source_latents, batch)
        source_tokens = legacy._add_type_embedding(
            source_tokens,
            tokenlight_type_embedding,
            legacy.TOKENLIGHT_TYPE_SOURCE,
        )
        prefix_tokens.append(source_tokens)
        prefix_freqs.append(legacy._freqs_for_grid(dit, source_grid, target_tokens.device))

    if tokenlight_mask_latents is not None:
        mask_tokens, mask_grid = legacy._patch_to_tokens(dit, tokenlight_mask_latents, batch)
        mask_tokens = legacy._add_type_embedding(
            mask_tokens,
            tokenlight_type_embedding,
            legacy.TOKENLIGHT_TYPE_MASK,
        )
        prefix_tokens.append(mask_tokens)
        prefix_freqs.append(legacy._freqs_for_grid(dit, mask_grid, target_tokens.device))

    if tokenlight_spatial_map is not None:
        if tokenlight_spatial_encoder is None:
            raise ValueError("tokenlight_spatial_map was provided without tokenlight_spatial_encoder")
        spatial_tokens = tokenlight_spatial_encoder(
            tokenlight_spatial_map,
            output_size=(int(target_grid[1]), int(target_grid[2])),
            present=tokenlight_spatial_present,
            drop_condition=tokenlight_drop_spatial,
        ).to(device=target_tokens.device, dtype=target_tokens.dtype)
        spatial_tokens = _add_spatial_type(
            spatial_tokens,
            tokenlight_spatial_type_embedding,
            SPATIAL_PREFIX_TYPE_CONDITION,
        )
        spatial_grid = (1, int(target_grid[1]), int(target_grid[2]))
        prefix_tokens.append(spatial_tokens)
        prefix_freqs.append(legacy._freqs_for_grid(dit, spatial_grid, target_tokens.device))

    if tokenlight_layout_tokens is not None:
        layout_tokens = _prepare_layout_tokens(
            tokenlight_layout_tokens,
            batch=batch,
            token_dim=int(target_tokens.shape[-1]),
            device=target_tokens.device,
            dtype=target_tokens.dtype,
        )
        layout_tokens = _add_spatial_type(
            layout_tokens,
            tokenlight_spatial_type_embedding,
            SPATIAL_PREFIX_TYPE_LAYOUT,
        )
        prefix_tokens.append(layout_tokens)
        prefix_freqs.append(
            torch.ones(
                layout_tokens.shape[1],
                1,
                target_freqs.shape[-1],
                device=target_tokens.device,
                dtype=target_freqs.dtype,
            )
        )

    if tokenlight_light_encoder is not None:
        light_tokens = tokenlight_light_encoder(
            tokenlight_attrs,
            batch_size=batch,
            device=target_tokens.device,
            dtype=target_tokens.dtype,
            drop_light=tokenlight_drop_light,
        )
        light_tokens = legacy._add_type_embedding(
            light_tokens,
            tokenlight_type_embedding,
            legacy.TOKENLIGHT_TYPE_LIGHT,
        )
        prefix_tokens.append(light_tokens)
        prefix_freqs.append(
            torch.ones(
                light_tokens.shape[1],
                1,
                target_freqs.shape[-1],
                device=target_tokens.device,
                dtype=target_freqs.dtype,
            )
        )

    if prefix_tokens:
        prefix_len = sum(tokens.shape[1] for tokens in prefix_tokens)
        x = torch.cat([*prefix_tokens, target_tokens], dim=1)
        freqs = torch.cat([*prefix_freqs, target_freqs], dim=0)
        t_mod = _promote_clean_prefix_t_mod(
            dit,
            t_mod,
            prefix_len=prefix_len,
            target_len=int(target_tokens.shape[1]),
            batch=batch,
        )
    else:
        prefix_len = 0
        x = target_tokens
        freqs = target_freqs

    for block in dit.blocks:
        x = legacy.gradient_checkpoint_forward_compatible(
            block,
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            x,
            context,
            t_mod,
            freqs,
        )

    x = x[:, prefix_len:] if prefix_len else x
    x = dit.head(x, t_head)
    return dit.unpatchify(x, target_grid)


__all__ = [
    "SPATIAL_PREFIX_TYPE_CONDITION",
    "SPATIAL_PREFIX_TYPE_LAYOUT",
    "SPATIAL_PREFIX_TYPE_NAMES",
    "SpatialConditionEncoder",
    "SpatialPrefixTypeEmbedding",
    "_promote_clean_prefix_t_mod",
    "model_fn_wan_video_tokenlight_spatial",
]
