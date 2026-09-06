from __future__ import annotations

"""Isolated PBR model function with correct per-stream timestep semantics.

The legacy :mod:`model.tokenlight_wan_pbr` module remains untouched.  This
entrypoint reuses its stable token schema/helpers, but promotes one-frame Wan
timestep modulation to per-token modulation so clean source/light/PBR
conditions use t=0 while RGB and PBR targets use the sampled flow timestep.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from model.lightoken_encoder import LightokenEncoder
from model import tokenlight_wan_pbr as legacy


TokenLightPBRTypeEmbedding = legacy.TokenLightPBRTypeEmbedding
tokenlight_pbr_type_count = legacy.tokenlight_pbr_type_count


def _t_mod_for_token_length(t_mod: torch.Tensor, token_length: int) -> torch.Tensor:
    """Expand a one-frame ``[B,6,D]`` modulation over a token segment."""

    if t_mod.ndim == 3:
        return t_mod.unsqueeze(1).expand(-1, int(token_length), -1, -1)
    if t_mod.ndim != 4:
        raise ValueError(
            f"Expected timestep modulation rank 3 or 4, got {tuple(t_mod.shape)}"
        )
    if int(t_mod.shape[1]) == int(token_length):
        return t_mod
    if int(t_mod.shape[1]) == 1:
        return t_mod.expand(-1, int(token_length), -1, -1)
    raise ValueError(
        f"Timestep modulation length {t_mod.shape[1]} does not match token length "
        f"{token_length}"
    )


def model_fn_wan_video_tokenlight_pbr_safe(
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
    tokenlight_type_embedding: TokenLightPBRTypeEmbedding | None = None,
    tokenlight_attrs: Mapping[str, Any]
    | Sequence[Mapping[str, Any]]
    | torch.Tensor
    | None = None,
    tokenlight_drop_light: bool | torch.Tensor | Sequence[bool] = False,
    tokenlight_source_latents: torch.Tensor | None = None,
    tokenlight_pbr_latents: torch.Tensor | None = None,
    tokenlight_pbr_stream_latents: Mapping[str, torch.Tensor]
    | Sequence[tuple[str, torch.Tensor]]
    | None = None,
    tokenlight_pbr_is_target: bool | torch.Tensor | Sequence[bool] = True,
    tokenlight_pbr_stream_is_target: Mapping[
        str, bool | torch.Tensor | Sequence[bool]
    ]
    | None = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor] | None]:
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
        t_head = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0)
        )
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
        context = torch.cat(
            [dit.img_emb(legacy._repeat_to_batch(clip_feature, batch)), context], dim=1
        )

    patches = dit.patchify(x, control_camera_latents_input)
    target_grid = patches.shape[2:]
    target_tokens = rearrange(patches, "b c f h w -> b (f h w) c").contiguous()
    target_tokens = legacy._add_type_embedding(
        target_tokens,
        tokenlight_type_embedding,
        legacy.TOKENLIGHT_TYPE_RGB_TARGET,
    )
    target_freqs = legacy._freqs_for_grid(dit, target_grid, target_tokens.device)
    stream_t_mod = legacy._repeat_to_batch(t_mod, batch)

    all_tokens: list[torch.Tensor] = []
    all_freqs: list[torch.Tensor] = []
    t_mod_pieces: list[torch.Tensor] = []
    rgb_len = int(target_tokens.shape[1])
    pbr_segments: dict[str, tuple[int, int, tuple[int, int, int]]] = {}

    if tokenlight_source_latents is not None:
        source_tokens, source_grid = legacy._patch_to_tokens(
            dit, tokenlight_source_latents, batch
        )
        source_tokens = legacy._add_type_embedding(
            source_tokens,
            tokenlight_type_embedding,
            legacy.TOKENLIGHT_TYPE_SOURCE,
        )
        all_tokens.append(source_tokens)
        all_freqs.append(
            legacy._freqs_for_grid(dit, source_grid, target_tokens.device)
        )
        t_mod_pieces.append(
            legacy._clean_prefix_t_mod(
                dit,
                int(source_tokens.shape[1]),
                batch,
                stream_t_mod.dtype,
                stream_t_mod.device,
            )
        )

    rgb_start = sum(int(tokens.shape[1]) for tokens in all_tokens)
    all_tokens.append(target_tokens)
    all_freqs.append(target_freqs)
    t_mod_pieces.append(_t_mod_for_token_length(stream_t_mod, rgb_len))

    pbr_streams, legacy_pbr_stream = legacy._pbr_stream_items(
        tokenlight_pbr_stream_latents, tokenlight_pbr_latents
    )
    for stream_index, (stream_name, stream_latents) in enumerate(pbr_streams):
        pbr_target_mask = legacy._bool_mask(
            legacy._pbr_stream_target_value(
                tokenlight_pbr_stream_is_target,
                stream_name,
                tokenlight_pbr_is_target,
            ),
            batch,
            target_tokens.device,
        )
        pbr_tokens, pbr_grid = legacy._patch_to_tokens(dit, stream_latents, batch)
        if pbr_grid != target_grid:
            raise ValueError(
                f"PBR stream {stream_name!r} latent grid {pbr_grid} must match "
                f"RGB target grid {target_grid}"
            )
        target_type = (
            legacy.TOKENLIGHT_TYPE_PBR_TARGET_BASE
            + stream_index * legacy.TOKENLIGHT_TYPE_PBR_STRIDE
        )
        condition_type = (
            legacy.TOKENLIGHT_TYPE_PBR_CONDITION_BASE
            + stream_index * legacy.TOKENLIGHT_TYPE_PBR_STRIDE
        )
        pbr_type = torch.where(
            pbr_target_mask,
            torch.full_like(pbr_target_mask, target_type, dtype=torch.long),
            torch.full_like(pbr_target_mask, condition_type, dtype=torch.long),
        )
        pbr_tokens = legacy._add_type_embedding(
            pbr_tokens, tokenlight_type_embedding, pbr_type
        )
        pbr_start = sum(int(tokens.shape[1]) for tokens in all_tokens)
        pbr_len = int(pbr_tokens.shape[1])
        pbr_segments[stream_name] = (pbr_start, pbr_len, pbr_grid)
        all_tokens.append(pbr_tokens)
        all_freqs.append(
            legacy._freqs_for_grid(dit, pbr_grid, target_tokens.device)
        )
        clean_t_mod = legacy._clean_prefix_t_mod(
            dit,
            pbr_len,
            batch,
            stream_t_mod.dtype,
            stream_t_mod.device,
        )
        noisy_t_mod = _t_mod_for_token_length(stream_t_mod, pbr_len)
        t_mod_pieces.append(
            legacy._select_t_mod_by_mask(
                clean_t_mod, noisy_t_mod, pbr_target_mask
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
        t_mod_pieces.append(
            legacy._clean_prefix_t_mod(
                dit,
                int(light_tokens.shape[1]),
                batch,
                stream_t_mod.dtype,
                stream_t_mod.device,
            )
        )
        all_tokens.append(light_tokens)
        all_freqs.append(
            torch.ones(
                light_tokens.shape[1],
                1,
                target_freqs.shape[-1],
                device=target_tokens.device,
                dtype=target_freqs.dtype,
            )
        )

    x = torch.cat(all_tokens, dim=1)
    freqs = torch.cat(all_freqs, dim=0)
    t_mod = torch.cat(t_mod_pieces, dim=1)
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

    rgb_tokens = x[:, rgb_start : rgb_start + rgb_len]
    rgb = dit.unpatchify(dit.head(rgb_tokens, t_head), target_grid)
    if not pbr_segments:
        return rgb
    pbr_outputs: dict[str, torch.Tensor] = {}
    for stream_name, (pbr_start, pbr_len, pbr_grid) in pbr_segments.items():
        pbr_tokens = x[:, pbr_start : pbr_start + pbr_len]
        pbr_outputs[stream_name] = dit.unpatchify(
            dit.head(pbr_tokens, t_head), pbr_grid
        )
    if legacy_pbr_stream:
        return rgb, next(iter(pbr_outputs.values()))
    return rgb, pbr_outputs


# The safe trainer/inference wrapper use this conventional name locally.
model_fn_wan_video_tokenlight_pbr = model_fn_wan_video_tokenlight_pbr_safe


__all__ = [
    "TokenLightPBRTypeEmbedding",
    "model_fn_wan_video_tokenlight_pbr",
    "model_fn_wan_video_tokenlight_pbr_safe",
    "tokenlight_pbr_type_count",
]
