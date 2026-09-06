from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from model.light_encoder import LightokenEncoder


TOKENLIGHT_TYPE_SOURCE = 0
TOKENLIGHT_TYPE_MASK = 1
TOKENLIGHT_TYPE_LIGHT = 2
TOKENLIGHT_TYPE_TARGET = 3


class TokenLightTypeEmbedding(nn.Module):
    """Learned token-type embedding for TokenLight prefix and target tokens."""

    def __init__(self, token_dim: int, num_types: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(num_types), int(token_dim))
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, type_id: int) -> torch.Tensor:
        type_token = self.embedding.weight[int(type_id)].to(device=tokens.device, dtype=tokens.dtype)
        return tokens + type_token.view(1, 1, -1)


def _repeat_to_batch(tensor: torch.Tensor, batch: int) -> torch.Tensor:
    if tensor.shape[0] == batch:
        return tensor
    if tensor.shape[0] != 1:
        raise ValueError(f"Cannot expand batch {tensor.shape[0]} to {batch}")
    return tensor.expand(batch, *tensor.shape[1:]).contiguous()


def _condition_latents_for_patchify(dit: nn.Module, latents: torch.Tensor) -> torch.Tensor:
    in_channels = int(getattr(getattr(dit, "patch_embedding", None), "in_channels", latents.shape[1]))
    if latents.shape[1] == in_channels:
        return latents
    if latents.shape[1] > in_channels:
        return latents[:, :in_channels].contiguous()
    padding = torch.zeros(
        latents.shape[0],
        in_channels - latents.shape[1],
        *latents.shape[2:],
        device=latents.device,
        dtype=latents.dtype,
    )
    return torch.cat([latents, padding], dim=1)


def _add_type_embedding(
    tokens: torch.Tensor,
    type_embedding: TokenLightTypeEmbedding | None,
    type_id: int,
) -> torch.Tensor:
    return tokens if type_embedding is None else type_embedding(tokens, type_id)


def _patch_to_tokens(dit: nn.Module, latents: torch.Tensor, batch: int, control_camera_latents_input=None):
    from einops import rearrange

    latents = _condition_latents_for_patchify(dit, _repeat_to_batch(latents, batch))
    patches = dit.patchify(latents, control_camera_latents_input)
    f, h, w = patches.shape[2:]
    tokens = rearrange(patches, "b c f h w -> b (f h w) c").contiguous()
    return tokens, (f, h, w)


def _freqs_for_grid(dit: nn.Module, grid: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    f, h, w = grid
    freqs = torch.cat(
        [
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    )
    return freqs.reshape(f * h * w, 1, -1).to(device)


def _clean_prefix_t_mod(dit: nn.Module, prefix_len: int, batch: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d

    timesteps = torch.zeros(prefix_len, dtype=dtype, device=device)
    clean = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timesteps).unsqueeze(0))
    clean = dit.time_projection(clean).unflatten(2, (6, dit.dim))
    return clean.expand(batch, -1, -1, -1).contiguous()


def _has_deepspeed_zero3_params(module: nn.Module) -> bool:
    return any(hasattr(param, "ds_id") for param in module.parameters(recurse=True))


def _with_checkpoint_input_grad(inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    if any(isinstance(item, torch.Tensor) and item.requires_grad for item in inputs):
        return inputs
    patched = list(inputs)
    for index, item in enumerate(patched):
        if isinstance(item, torch.Tensor) and torch.is_floating_point(item):
            patched[index] = item.detach().requires_grad_(True)
            return tuple(patched)
    return inputs


def gradient_checkpoint_forward_compatible(
    module: nn.Module,
    use_gradient_checkpointing: bool,
    use_gradient_checkpointing_offload: bool,
    *inputs: Any,
) -> Any:
    if not use_gradient_checkpointing:
        return module(*inputs)
    if _has_deepspeed_zero3_params(module):
        return checkpoint(module, *_with_checkpoint_input_grad(inputs), use_reentrant=True)

    from diffsynth.core.gradient import gradient_checkpoint_forward

    return gradient_checkpoint_forward(
        module,
        use_gradient_checkpointing,
        use_gradient_checkpointing_offload,
        *inputs,
    )


def model_fn_wan_video_tokenlight(
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
    tokenlight_attrs: Mapping[str, Any] | Sequence[Mapping[str, Any]] | torch.Tensor | None = None,
    tokenlight_drop_light: bool | torch.Tensor | Sequence[bool] = False,
    tokenlight_source_latents: torch.Tensor | None = None,
    tokenlight_mask_latents: torch.Tensor | None = None,
    tokenlight_extra_mask_latents: Sequence[torch.Tensor] | None = None,
    tokenlight_additive_condition_latents: torch.Tensor | None = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
) -> torch.Tensor:
    del kwargs
    from einops import rearrange
    from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d

    if getattr(dit, "seperated_timestep", False) and fuse_vae_embedding_in_latents:
        timestep = torch.concat(
            [
                torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
                torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep,
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
    batch = context.shape[0]
    x = latents if latents.shape[0] == batch else torch.cat([latents] * batch, dim=0)
    if y is not None and getattr(dit, "require_vae_embedding", True):
        x = torch.cat([x, _repeat_to_batch(y, batch)], dim=1)
    if clip_feature is not None and getattr(dit, "require_clip_embedding", True):
        context = torch.cat([dit.img_emb(_repeat_to_batch(clip_feature, batch)), context], dim=1)

    patches = dit.patchify(x, control_camera_latents_input)
    target_grid = patches.shape[2:]
    target_tokens = rearrange(patches, "b c f h w -> b (f h w) c").contiguous()
    target_tokens = _add_type_embedding(target_tokens, tokenlight_type_embedding, TOKENLIGHT_TYPE_TARGET)
    if tokenlight_additive_condition_latents is not None:
        condition_tokens, condition_grid = _patch_to_tokens(
            dit,
            tokenlight_additive_condition_latents,
            batch,
            control_camera_latents_input,
        )
        # ``dit.patchify`` is affine (its Conv3d has a bias).  Remove the
        # zero-input response so a zero-initialized additive condition is an
        # exact no-op rather than adding the patch-embedding bias twice.
        zero_condition_tokens, zero_condition_grid = _patch_to_tokens(
            dit,
            torch.zeros_like(tokenlight_additive_condition_latents),
            batch,
            control_camera_latents_input,
        )
        if tuple(condition_grid) != tuple(target_grid):
            raise ValueError(
                "Additive TokenLight condition grid must match the target grid: "
                f"{tuple(condition_grid)} != {tuple(target_grid)}"
            )
        if tuple(zero_condition_grid) != tuple(condition_grid):
            raise RuntimeError("Zero additive TokenLight condition changed the patch grid")
        target_tokens = target_tokens + condition_tokens - zero_condition_tokens
    target_freqs = _freqs_for_grid(dit, target_grid, target_tokens.device)

    prefix_tokens: list[torch.Tensor] = []
    prefix_freqs: list[torch.Tensor] = []
    if tokenlight_source_latents is not None:
        source_tokens, source_grid = _patch_to_tokens(dit, tokenlight_source_latents, batch)
        source_tokens = _add_type_embedding(source_tokens, tokenlight_type_embedding, TOKENLIGHT_TYPE_SOURCE)
        prefix_tokens.append(source_tokens)
        prefix_freqs.append(_freqs_for_grid(dit, source_grid, target_tokens.device))
    if tokenlight_mask_latents is not None:
        mask_tokens, mask_grid = _patch_to_tokens(dit, tokenlight_mask_latents, batch)
        mask_tokens = _add_type_embedding(mask_tokens, tokenlight_type_embedding, TOKENLIGHT_TYPE_MASK)
        prefix_tokens.append(mask_tokens)
        prefix_freqs.append(_freqs_for_grid(dit, mask_grid, target_tokens.device))
    for extra_mask_latents in tokenlight_extra_mask_latents or ():
        mask_tokens, mask_grid = _patch_to_tokens(dit, extra_mask_latents, batch)
        mask_tokens = _add_type_embedding(mask_tokens, tokenlight_type_embedding, TOKENLIGHT_TYPE_MASK)
        prefix_tokens.append(mask_tokens)
        prefix_freqs.append(_freqs_for_grid(dit, mask_grid, target_tokens.device))
    if tokenlight_light_encoder is not None:
        light_tokens = tokenlight_light_encoder(
            tokenlight_attrs,
            batch_size=batch,
            device=target_tokens.device,
            dtype=target_tokens.dtype,
            drop_light=tokenlight_drop_light,
        )
        light_tokens = _add_type_embedding(light_tokens, tokenlight_type_embedding, TOKENLIGHT_TYPE_LIGHT)
        prefix_tokens.append(light_tokens)
        prefix_freqs.append(torch.ones(light_tokens.shape[1], 1, target_freqs.shape[-1], device=target_tokens.device, dtype=target_freqs.dtype))

    if prefix_tokens:
        prefix_len = sum(tokens.shape[1] for tokens in prefix_tokens)
        x = torch.cat([*prefix_tokens, target_tokens], dim=1)
        freqs = torch.cat([*prefix_freqs, target_freqs], dim=0)
        if t_mod.ndim == 4:
            clean_t_mod = _clean_prefix_t_mod(dit, prefix_len, t_mod.shape[0], t_mod.dtype, t_mod.device)
            t_mod = torch.cat([clean_t_mod, t_mod], dim=1)
    else:
        prefix_len = 0
        x = target_tokens
        freqs = target_freqs

    for block in dit.blocks:
        x = gradient_checkpoint_forward_compatible(
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
