"""Wan adaptation of MultiShadow's text-grounded shadow-box conditioning.

Wan is intentionally retained, but the paper's separation between the dense
image pathway and prompt pathway is preserved: source/mask/light are DiT image
prefixes, while semantic and quantized box tokens are appended to Wan's text
cross-attention context.  The latter must not be inserted into image
self-attention, which was an important mismatch in the first implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from model.lightoken_encoder import LightokenEncoder
from model.tokenlight_wan import (
    TOKENLIGHT_TYPE_LIGHT,
    TOKENLIGHT_TYPE_MASK,
    TOKENLIGHT_TYPE_SOURCE,
    TOKENLIGHT_TYPE_TARGET,
    TokenLightTypeEmbedding,
    _add_type_embedding,
    _clean_prefix_t_mod,
    _condition_latents_for_patchify,
    _freqs_for_grid,
    _repeat_to_batch,
    gradient_checkpoint_forward_compatible,
)


class CoShadowLayoutTokenEmbedding(nn.Module):
    """Encode quantized XYXY boxes as exactly four learned layout tokens."""

    def __init__(self, token_dim: int, *, bins: int = 16) -> None:
        super().__init__()
        if int(bins) < 2:
            raise ValueError("CoShadow layout bins must be at least 2")
        self.token_dim = int(token_dim)
        self.bins = int(bins)
        self.x_embedding = nn.Embedding(self.bins, self.token_dim)
        self.y_embedding = nn.Embedding(self.bins, self.token_dim)
        self.coordinate_slot_embedding = nn.Parameter(torch.empty(4, self.token_dim))
        self.layout_type_embedding = nn.Parameter(torch.empty(self.token_dim))
        self.presence_embedding = nn.Embedding(2, self.token_dim)
        self.invalid_coordinate_embedding = nn.Parameter(torch.empty(4, self.token_dim))
        self.norm = nn.LayerNorm(self.token_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.x_embedding.weight, std=0.02)
        nn.init.normal_(self.y_embedding.weight, std=0.02)
        nn.init.normal_(self.coordinate_slot_embedding, std=0.02)
        nn.init.normal_(self.layout_type_embedding, std=0.02)
        nn.init.normal_(self.presence_embedding.weight, std=0.02)
        nn.init.normal_(self.invalid_coordinate_embedding, std=0.02)
        self.norm.reset_parameters()

    def forward(self, bins_xyxy: torch.Tensor, valid: torch.Tensor | Sequence[bool] | bool) -> torch.Tensor:
        if bins_xyxy.ndim == 1:
            bins_xyxy = bins_xyxy.unsqueeze(0)
        if bins_xyxy.ndim != 2 or bins_xyxy.shape[1] != 4:
            raise ValueError(f"Expected bbox bins [B,4], got {tuple(bins_xyxy.shape)}")
        bins_xyxy = bins_xyxy.to(device=self.layout_type_embedding.device, dtype=torch.long)
        if bool(((bins_xyxy < 0) | (bins_xyxy >= self.bins)).any()):
            raise ValueError(f"CoShadow bbox bin index outside [0,{self.bins - 1}]")
        batch = int(bins_xyxy.shape[0])
        valid_tensor = torch.as_tensor(valid, device=bins_xyxy.device, dtype=torch.bool).flatten()
        if valid_tensor.numel() == 1:
            valid_tensor = valid_tensor.expand(batch)
        if valid_tensor.numel() != batch:
            raise ValueError(f"Got {valid_tensor.numel()} bbox-valid flags for batch {batch}")

        tokens = torch.stack(
            [
                self.x_embedding(bins_xyxy[:, 0]),
                self.y_embedding(bins_xyxy[:, 1]),
                self.x_embedding(bins_xyxy[:, 2]),
                self.y_embedding(bins_xyxy[:, 3]),
            ],
            dim=1,
        )
        tokens = tokens + self.coordinate_slot_embedding.unsqueeze(0)
        invalid = self.invalid_coordinate_embedding.unsqueeze(0).expand(batch, -1, -1)
        tokens = torch.where(valid_tensor[:, None, None], tokens, invalid)
        tokens = tokens + self.layout_type_embedding.view(1, 1, -1)
        tokens = tokens + self.presence_embedding(valid_tensor.long())[:, None, :]
        return self.norm(tokens)


class CoShadowPromptEmbedding(nn.Module):
    """Paper-style ``object casting shadow [box]`` cross-attention tokens.

    The two semantic tokens replace the frozen CLIP embeddings used by the
    original SD1.5 model.  This is the smallest faithful Wan adaptation: Wan's
    existing text context remains frozen and these new tokens alone are
    trained, exactly like the paper's newly introduced positional vocabulary.
    """

    def __init__(self, token_dim: int, *, semantic_tokens: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(int(semantic_tokens), int(token_dim)))
        self.norm = nn.LayerNorm(int(token_dim))
        nn.init.normal_(self.embedding, std=0.02)

    def forward(self, batch: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        value = self.norm(self.embedding).to(device=device, dtype=dtype)
        return value.unsqueeze(0).expand(int(batch), -1, -1)


def _actual_bbox_cross_attention_probability(
    cross_attn: nn.Module,
    image_tokens: torch.Tensor,
    context: torch.Tensor,
    *,
    bbox_token_count: int = 4,
    query_chunk_size: int = 256,
) -> torch.Tensor:
    """Compute actual Wan attention mass for the trailing bbox tokens."""
    ctx = context[:, 257:] if bool(getattr(cross_attn, "has_image_input", False)) else context
    if ctx.shape[1] < int(bbox_token_count):
        raise ValueError("Cross-attention context is shorter than the bbox token group")
    heads, head_dim = int(cross_attn.num_heads), int(cross_attn.head_dim)
    q = cross_attn.norm_q(cross_attn.q(image_tokens)).unflatten(-1, (heads, head_dim)).transpose(1, 2)
    k = cross_attn.norm_k(cross_attn.k(ctx)).unflatten(-1, (heads, head_dim)).transpose(1, 2)
    k_t, pieces = k.transpose(-1, -2), []
    for q_piece in q.split(max(1, int(query_chunk_size)), dim=2):
        logits = torch.matmul(q_piece.float(), k_t.float()) * (head_dim ** -0.5)
        selected = (logits[..., -int(bbox_token_count):] - torch.logsumexp(logits, dim=-1, keepdim=True)).exp()
        pieces.append(selected.mean(dim=1).mean(dim=-1))
    probability = torch.cat(pieces, dim=1)
    return probability / probability.sum(dim=1, keepdim=True).clamp_min(1e-8)


def _block_forward_with_actual_alignment(block, x, context, t_mod, freqs, *, target_start, query_chunk_size):
    """Run a stock Wan block while returning its real bbox attention map."""
    from diffsynth.models.wan_video_dit import modulate
    has_seq = t_mod.ndim == 4
    values = (block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=2 if has_seq else 1)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = values
    if has_seq:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (value.squeeze(2) for value in values)
    input_x = modulate(block.norm1(x), shift_msa, scale_msa)
    x = block.gate(x, gate_msa, block.self_attn(input_x, freqs))
    cross_input = block.norm3(x)
    probability = _actual_bbox_cross_attention_probability(
        block.cross_attn, cross_input[:, int(target_start):], context,
        query_chunk_size=query_chunk_size,
    )
    x = x + block.cross_attn(cross_input, context)
    input_x = modulate(block.norm2(x), shift_mlp, scale_mlp)
    return block.gate(x, gate_mlp, block.ffn(input_x)), probability


class CoShadowLatentMaskHead(nn.Module):
    """Predict a shadow mask on the DiT target-token grid."""

    def __init__(self, token_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(token_dim))
        self.projection = nn.Linear(int(token_dim), 1)

    def forward(self, target_tokens: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
        from einops import rearrange

        f, h, w = grid
        logits = self.projection(self.norm(target_tokens))
        return rearrange(logits, "b (f h w) c -> b c f h w", f=f, h=h, w=w).contiguous()


def _patch_to_tokens(dit: nn.Module, latents: torch.Tensor, batch: int, control_camera_latents_input=None):
    from einops import rearrange

    latents = _condition_latents_for_patchify(dit, _repeat_to_batch(latents, batch))
    patches = dit.patchify(latents, control_camera_latents_input)
    grid = tuple(int(value) for value in patches.shape[2:])
    tokens = rearrange(patches, "b c f h w -> b (f h w) c").contiguous()
    return tokens, grid


def _prefix_and_target_t_mod(
    dit: nn.Module,
    t_mod: torch.Tensor,
    *,
    prefix_len: int,
    target_len: int,
    batch: int,
) -> torch.Tensor:
    """Give clean prefix tokens t=0 and target tokens the sampled timestep.

    Wan's one-frame cached path normally produces a rank-3 modulation tensor.
    Concatenating prefix tokens without promoting it would silently broadcast
    the sampled target timestep onto source/mask/layout/light conditions.
    """

    if t_mod.ndim == 3:
        if t_mod.shape[0] == 1 and batch > 1:
            t_mod = t_mod.expand(batch, -1, -1)
        if t_mod.shape[0] != batch:
            raise ValueError(f"t_mod batch {t_mod.shape[0]} does not match context batch {batch}")
        target_mod = t_mod[:, None, :, :].expand(batch, target_len, -1, -1).contiguous()
    elif t_mod.ndim == 4:
        if t_mod.shape[0] == 1 and batch > 1:
            t_mod = t_mod.expand(batch, -1, -1, -1)
        if t_mod.shape[0] != batch or t_mod.shape[1] != target_len:
            raise ValueError(
                f"Tokenwise t_mod shape {tuple(t_mod.shape)} does not match batch/target "
                f"{(batch, target_len)}"
            )
        target_mod = t_mod
    else:
        raise ValueError(f"Unsupported Wan timestep modulation shape: {tuple(t_mod.shape)}")
    if prefix_len <= 0:
        return target_mod
    clean_mod = _clean_prefix_t_mod(dit, prefix_len, batch, t_mod.dtype, t_mod.device)
    return torch.cat([clean_mod, target_mod], dim=1)


def model_fn_wan_video_tokenlight_coshadow(
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
    coshadow_layout_embedding: CoShadowLayoutTokenEmbedding | None = None,
    coshadow_prompt_embedding: CoShadowPromptEmbedding | None = None,
    coshadow_collect_actual_alignment: bool = False,
    coshadow_alignment_query_chunk_size: int = 256,
    coshadow_mask_head: CoShadowLatentMaskHead | None = None,
    tokenlight_attrs: Mapping[str, Any] | Sequence[Mapping[str, Any]] | torch.Tensor | None = None,
    tokenlight_drop_light: bool | torch.Tensor | Sequence[bool] = False,
    tokenlight_source_latents: torch.Tensor | None = None,
    tokenlight_mask_latents: torch.Tensor | None = None,
    coshadow_layout_bins: torch.Tensor | None = None,
    coshadow_bbox_valid: torch.Tensor | Sequence[bool] | bool = True,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
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
    x_latents = latents if latents.shape[0] == batch else torch.cat([latents] * batch, dim=0)
    if y is not None and getattr(dit, "require_vae_embedding", True):
        x_latents = torch.cat([x_latents, _repeat_to_batch(y, batch)], dim=1)
    if clip_feature is not None and getattr(dit, "require_clip_embedding", True):
        context = torch.cat([dit.img_emb(_repeat_to_batch(clip_feature, batch)), context], dim=1)

    patches = dit.patchify(x_latents, control_camera_latents_input)
    target_grid = tuple(int(value) for value in patches.shape[2:])
    target_tokens = rearrange(patches, "b c f h w -> b (f h w) c").contiguous()
    target_tokens = _add_type_embedding(target_tokens, tokenlight_type_embedding, TOKENLIGHT_TYPE_TARGET)
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
    layout_tokens = None
    if coshadow_layout_embedding is not None:
        if coshadow_layout_bins is None:
            raise ValueError("CoShadow layout embedding requires coshadow_layout_bins")
        layout_tokens = coshadow_layout_embedding(coshadow_layout_bins, coshadow_bbox_valid)
        layout_tokens = layout_tokens.to(device=target_tokens.device, dtype=target_tokens.dtype)
        # MultiShadow puts these tokens in the prompt pathway.  Append them to
        # Wan's already projected text context so every block sees them through
        # cross-attention, not image-token self-attention.
        semantic = (
            coshadow_prompt_embedding(
                batch, device=target_tokens.device, dtype=target_tokens.dtype
            )
            if coshadow_prompt_embedding is not None
            else layout_tokens[:, :0]
        )
        context = torch.cat([context, semantic, layout_tokens], dim=1)
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
        prefix_freqs.append(
            torch.ones(
                light_tokens.shape[1],
                1,
                target_freqs.shape[-1],
                device=target_tokens.device,
                dtype=target_freqs.dtype,
            )
        )

    prefix_len = sum(tokens.shape[1] for tokens in prefix_tokens)
    if prefix_tokens:
        x = torch.cat([*prefix_tokens, target_tokens], dim=1)
        freqs = torch.cat([*prefix_freqs, target_freqs], dim=0)
        t_mod = _prefix_and_target_t_mod(
            dit,
            t_mod,
            prefix_len=prefix_len,
            target_len=target_tokens.shape[1],
            batch=batch,
        )
    else:
        x = target_tokens
        freqs = target_freqs

    alignment_maps = []
    block_count = len(dit.blocks)
    alignment_blocks = {block_count // 2, (3 * block_count) // 4, block_count - 1}
    for block_index, block in enumerate(dit.blocks):
        if coshadow_collect_actual_alignment and block_index in alignment_blocks:
            x, probability = _block_forward_with_actual_alignment(
                block, x, context, t_mod, freqs, target_start=prefix_len,
                query_chunk_size=coshadow_alignment_query_chunk_size,
            )
            alignment_maps.append(probability)
        else:
            x = gradient_checkpoint_forward_compatible(
                block, use_gradient_checkpointing, use_gradient_checkpointing_offload,
                x, context, t_mod, freqs,
            )
    target_hidden = x[:, prefix_len:] if prefix_len else x
    velocity = dit.unpatchify(dit.head(target_hidden, t_head), target_grid)
    alignment = torch.stack(alignment_maps, dim=0) if alignment_maps else None
    if coshadow_mask_head is None and alignment is None:
        return velocity
    mask_logits = coshadow_mask_head(target_hidden, target_grid) if coshadow_mask_head is not None else None
    return velocity, mask_logits, alignment
