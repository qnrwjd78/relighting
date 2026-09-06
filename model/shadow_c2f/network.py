"""Direct 480x480 geometry-conditioned Adapter shadow refinement network.

The public class name is kept for checkpoint/script compatibility.  The
default model is now a single full-resolution residual ResUNet: there is no
separately supervised 60x60 coarse mask.  Exact light position is encoded as
multiple tokens and injected at the bottleneck with cross-attention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


def _groups(channels: int) -> int:
    for value in (16, 8, 4, 2, 1):
        if channels % value == 0 and channels // value >= 2:
            return value
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(x) + self.skip(x))


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(),
            ResidualBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ResidualBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat((x, skip), dim=1))


class LightTokenEncoder(nn.Module):
    """Turn one exact [x,y,z] position into several useful attention tokens."""

    def __init__(self, hidden: int, tokens: int = 8) -> None:
        super().__init__()
        self.tokens = int(tokens)
        frequencies = (1.0, 2.0, 4.0, 8.0)
        self.register_buffer("frequencies", torch.tensor(frequencies), persistent=False)
        encoded = 3 + 3 * 2 * len(frequencies)
        self.projection = nn.Sequential(
            nn.Linear(encoded, hidden), nn.SiLU(), nn.Linear(hidden, hidden * self.tokens)
        )

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        angles = position[..., None] * self.frequencies
        encoded = torch.cat((position, angles.sin().flatten(-2), angles.cos().flatten(-2)), dim=-1)
        return self.projection(encoded).view(position.shape[0], self.tokens, -1)


@dataclass(frozen=True)
class ShadowC2FConfig:
    """Configuration for the direct 480x480 refiner.

    ``coarse_size`` and ``coarse_base_channels`` remain accepted so old
    manifests/checkpoint readers fail gracefully, but are not used by the
    single-stage architecture.
    """

    coarse_size: tuple[int, int] = (60, 60)
    coarse_base_channels: int = 32
    fine_channels: int = 32
    fine_blocks: int = 4
    use_receiver_mask: bool = True
    adapter_feature_channels: int = 1
    light_tokens: int = 8
    attention_heads: int = 8
    eps: float = 1e-6
    architecture: str = "direct_resunet_v2"

    def __post_init__(self) -> None:
        if self.architecture != "direct_resunet_v2":
            raise ValueError(f"Only direct_resunet_v2 is supported, got {self.architecture!r}")
        if self.fine_channels <= 0 or self.fine_blocks <= 0 or self.light_tokens <= 0:
            raise ValueError("fine_channels, fine_blocks and light_tokens must be positive")
        if self.attention_heads <= 0 or (self.fine_channels * 8) % self.attention_heads:
            raise ValueError("attention_heads must divide bottleneck channels")
        if self.adapter_feature_channels != 1:
            raise ValueError("direct_resunet_v2 consumes one Adapter positive-delta channel")
        if self.eps <= 0:
            raise ValueError("eps must be positive")

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["coarse_size"] = list(self.coarse_size)
        return values


def build_adapter_features(target_probability: torch.Tensor, source_probability: torch.Tensor) -> torch.Tensor:
    """Compatibility helper returning target, source, and positive delta."""
    if target_probability.shape != source_probability.shape or target_probability.ndim != 4:
        raise ValueError("adapter probabilities must have matching [B,1,H,W] shapes")
    delta = (target_probability - source_probability).clamp_min(0.0)
    return torch.cat((target_probability, source_probability, delta), dim=1)


class ShadowCoarseToFine(nn.Module):
    """Single-stage 480x480 ResUNet with light-position bottleneck attention."""

    def __init__(self, config: ShadowC2FConfig | None = None) -> None:
        super().__init__()
        self.config = config or ShadowC2FConfig()
        c = self.config.fine_channels
        bottleneck = c * 8
        self.stem = ResidualBlock(8, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, bottleneck)
        self.middle = nn.Sequential(*[ResidualBlock(bottleneck, bottleneck) for _ in range(self.config.fine_blocks)])
        self.light_encoder = LightTokenEncoder(bottleneck, self.config.light_tokens)
        self.light_attention = nn.MultiheadAttention(
            bottleneck, self.config.attention_heads, batch_first=True
        )
        self.up3 = Up(bottleneck, c * 4, c * 4)
        self.up2 = Up(c * 4, c * 2, c * 2)
        self.up1 = Up(c * 2, c, c)
        self.head = nn.Conv2d(c, 1, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        image: torch.Tensor,
        object_mask: torch.Tensor,
        point_map: torch.Tensor,
        light_position: torch.Tensor,
        coarse_prior: torch.Tensor | None = None,
        adapter_shadow_probability: torch.Tensor | None = None,
        *,
        receiver_mask: torch.Tensor | None = None,
        adapter_features: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"image must be [B,3,H,W], got {tuple(image.shape)}")
        b, _, h, w = image.shape
        expected = (b, 1, h, w)
        for name, value in (("object_mask", object_mask), ("point_map", point_map)):
            channels = 1 if name == "object_mask" else 3
            if tuple(value.shape) != (b, channels, h, w):
                raise ValueError(f"{name} must be {(b, channels, h, w)}, got {tuple(value.shape)}")
        if self.config.use_receiver_mask and receiver_mask is None:
            raise ValueError("receiver_mask is required for output/loss gating")
        delta = None
        if adapter_features is not None:
            if adapter_features.shape[0] != b or adapter_features.shape[1] < 3 or tuple(adapter_features.shape[-2:]) != (h, w):
                raise ValueError("adapter_features must contain [B,3,H,W] target/source/delta features")
            delta = adapter_features[:, 2:3]
        elif adapter_shadow_probability is not None:
            delta = adapter_shadow_probability
        elif coarse_prior is not None:
            delta = coarse_prior
        else:
            raise ValueError("an Adapter positive-delta mask is required")
        if tuple(delta.shape) != expected:
            delta = F.interpolate(delta, size=(h, w), mode="bilinear", align_corners=False)
        delta = delta.to(device=image.device, dtype=image.dtype).clamp(0.0, 1.0)
        if light_position.ndim == 1:
            light_position = light_position[None].expand(b, -1)
        if tuple(light_position.shape) != (b, 3) or not bool(torch.isfinite(light_position).all()):
            raise ValueError(f"light_position must be finite [{b},3]")
        x = torch.cat((image, delta, object_mask.to(image), point_map.to(image)), dim=1)
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        z = self.middle(self.down3(s2))
        tokens = self.light_encoder(light_position.to(device=image.device, dtype=image.dtype))
        query = z.flatten(2).transpose(1, 2)
        attended, _ = self.light_attention(query, tokens, tokens, need_weights=False)
        z = z + attended.transpose(1, 2).reshape_as(z)
        z = self.up3(z, s2)
        z = self.up2(z, s1)
        z = self.up1(z, s0)
        residual_logits = self.head(z)
        initial_logits = torch.logit(delta.clamp(self.config.eps, 1.0 - self.config.eps))
        logits = initial_logits + residual_logits
        mask = torch.sigmoid(logits)
        if receiver_mask is not None:
            gate = receiver_mask.to(mask) * (1.0 - object_mask.to(mask))
            receiver_masked = mask * gate
        else:
            receiver_masked = mask
        # The aliases keep downstream diagnostics readable; no coarse branch is
        # trained in direct_resunet_v2.
        return {
            "logits": logits,
            "mask": mask,
            "residual_logits": residual_logits,
            "initial_logits": initial_logits,
            "coarse_logits": logits,
            "coarse_logits_full": logits,
            "receiver_masked_mask": receiver_masked,
            "single_stage": torch.ones((), device=logits.device),
        }


__all__ = ["ShadowC2FConfig", "ShadowCoarseToFine", "build_adapter_features"]
