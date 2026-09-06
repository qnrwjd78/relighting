"""LightNet: inverse single-light estimation from RGB and PBR geometry."""

from __future__ import annotations

import torch
from torch import nn

from .models import DownBlock, ResidualBlock


class LightNet(nn.Module):
    """Estimate light parameters from paired source and target RGB images."""

    def __init__(self, in_channels: int = 6, base_channels: int = 32) -> None:
        super().__init__()
        b = base_channels
        self.encoder = nn.Sequential(
            ResidualBlock(in_channels, b),
            DownBlock(b, b * 2),
            DownBlock(b * 2, b * 4),
            DownBlock(b * 4, b * 8),
            DownBlock(b * 8, b * 8),
            DownBlock(b * 8, b * 8),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(b * 8, b * 8),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(b * 8, 10),
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.head(self.pool(self.encoder(inputs)))
        return {
            "direction": torch.nn.functional.normalize(raw[:, 0:3], dim=1, eps=1e-6),
            "log_distance": raw[:, 3],
            "log_intensity": raw[:, 4],
            "log_radius": raw[:, 5],
            "color": torch.sigmoid(raw[:, 6:9]),
            "ambient_scale": torch.sigmoid(raw[:, 9]) * 1.5,
        }
