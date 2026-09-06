"""VisibilityNet for front-facing object visibility under one point light."""

from __future__ import annotations

import torch
from torch import nn

from .models import DownBlock, ResidualBlock, UpBlock


class VisibilityNet(nn.Module):
    """Predict logits for the fixed32 direct-lit geometry-ray mask."""

    def __init__(self, in_channels: int = 15, base_channels: int = 32) -> None:
        super().__init__()
        b = base_channels
        self.stem = ResidualBlock(in_channels, b)
        self.down1 = DownBlock(b, b * 2)
        self.down2 = DownBlock(b * 2, b * 4)
        self.down3 = DownBlock(b * 4, b * 8)
        self.down4 = DownBlock(b * 8, b * 8)
        self.middle = nn.Sequential(ResidualBlock(b * 8, b * 8), ResidualBlock(b * 8, b * 8))
        self.up4 = UpBlock(b * 8, b * 8, b * 8)
        self.up3 = UpBlock(b * 8, b * 4, b * 4)
        self.up2 = UpBlock(b * 4, b * 2, b * 2)
        self.up1 = UpBlock(b * 2, b, b)
        self.head = nn.Conv2d(b, 1, 3, padding=1)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        x0 = self.stem(inputs)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.middle(self.down4(x3))
        decoded = self.up1(self.up2(self.up3(self.up4(x4, x3), x2), x1), x0)
        return {"visibility_logits": self.head(decoded)}
