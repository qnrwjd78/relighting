"""Shared convolution blocks for the three physical-task models."""

from __future__ import annotations

import torch
from torch import nn


def groups(channels: int) -> int:
    for value in (16, 8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups(out_channels), out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        self.activation = nn.SiLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(inputs) + self.skip(inputs))


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.block = ResidualBlock(out_channels, out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(inputs))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.block = ResidualBlock(out_channels + skip_channels, out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = self.up(inputs)
        if inputs.shape[-2:] != skip.shape[-2:]:
            inputs = torch.nn.functional.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([inputs, skip], dim=1))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
