from __future__ import annotations

import torch
from torch import nn


class ResidualRefiner(nn.Module):
    def __init__(self, hidden_channels: int, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3 * channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, current: torch.Tensor, gradient: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        stacked = torch.cat([current, gradient, target], dim=1)
        correction = self.layers(stacked)
        return current + correction
