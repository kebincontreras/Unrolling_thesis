from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from project.config import ExperimentConfig
from project.model.blocks import ResidualRefiner


def apply_filter(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    padding = kernel.shape[-1] // 2
    channels = image.shape[1]
    kernel_rgb = kernel.repeat(channels, 1, 1, 1)
    return F.conv2d(image, kernel_rgb, padding=padding, groups=channels)


class UnrolledPrecompensationNet(nn.Module):
    def __init__(self, config: ExperimentConfig, psf: torch.Tensor) -> None:
        super().__init__()
        self.k_stages = config.k_stages
        self.channels = config.input_channels
        self.register_buffer("h", psf)
        self.register_buffer("h_t", torch.flip(psf, dims=(-1, -2)))
        self.alphas = nn.Parameter(torch.full((self.k_stages,), config.alpha_init, dtype=torch.float32))
        self.refiners = nn.ModuleList(
            [ResidualRefiner(config.hidden_channels, self.channels) for _ in range(self.k_stages)]
        )

    def forward(self, I: torch.Tensor) -> dict[str, torch.Tensor]:
        I_new = I.clone()
        stage_outputs: list[torch.Tensor] = []

        for stage_idx in range(self.k_stages):
            blurred = apply_filter(I_new, self.h)
            residual = blurred - I
            gradient = apply_filter(residual, self.h_t)
            z = I_new - self.alphas[stage_idx] * gradient
            I_new = torch.clamp(self.refiners[stage_idx](z, gradient, I), 0.0, 1.0)
            stage_outputs.append(I_new)

        I_blur = apply_filter(I_new, self.h)
        return {
            "I": I,
            "I_new": I_new,
            "I_blur": I_blur,
            "stage_outputs": torch.stack(stage_outputs, dim=1),
        }
