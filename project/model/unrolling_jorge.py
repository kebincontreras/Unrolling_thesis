from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from project.config import ExperimentConfig
from project.physics.wiener import psf_to_otf


def apply_filter(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    padding = kernel.shape[-1] // 2
    channels = image.shape[1]
    kernel_rgb = kernel.repeat(channels, 1, 1, 1)
    return F.conv2d(image, kernel_rgb, padding=padding, groups=channels)


class JorgeDenoiser(nn.Module):
    def __init__(self, hidden_channels: int, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(2 * channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, primal_with_dual: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        stacked = torch.cat([primal_with_dual, target], dim=1)
        correction = self.layers(stacked)
        return torch.clamp(primal_with_dual + correction, 0.0, 1.0)


class JorgeADMMUnrollingNet(nn.Module):
    def __init__(self, config: ExperimentConfig, psf: torch.Tensor) -> None:
        super().__init__()
        self.k_stages = config.k_stages
        self.channels = config.input_channels
        self.register_buffer("h", psf)
        self.rhos = nn.Parameter(torch.full((self.k_stages,), config.rho_init, dtype=torch.float32))
        self.denoisers = nn.ModuleList(
            [JorgeDenoiser(config.hidden_channels, self.channels) for _ in range(self.k_stages)]
        )

    def _x_update(
        self,
        I: torch.Tensor,
        z: torch.Tensor,
        u: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        otf = psf_to_otf(self.h, (I.shape[-2], I.shape[-1]))
        if otf.shape[1] == 1 and I.shape[1] > 1:
            otf = otf.repeat(1, I.shape[1], 1, 1)

        rhs = torch.conj(otf) * torch.fft.fft2(I) + rho * torch.fft.fft2(z - u)
        denom = torch.abs(otf) ** 2 + rho
        x = torch.real(torch.fft.ifft2(rhs / denom))
        return torch.clamp(x, 0.0, 1.0)

    def forward(self, I: torch.Tensor) -> dict[str, torch.Tensor]:
        x = I.clone()
        z = I.clone()
        u = torch.zeros_like(I)
        stage_outputs: list[torch.Tensor] = []

        for stage_idx in range(self.k_stages):
            rho = torch.clamp(self.rhos[stage_idx], min=1e-4)
            x = self._x_update(I, z, u, rho)
            z = self.denoisers[stage_idx](x + u, I)
            u = u + x - z
            stage_outputs.append(z)

        I_new = z
        I_blur = apply_filter(I_new, self.h)
        return {
            "I": I,
            "I_new": I_new,
            "I_blur": I_blur,
            "stage_outputs": torch.stack(stage_outputs, dim=1),
        }
