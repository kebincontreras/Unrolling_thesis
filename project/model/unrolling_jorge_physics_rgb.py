from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from project.config import ExperimentConfig
from project.physics.wiener import psf_to_otf


def apply_filter(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    padding = kernel.shape[-1] // 2
    channels = image.shape[1]
    if kernel.shape[0] == channels:
        channel_kernel = kernel
    elif kernel.shape[0] == 1:
        channel_kernel = kernel.repeat(channels, 1, 1, 1)
    else:
        raise ValueError("El numero de PSF no coincide con los canales de la imagen.")
    return F.conv2d(image, channel_kernel, padding=padding, groups=channels)


class PhysicsGuidedRGBDenoiser(nn.Module):
    def __init__(self, hidden_channels: int, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(4 * channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        hx: torch.Tensor,
        grad: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        stacked = torch.cat([x, hx, grad, target], dim=1)
        correction = self.layers(stacked)
        return torch.clamp(x + correction, 0.0, 1.0)


class JorgePhysicsRGBADMMUnrollingNet(nn.Module):
    def __init__(self, config: ExperimentConfig, psf: torch.Tensor) -> None:
        super().__init__()
        self.k_stages = config.k_stages
        self.channels = config.input_channels
        self.register_buffer("h", psf)
        self.register_buffer("h_t", torch.flip(psf, dims=(-1, -2)))
        self.rhos = nn.Parameter(torch.full((self.k_stages,), config.rho_init, dtype=torch.float32))
        self.denoisers = nn.ModuleList(
            [PhysicsGuidedRGBDenoiser(config.hidden_channels, self.channels) for _ in range(self.k_stages)]
        )

    def _otf_channels(self, spatial_shape: tuple[int, int]) -> torch.Tensor:
        otf_list = []
        for idx in range(self.h.shape[0]):
            channel_otf = psf_to_otf(self.h[idx:idx + 1], spatial_shape)
            otf_list.append(channel_otf)
        return torch.cat(otf_list, dim=1)

    def _x_update(
        self,
        I: torch.Tensor,
        z: torch.Tensor,
        u: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        otf = self._otf_channels((I.shape[-2], I.shape[-1]))
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
            hx = apply_filter(x, self.h)
            residual = hx - I
            grad = apply_filter(residual, self.h_t)
            z = self.denoisers[stage_idx](x + u, hx, grad, I)
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
