from __future__ import annotations

import torch
from torch import nn

from project.config import ExperimentConfig
from project.losses.ssim import psnr_value, ssim_index


class ReconstructionObjective(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.lambda_ssim = config.lambda_ssim
        self.lambda_psnr = config.lambda_psnr

    def forward(self, I_blur: torch.Tensor, I: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        l1 = torch.mean(torch.abs(I_blur - I))
        ssim_value = ssim_index(I_blur, I)
        psnr = psnr_value(I_blur, I)
        psnr_term = -psnr / 50.0
        loss = l1 + self.lambda_ssim * (1.0 - ssim_value) + self.lambda_psnr * psnr_term
        metrics = {
            "loss": float(loss.detach().cpu().item()),
            "l1": float(l1.detach().cpu().item()),
            "ssim": float(ssim_value.detach().cpu().item()),
            "psnr": float(psnr.detach().cpu().item()),
        }
        return loss, metrics
