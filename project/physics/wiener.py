from __future__ import annotations

import torch
import torch.nn.functional as F


def psf_to_otf(psf: torch.Tensor, image_shape: tuple[int, int]) -> torch.Tensor:
    _, _, kh, kw = psf.shape
    height, width = image_shape
    padded = F.pad(psf, (0, width - kw, 0, height - kh))
    padded = torch.roll(padded, shifts=(-kh // 2, -kw // 2), dims=(-2, -1))
    return torch.fft.fft2(padded)


def wiener_deconvolution(blurred: torch.Tensor, psf: torch.Tensor, balance: float) -> torch.Tensor:
    otf = psf_to_otf(psf, (blurred.shape[-2], blurred.shape[-1]))
    otf = otf.squeeze(1)
    if otf.shape[0] == 1 and blurred.shape[1] > 1:
        otf = otf.repeat(blurred.shape[1], 1, 1)
    elif otf.shape[0] != blurred.shape[1]:
        raise ValueError("El numero de PSF debe ser 1 o coincidir con los canales de la imagen.")
    otf = otf.unsqueeze(0)
    blurred_fft = torch.fft.fft2(blurred)
    denom = torch.abs(otf) ** 2 + balance
    estimate_fft = torch.conj(otf) * blurred_fft / denom
    estimate = torch.real(torch.fft.ifft2(estimate_fft))
    return torch.clamp(estimate, 0.0, 1.0)
