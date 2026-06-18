from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from project.config import ExperimentConfig
from project.utils.io_utils import ensure_dir


def circular_pupil(n: int, extent: float, radius: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-extent / 2.0, extent / 2.0, n, endpoint=False)
    xx, yy = np.meshgrid(x, x)
    pupil = (xx**2 + yy**2 <= radius**2).astype(np.complex128)
    return x, pupil


def lens_phase(xx: np.ndarray, yy: np.ndarray, wavelength: float, focal_length: float) -> np.ndarray:
    k = 2.0 * np.pi / wavelength
    return np.exp(-1j * k * (xx**2 + yy**2) / (2.0 * focal_length))


def defocus_phase(
    xx: np.ndarray,
    yy: np.ndarray,
    pupil_radius: float,
    severity: float,
) -> np.ndarray:
    rho2 = (xx**2 + yy**2) / max(pupil_radius**2, 1e-18)
    phase = np.pi * severity * rho2
    return np.exp(1j * phase)


def fraunhofer_focal_field(
    pupil_field: np.ndarray,
    x_in: np.ndarray,
    wavelength: float,
    focal_length: float,
    x_out: np.ndarray,
) -> np.ndarray:
    dx = x_in[1] - x_in[0]
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_field))) * (dx**2)
    fx = np.fft.fftshift(np.fft.fftfreq(x_in.size, d=dx))
    x_focal = wavelength * focal_length * fx

    ix = np.searchsorted(x_focal, x_out)
    ix = np.clip(ix, 1, len(x_focal) - 1)
    left = x_focal[ix - 1]
    right = x_focal[ix]
    choose_right = np.abs(x_out - right) < np.abs(x_out - left)
    idx = ix.copy()
    idx[~choose_right] -= 1
    return spectrum[np.ix_(idx, idx)]


def crop_center(data: np.ndarray, size: int) -> np.ndarray:
    center_y = data.shape[0] // 2
    center_x = data.shape[1] // 2
    half = size // 2
    return data[center_y - half:center_y + half + 1, center_x - half:center_x + half + 1]


def normalize_psf(psf: np.ndarray) -> np.ndarray:
    psf = np.maximum(psf, 0.0)
    psf_sum = np.sum(psf)
    if psf_sum <= 0:
        raise ValueError("La PSF generada no tiene energia positiva.")
    return psf / psf_sum


def save_psf_preview(psf: np.ndarray, output_path: Path) -> None:
    preview = psf - psf.min()
    preview = preview / max(preview.max(), 1e-12)
    Image.fromarray((255.0 * preview).astype(np.uint8)).save(output_path)


def save_psf_panel(psf: np.ndarray, output_path: Path, severity: float) -> None:
    normalized = psf / max(np.max(psf), 1e-12)
    log_view = np.log1p(30.0 * normalized)
    log_view = log_view / max(np.max(log_view), 1e-12)

    panel_size = 280
    header_h = 44
    canvas = Image.new("L", (panel_size * 2, panel_size + header_h), color=255)
    draw = ImageDraw.Draw(canvas)

    left = Image.fromarray((255.0 * normalized).astype(np.uint8)).resize((panel_size, panel_size), Image.Resampling.BICUBIC)
    right = Image.fromarray((255.0 * log_view).astype(np.uint8)).resize((panel_size, panel_size), Image.Resampling.BICUBIC)

    canvas.paste(left, (0, header_h))
    canvas.paste(right, (panel_size, header_h))
    draw.text((10, 12), f"PSF lineal | severidad={severity:.2f}", fill=0)
    draw.text((panel_size + 10, 12), "PSF log", fill=0)
    canvas.save(output_path)


def generate_fixed_psf(config: ExperimentConfig) -> np.ndarray:
    return generate_psf_for_wavelength(config, config.wavelength)


def generate_psf_for_wavelength(config: ExperimentConfig, wavelength: float) -> np.ndarray:
    x_in, pupil = circular_pupil(config.psf_grid_size, config.pupil_extent, config.pupil_radius)
    xx_in, yy_in = np.meshgrid(x_in, x_in)
    pupil_field = pupil * defocus_phase(xx_in, yy_in, config.pupil_radius, config.myopia_severity)

    x_out = np.linspace(-config.psf_output_extent / 2.0, config.psf_output_extent / 2.0, config.psf_output_n)
    field = fraunhofer_focal_field(
        pupil_field,
        x_in,
        wavelength,
        config.propagation_distance,
        x_out,
    )
    psf = np.abs(field) ** 2
    kernel = crop_center(psf, config.psf_kernel_size)
    return normalize_psf(kernel)


def build_psf_tensor(config: ExperimentConfig, device: torch.device) -> torch.Tensor:
    ensure_dir(config.psf_dir)
    psf = generate_fixed_psf(config)
    severity_tag = str(config.myopia_severity).replace(".", "_")
    save_psf_preview(psf, config.psf_dir / f"psf_myopia_severity_{severity_tag}.png")
    save_psf_panel(psf, config.psf_dir / f"psf_panel_severity_{severity_tag}.png", config.myopia_severity)
    np.save(config.psf_dir / f"psf_myopia_severity_{severity_tag}.npy", psf)
    tensor = torch.from_numpy(psf.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)


def build_rgb_psf_tensor(
    config: ExperimentConfig,
    device: torch.device,
    wavelengths: tuple[float, float, float],
) -> torch.Tensor:
    ensure_dir(config.psf_dir)
    channel_names = ("red", "green", "blue")
    psfs: list[np.ndarray] = []
    severity_tag = str(config.myopia_severity).replace(".", "_")

    for channel_name, wavelength in zip(channel_names, wavelengths):
        psf = generate_psf_for_wavelength(config, wavelength)
        psfs.append(psf)
        wavelength_nm = int(round(wavelength * 1e9))
        save_psf_preview(
            psf,
            config.psf_dir / f"psf_{channel_name}_{wavelength_nm}nm_severity_{severity_tag}.png",
        )
        save_psf_panel(
            psf,
            config.psf_dir / f"psf_panel_{channel_name}_{wavelength_nm}nm_severity_{severity_tag}.png",
            config.myopia_severity,
        )
        np.save(
            config.psf_dir / f"psf_{channel_name}_{wavelength_nm}nm_severity_{severity_tag}.npy",
            psf,
        )

    stacked = np.stack(psfs, axis=0).astype(np.float32)
    tensor = torch.from_numpy(stacked).unsqueeze(1)
    return tensor.to(device)
