from __future__ import annotations

import numpy as np


def make_frequency_grid(n: int, dx: float) -> tuple[np.ndarray, np.ndarray]:
    freq = np.fft.fftfreq(n, d=dx)
    return np.meshgrid(freq, freq)


def angular_spectrum_propagate(field: np.ndarray, wavelength: float, dx: float, z: float) -> np.ndarray:
    fx, fy = make_frequency_grid(field.shape[0], dx)
    k = 2.0 * np.pi / wavelength
    kx = 2.0 * np.pi * fx
    ky = 2.0 * np.pi * fy
    kz = np.sqrt((k**2 - kx**2 - ky**2).astype(np.complex128))
    transfer = np.exp(1j * kz * z)
    return np.fft.ifft2(np.fft.fft2(field) * transfer)
