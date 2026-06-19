<<<<<<< HEAD
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _has_required_packages(python_executable: str) -> bool:
    check_code = (
        "import torch, numpy, PIL, matplotlib, lpips, torchvision; "
        "print('ok')"
    )
    try:
        result = subprocess.run(
            [python_executable, "-c", check_code],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return result.returncode == 0


def _bootstrap_compatible_python() -> None:
    if os.environ.get("UNROLLING_PYTHON_BOOTSTRAPPED") == "1":
        return

    if _has_required_packages(sys.executable):
        return

    home = Path.home()
    candidates = [
        home / "mlenv" / "bin" / "python",
        home / "miniconda3" / "bin" / "python3",
    ]
    for candidate in candidates:
        if candidate.exists() and _has_required_packages(str(candidate)):
            env = os.environ.copy()
            env["UNROLLING_PYTHON_BOOTSTRAPPED"] = "1"
            os.execve(str(candidate), [str(candidate), __file__, *sys.argv[1:]], env)

    raise ModuleNotFoundError(
        "No se encontro un interprete de Python con las dependencias requeridas "
        "(torch, numpy, matplotlib, lpips, torchvision). "
        "Prueba con ~/miniconda3/bin/python3 o crea ~/mlenv segun el README."
    )


_bootstrap_compatible_python()

=======
>>>>>>> 81619f677812dc90ccebf2f11fb7b0f57ed4e709
from project.config import ExperimentConfig
from project.training.trainer_jorge_physics_rgb import run_experiment_jorge_physics_rgb


def main() -> None:
    config = ExperimentConfig(
<<<<<<< HEAD
        epochs=1200,
        run_prefix="jorge_physics_rgb",
        channel_mode="rgb",
        batch_size=32,
        learning_rate=5e-4,
        image_size=256,
        train_split=0.8,
        device="cuda",
        early_stopping_patience=100,
=======
        epochs=72,
        run_prefix="jorge_physics_rgb",
        channel_mode="rgb",
        batch_size=4,
        learning_rate=5e-4,
        image_size=256,
        train_split=0.8,
        device="cpu",
        early_stopping_patience=10,
>>>>>>> 81619f677812dc90ccebf2f11fb7b0f57ed4e709
        k_stages=8,
        hidden_channels=32,
        rho_init=0.25,
        lambda_ssim=0.3,
        lambda_psnr=0.01,
        myopia_severity=0.5,
        psf_grid_size=256,
        psf_kernel_size=41,
        pupil_radius=1.5e-3,
        pupil_extent=4.0e-3,
        psf_output_extent=40e-6,
        psf_output_n=121,
        wavelength=550e-9,
        propagation_distance=24e-3,
        defocus_plane_shift=0.0,
        seed=7,
    )
    run_experiment_jorge_physics_rgb(config)


if __name__ == "__main__":
    main()
