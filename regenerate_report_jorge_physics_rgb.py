from pathlib import Path

import torch

from project.config import ExperimentConfig
from project.data.datamodule import build_dataloaders
from project.model.unrolling_jorge_physics_rgb import JorgePhysicsRGBADMMUnrollingNet
from project.physics.psf import build_rgb_psf_tensor
from project.training.trainer_jorge_physics_rgb import (
    build_final_report,
    prepare_output_dirs,
    select_best_qualitative_sample,
)


def main() -> None:
    run_dir = Path("outputs/jorge_physics_rgb_72")
    config = ExperimentConfig(
        epochs=72,
        run_prefix="jorge_physics_rgb",
        channel_mode="rgb",
        batch_size=4,
        learning_rate=5e-4,
        image_size=256,
        train_split=0.8,
        device="cpu",
        early_stopping_patience=10,
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
    config.run_dir = run_dir
    config.checkpoint_dir = run_dir / "checkpoints"
    config.sample_dir = run_dir / "samples"
    config.psf_dir = run_dir / "physics"
    config.report_dir = run_dir / "reports"

    prepare_output_dirs(config)
    device = torch.device(config.device)
    train_loader, val_loader, report_loader = build_dataloaders(config)
    psf_tensor = build_rgb_psf_tensor(config, device, (620e-9, 550e-9, 450e-9))
    model = JorgePhysicsRGBADMMUnrollingNet(config, psf_tensor).to(device)
    model.load_state_dict(torch.load(config.checkpoint_dir / "best_model.pt", map_location=device))

    qualitative_sample = select_best_qualitative_sample(model, [train_loader, val_loader], config, device)
    build_final_report(model, report_loader, qualitative_sample, config, device)
    print(f"Reporte regenerado en {config.report_dir}")


if __name__ == "__main__":
    main()
