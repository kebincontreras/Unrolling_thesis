from project.config import ExperimentConfig
from project.training.trainer_jorge_physics import run_experiment_jorge_physics


def main() -> None:
    config = ExperimentConfig(
        epochs=60,
        run_prefix="jorge_physics",
        channel_mode="red",
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
    run_experiment_jorge_physics(config)


if __name__ == "__main__":
    main()
