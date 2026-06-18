from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def make_unique_run_dir(output_dir: Path, base_name: str) -> Path:
    candidate = output_dir / base_name
    if not candidate.exists():
        return candidate

    suffix = 1
    while True:
        candidate = output_dir / f"{base_name}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


@dataclass
class ExperimentConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    image_dir: Path = field(init=False)
    validation_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    run_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    sample_dir: Path = field(init=False)
    psf_dir: Path = field(init=False)
    report_dir: Path = field(init=False)
    image_size: int = 256
    channel_mode: str = "rgb"
    batch_size: int = 4
    num_workers: int = 0
    train_horizontal_flip: bool = False
    epochs: int = 20
    run_prefix: str = "epochs"
    learning_rate: float = 1e-3
    train_split: float = 0.8
    seed: int = 7
    device: str = "cpu"
    early_stopping_patience: int = 20
    lr_scheduler_patience: int = 4
    lr_scheduler_factor: float = 0.5
    min_learning_rate: float = 1e-5
    k_stages: int = 5
    hidden_channels: int = 24
    alpha_init: float = 0.15
    rho_init: float = 0.25
    lambda_ssim: float = 0.3
    lambda_psnr: float = 0.02
    log_every: int = 1
    psf_grid_size: int = 128
    psf_kernel_size: int = 41
    pupil_radius: float = 1.2e-3
    pupil_extent: float = 5.0e-3
    psf_output_extent: float = 40e-6
    psf_output_n: int = 121
    wavelength: float = 550e-9
    propagation_distance: float = 24e-3
    defocus_plane_shift: float = 0.001e-3
    myopia_severity: float = 0.5
    wiener_k: float = 1e-3

    def __post_init__(self) -> None:
        valid_modes = {"rgb", "red"}
        if self.channel_mode not in valid_modes:
            raise ValueError(f"channel_mode debe ser uno de {sorted(valid_modes)}")
        self.image_dir = self.project_root / "Image"
        self.validation_dir = self.image_dir / "Validation"
        self.output_dir = self.project_root / "outputs"
        base_name = f"{self.run_prefix}_{self.epochs}"
        self.run_dir = make_unique_run_dir(self.output_dir, base_name)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.sample_dir = self.run_dir / "samples"
        self.psf_dir = self.run_dir / "physics"
        self.report_dir = self.run_dir / "reports"

    @property
    def input_channels(self) -> int:
        return 1 if self.channel_mode == "red" else 3
