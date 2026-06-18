from __future__ import annotations

from pathlib import Path

import torch

from project.config import ExperimentConfig
from project.data.datamodule import build_dataloaders
from project.model.unrolling_jorge import JorgeADMMUnrollingNet
from project.physics.psf import build_psf_tensor
from project.training.trainer_jorge import (
    build_final_report,
    prepare_output_dirs,
    select_best_qualitative_sample,
)
from project.utils.io_utils import save_json, set_seed


def scaling_equivariance_loss(
    model: JorgeADMMUnrollingNet,
    I: torch.Tensor,
    alpha_min: float,
    alpha_max: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch_size = I.shape[0]
    alphas = torch.empty(batch_size, 1, 1, 1, device=I.device).uniform_(alpha_min, alpha_max)

    outputs_base = model(I)
    x2 = torch.clamp(alphas * outputs_base["I_new"], 0.0, 1.0)

    scaled_target = torch.clamp(alphas * I, 0.0, 1.0)
    outputs_scaled = model(scaled_target)
    x3 = outputs_scaled["I_new"]

    loss = torch.mean((x2 - x3) ** 2)
    metrics = {
        "loss": float(loss.detach().cpu().item()),
        "alpha_mean": float(alphas.mean().detach().cpu().item()),
    }
    return loss, metrics


def find_latest_jorge_checkpoint(output_dir: Path) -> Path:
    candidates = []
    for path in output_dir.glob("*/checkpoints/best_model.pt"):
        run_name = path.parents[1].name
        if run_name.startswith("jorge_"):
            candidates.append(path)
    checkpoints = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
    if not checkpoints:
        raise FileNotFoundError("No se encontro ningun checkpoint previo de Jorge en outputs/jorge_*/checkpoints.")
    return checkpoints[0]


def run_scaling_equivariance_finetune(
    config: ExperimentConfig,
    checkpoint_path: Path | None = None,
    alpha_min: float = 0.85,
    alpha_max: float = 1.15,
) -> None:
    prepare_output_dirs(config)
    set_seed(config.seed)
    device = torch.device(config.device)

    train_loader, val_loader, report_loader = build_dataloaders(config)
    psf_tensor = build_psf_tensor(config, device)
    model = JorgeADMMUnrollingNet(config, psf_tensor).to(device)

    if checkpoint_path is None:
        checkpoint_path = find_latest_jorge_checkpoint(config.output_dir)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
        min_lr=config.min_learning_rate,
    )

    history: list[dict[str, float]] = []
    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        train_alpha = 0.0
        train_batches = 0

        for I in train_loader:
            I = I.to(device)
            loss, metrics = scaling_equivariance_loss(model, I, alpha_min, alpha_max)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += metrics["loss"]
            train_alpha += metrics["alpha_mean"]
            train_batches += 1

        model.eval()
        val_loss = 0.0
        val_alpha = 0.0
        val_batches = 0
        with torch.no_grad():
            for I in val_loader:
                I = I.to(device)
                loss, metrics = scaling_equivariance_loss(model, I, alpha_min, alpha_max)
                val_loss += metrics["loss"]
                val_alpha += metrics["alpha_mean"]
                val_batches += 1

        train_loss /= max(train_batches, 1)
        train_alpha /= max(train_batches, 1)
        val_loss /= max(val_batches, 1)
        val_alpha /= max(val_batches, 1)

        history.append(
            {
                "epoch": epoch,
                "train_loss_se": train_loss,
                "train_alpha_mean": train_alpha,
                "val_loss_se": val_loss,
                "val_alpha_mean": val_alpha,
            }
        )

        best_marker = ""
        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improvement = 0
            best_marker = " | mejora"
            torch.save(model.state_dict(), config.checkpoint_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        print(
            f"Etapa {epoch:03d}/{config.epochs:03d} | "
            f"val_se={val_loss:.6f} | "
            f"alpha={val_alpha:.3f}{best_marker}"
        )
        scheduler.step(val_loss)

        if epochs_without_improvement >= config.early_stopping_patience:
            print(
                f"Parada temprana: sin mejora en "
                f"{config.early_stopping_patience} epocas consecutivas."
            )
            break

    save_json(
        {
            "history": history,
            "source_checkpoint": str(checkpoint_path),
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
        },
        config.run_dir / "history.json",
    )
    torch.save(model.state_dict(), config.checkpoint_dir / "last_model.pt")
    best_model_path = config.checkpoint_dir / "best_model.pt"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    qualitative_sample = select_best_qualitative_sample(
        model,
        [train_loader, val_loader],
        config,
        device,
    )
    build_final_report(model, report_loader, qualitative_sample, config, device)
    print()
    print("Afinamiento SE finalizado.")
    print(f"Checkpoint base: {checkpoint_path}")
    print(f"Mejor modelo: {config.checkpoint_dir / 'best_model.pt'}")
    print(f"Carpeta de corrida: {config.run_dir}")
    print(f"Historial: {config.run_dir / 'history.json'}")
    print(f"Metricas: {config.report_dir / 'metrics.txt'}")
