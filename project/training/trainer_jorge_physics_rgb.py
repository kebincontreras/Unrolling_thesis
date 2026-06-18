from __future__ import annotations

from dataclasses import asdict
from statistics import mean, pstdev

import torch

from project.config import ExperimentConfig
from project.data.datamodule import build_dataloaders
from project.losses.objective import ReconstructionObjective
from project.losses.perceptual import LPIPSMetric
from project.losses.ssim import psnr_value, ssim_index
from project.model.unrolling_jorge_physics_rgb import JorgePhysicsRGBADMMUnrollingNet, apply_filter
from project.physics.psf import build_rgb_psf_tensor
from project.physics.wiener import wiener_deconvolution
from project.training.visualization import save_metrics_table_image, save_qualitative_panel
from project.utils.io_utils import ensure_dir, save_json, set_seed


def prepare_output_dirs(config: ExperimentConfig) -> None:
    ensure_dir(config.output_dir)
    ensure_dir(config.run_dir)
    ensure_dir(config.checkpoint_dir)
    ensure_dir(config.sample_dir)
    ensure_dir(config.psf_dir)
    ensure_dir(config.report_dir)


def evaluate(
    model: JorgePhysicsRGBADMMUnrollingNet,
    loader: torch.utils.data.DataLoader,
    criterion: ReconstructionObjective,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    stats = {"loss": 0.0, "l1": 0.0, "ssim": 0.0, "psnr": 0.0}
    batches = 0
    with torch.no_grad():
        for I in loader:
            I = I.to(device)
            outputs = model(I)
            _, metrics = criterion(outputs["I_blur"], outputs["I"])
            for key in stats:
                stats[key] += metrics[key]
            batches += 1
    if batches == 0:
        return stats
    return {key: value / batches for key, value in stats.items()}


def summarize_metric(values: list[float]) -> str:
    if not values:
        return "0.000 +- 0.000"
    mu = mean(values)
    sigma = pstdev(values) if len(values) > 1 else 0.0
    return f"{mu:.3f} +- {sigma:.3f}"


def select_best_qualitative_sample(
    model: JorgePhysicsRGBADMMUnrollingNet,
    loaders: list[torch.utils.data.DataLoader],
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, torch.Tensor | float] | None:
    model.eval()
    best_sample = None
    best_psnr = float("-inf")

    with torch.no_grad():
        for loader in loaders:
            for I in loader:
                I = I.to(device)
                outputs = model(I)
                I_new = outputs["I_new"]
                I_blur = outputs["I_blur"]
                I_baseline = apply_filter(I, model.h)
                I_wiener_new = wiener_deconvolution(I, model.h, config.wiener_k)
                I_wiener = apply_filter(I_wiener_new, model.h)

                for idx in range(I.shape[0]):
                    target = I[idx:idx + 1]
                    ours = I_blur[idx:idx + 1]
                    ours_psnr = float(psnr_value(ours, target).cpu().item())
                    if ours_psnr > best_psnr:
                        best_psnr = ours_psnr
                        best_sample = {
                            "I": target.cpu(),
                            "I_corrected": ours.cpu(),
                            "I_wiener": I_wiener[idx:idx + 1].cpu(),
                            "I_new": I_new[idx:idx + 1].cpu(),
                            "I_baseline": I_baseline[idx:idx + 1].cpu(),
                            "ours_psnr": ours_psnr,
                            "wiener_psnr": float(psnr_value(I_wiener[idx:idx + 1], target).cpu().item()),
                            "base_psnr": float(psnr_value(I_baseline[idx:idx + 1], target).cpu().item()),
                        }
    return best_sample


def build_final_report(
    model: JorgePhysicsRGBADMMUnrollingNet,
    loader: torch.utils.data.DataLoader,
    qualitative_sample: dict[str, torch.Tensor | float] | None,
    config: ExperimentConfig,
    device: torch.device,
) -> None:
    model.eval()
    lpips_metric = LPIPSMetric().to(device)
    lpips_metric.eval()
    ours_ssim: list[float] = []
    ours_psnr: list[float] = []
    ours_lpips: list[float] = []
    wiener_ssim: list[float] = []
    wiener_psnr: list[float] = []
    wiener_lpips: list[float] = []
    base_ssim: list[float] = []
    base_psnr: list[float] = []
    base_lpips: list[float] = []

    with torch.no_grad():
        for I in loader:
            I = I.to(device)
            outputs = model(I)
            I_blur = outputs["I_blur"]
            I_baseline = apply_filter(I, model.h)
            I_wiener_new = wiener_deconvolution(I, model.h, config.wiener_k)
            I_wiener = apply_filter(I_wiener_new, model.h)

            for idx in range(I.shape[0]):
                target = I[idx:idx + 1]
                ours = I_blur[idx:idx + 1]
                base = I_baseline[idx:idx + 1]
                wiener = I_wiener[idx:idx + 1]

                ours_ssim.append(float(ssim_index(ours, target).cpu().item()))
                ours_psnr.append(float(psnr_value(ours, target).cpu().item()))
                ours_lpips.append(float(lpips_metric(ours, target).cpu().item()))
                base_ssim.append(float(ssim_index(base, target).cpu().item()))
                base_psnr.append(float(psnr_value(base, target).cpu().item()))
                base_lpips.append(float(lpips_metric(base, target).cpu().item()))
                wiener_ssim.append(float(ssim_index(wiener, target).cpu().item()))
                wiener_psnr.append(float(psnr_value(wiener, target).cpu().item()))
                wiener_lpips.append(float(lpips_metric(wiener, target).cpu().item()))

    rows = [
        ("Baseline (Uncorrected)", summarize_metric(base_ssim), summarize_metric(base_psnr), summarize_metric(base_lpips)),
        ("Wiener Deconvolution", summarize_metric(wiener_ssim), summarize_metric(wiener_psnr), summarize_metric(wiener_lpips)),
        ("Ours (Jorge-Physics-RGB)", summarize_metric(ours_ssim), summarize_metric(ours_psnr), summarize_metric(ours_lpips)),
    ]

    config_lines = ["Run parameters"]
    for key, value in asdict(config).items():
        config_lines.append(f"{key}: {value}")
    config_lines.extend(
        [
            "rgb_wavelengths_nm: [620, 550, 450]",
        ]
    )

    report_txt = [
        "Final metrics",
        "Model = Jorge-ADMM-Physics-Unrolling-RGB",
        f"Channel mode = {config.channel_mode}",
        f"Myopia severity = {config.myopia_severity:.2f}",
        f"Wiener balance k = {config.wiener_k:.6f}",
        "",
        f"Baseline SSIM: {summarize_metric(base_ssim)}",
        f"Baseline PSNR: {summarize_metric(base_psnr)} dB",
        f"Baseline LPIPS: {summarize_metric(base_lpips)}",
        "",
        f"Wiener SSIM: {summarize_metric(wiener_ssim)}",
        f"Wiener PSNR: {summarize_metric(wiener_psnr)} dB",
        f"Wiener LPIPS: {summarize_metric(wiener_lpips)}",
        "",
        f"Ours SSIM: {summarize_metric(ours_ssim)}",
        f"Ours PSNR: {summarize_metric(ours_psnr)} dB",
        f"Ours LPIPS: {summarize_metric(ours_lpips)}",
        "",
        *config_lines,
    ]
    ensure_dir(config.report_dir)
    (config.report_dir / "metrics.txt").write_text("\n".join(report_txt), encoding="utf-8")

    save_metrics_table_image(
        rows,
        config.report_dir / "metrics_table.png",
        severity_label=f"{config.myopia_severity:.2f}",
    )

    if qualitative_sample is not None:
        labels = {
            "Corrected": f"PSNR {float(qualitative_sample['ours_psnr']):.2f}",
            "Wiener": f"PSNR {float(qualitative_sample['wiener_psnr']):.2f}",
            "Myopia": f"PSNR {float(qualitative_sample['base_psnr']):.2f}",
        }
        save_qualitative_panel(
            qualitative_sample["I"],
            qualitative_sample["I_corrected"],
            qualitative_sample["I_wiener"],
            qualitative_sample["I_new"],
            qualitative_sample["I_baseline"],
            labels,
            config.report_dir / "qualitative_panel.png",
        )


def run_experiment_jorge_physics_rgb(config: ExperimentConfig) -> None:
    prepare_output_dirs(config)
    set_seed(config.seed)
    device = torch.device(config.device)

    train_loader, val_loader, report_loader = build_dataloaders(config)
    rgb_wavelengths = (620e-9, 550e-9, 450e-9)
    psf_tensor = build_rgb_psf_tensor(config, device, rgb_wavelengths)
    model = JorgePhysicsRGBADMMUnrollingNet(config, psf_tensor).to(device)
    criterion = ReconstructionObjective(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
        min_lr=config.min_learning_rate,
    )

    history: list[dict[str, float]] = []
    best_val_psnr = float("-inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        running = {"loss": 0.0, "l1": 0.0, "ssim": 0.0, "psnr": 0.0}
        batches = 0

        for I in train_loader:
            I = I.to(device)
            outputs = model(I)
            loss, metrics = criterion(outputs["I_blur"], outputs["I"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for key in running:
                running[key] += metrics[key]
            batches += 1

        train_metrics = {key: value / max(batches, 1) for key, value in running.items()}
        val_metrics = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_l1": train_metrics["l1"],
                "train_ssim": train_metrics["ssim"],
                "train_psnr": train_metrics["psnr"],
                "val_loss": val_metrics["loss"],
                "val_l1": val_metrics["l1"],
                "val_ssim": val_metrics["ssim"],
                "val_psnr": val_metrics["psnr"],
            }
        )

        best_marker = ""
        if val_metrics["psnr"] > best_val_psnr:
            best_marker = " | mejora"
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Etapa {epoch:03d}/{config.epochs:03d} | "
            f"val_ssim={val_metrics['ssim']:.4f} | "
            f"val_psnr={val_metrics['psnr']:.2f} | "
            f"val_loss={val_metrics['loss']:.4f}{best_marker}"
        )
        scheduler.step(val_metrics["psnr"])

        if val_metrics["psnr"] > best_val_psnr:
            best_val_psnr = val_metrics["psnr"]
            torch.save(model.state_dict(), config.checkpoint_dir / "best_model.pt")

        if epochs_without_improvement >= config.early_stopping_patience:
            print(
                f"Parada temprana: sin mejora en "
                f"{config.early_stopping_patience} epocas consecutivas."
            )
            break

    save_json({"history": history}, config.run_dir / "history.json")
    torch.save(model.state_dict(), config.checkpoint_dir / "last_model.pt")
    best_model_path = config.checkpoint_dir / "best_model.pt"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    qualitative_sample = select_best_qualitative_sample(model, [train_loader, val_loader], config, device)
    build_final_report(model, report_loader, qualitative_sample, config, device)
    print()
    print("Entrenamiento finalizado.")
    print(f"Mejor modelo: {config.checkpoint_dir / 'best_model.pt'}")
    print(f"Carpeta de corrida: {config.run_dir}")
    print(f"Historial: {config.run_dir / 'history.json'}")
    print(f"Metricas: {config.report_dir / 'metrics.txt'}")
