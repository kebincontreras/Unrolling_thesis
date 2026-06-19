from __future__ import annotations

import csv
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


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
        "(torch, numpy, matplotlib, lpips, torchvision)."
    )


_bootstrap_compatible_python()

from project.config import ExperimentConfig
from project.training.trainer_jorge_physics_rgb import run_experiment_jorge_physics_rgb
from project.utils.io_utils import ensure_dir


BASE_EPOCHS = 1000
RUN_PREFIX = "jorge_physics_rgb_tuning"

# Edita estas listas para ampliar o reducir el tuning.
# El numero total de experimentos es el producto de todos los tamanos.
SEARCH_SPACE: dict[str, list[Any]] = {
    "learning_rate": [5e-4, 1e-4],
    "k_stages": [6, 8],
    "hidden_channels": [24, 32],
}

BASE_CONFIG: dict[str, Any] = {
    "channel_mode": "rgb",
    "batch_size": 4,
    "image_size": 256,
    "train_split": 0.8,
    "device": "cuda",
    "early_stopping_patience": 20,
    "rho_init": 0.25,
    "lambda_ssim": 0.3,
    "lambda_psnr": 0.01,
    "myopia_severity": 0.5,
    "psf_grid_size": 256,
    "psf_kernel_size": 41,
    "pupil_radius": 1.5e-3,
    "pupil_extent": 4.0e-3,
    "psf_output_extent": 40e-6,
    "psf_output_n": 121,
    "wavelength": 550e-9,
    "propagation_distance": 24e-3,
    "defocus_plane_shift": 0.0,
    "seed": 7,
}


def iter_trials() -> list[dict[str, Any]]:
    keys = list(SEARCH_SPACE.keys())
    values = [SEARCH_SPACE[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def parse_metrics(metrics_path: Path) -> dict[str, float]:
    text = metrics_path.read_text(encoding="utf-8")
    patterns = {
        "baseline_ssim_mean": r"Baseline SSIM: ([0-9.]+) \+\- ([0-9.]+)",
        "baseline_psnr_mean": r"Baseline PSNR: ([0-9.]+) \+\- ([0-9.]+) dB",
        "baseline_lpips_mean": r"Baseline LPIPS: ([0-9.]+) \+\- ([0-9.]+)",
        "wiener_ssim_mean": r"Wiener SSIM: ([0-9.]+) \+\- ([0-9.]+)",
        "wiener_psnr_mean": r"Wiener PSNR: ([0-9.]+) \+\- ([0-9.]+) dB",
        "wiener_lpips_mean": r"Wiener LPIPS: ([0-9.]+) \+\- ([0-9.]+)",
        "ours_ssim_mean": r"Ours SSIM: ([0-9.]+) \+\- ([0-9.]+)",
        "ours_psnr_mean": r"Ours PSNR: ([0-9.]+) \+\- ([0-9.]+) dB",
        "ours_lpips_mean": r"Ours LPIPS: ([0-9.]+) \+\- ([0-9.]+)",
    }

    parsed: dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match is None:
            raise ValueError(f"No pude leer la metrica '{key}' en {metrics_path}")
        parsed[key] = float(match.group(1))
        parsed[key.replace("_mean", "_std")] = float(match.group(2))
    return parsed


def load_best_validation_metrics(history_path: Path) -> dict[str, float]:
    history = json.loads(history_path.read_text(encoding="utf-8"))["history"]
    if not history:
        return {}
    best_row = max(history, key=lambda row: row["val_psnr"])
    return {
        "best_epoch": int(best_row["epoch"]),
        "best_val_loss": float(best_row["val_loss"]),
        "best_val_l1": float(best_row["val_l1"]),
        "best_val_ssim": float(best_row["val_ssim"]),
        "best_val_psnr": float(best_row["val_psnr"]),
    }


def save_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    trials = iter_trials()
    summary_dir = Path("outputs") / "hyperparameter_tuning_rgb"
    ensure_dir(summary_dir)

    print(f"Total de experimentos: {len(trials)}")
    print(f"Resumen global: {summary_dir}")
    print()

    results: list[dict[str, Any]] = []

    for trial_index, overrides in enumerate(trials):
        epoch_id = BASE_EPOCHS + trial_index
        config = ExperimentConfig(
            epochs=epoch_id,
            run_prefix=RUN_PREFIX,
            **BASE_CONFIG,
            **overrides,
        )

        print("=" * 80)
        print(
            f"Experimento {trial_index + 1}/{len(trials)} | "
            f"epochs={epoch_id} | "
            f"params={overrides}"
        )
        print(f"Salida: {config.run_dir}")
        print("=" * 80)

        run_experiment_jorge_physics_rgb(config)

        metrics = parse_metrics(config.report_dir / "metrics.txt")
        best_val = load_best_validation_metrics(config.run_dir / "history.json")
        row = {
            "trial_index": trial_index,
            "epochs": epoch_id,
            "run_dir": str(config.run_dir),
            **overrides,
            **best_val,
            **metrics,
        }
        results.append(row)

        results_sorted = sorted(results, key=lambda item: item["ours_psnr_mean"], reverse=True)
        save_summary_csv(results_sorted, summary_dir / "summary.csv")
        (summary_dir / "summary.json").write_text(json.dumps(results_sorted, indent=2), encoding="utf-8")

    leaderboard = sorted(results, key=lambda item: item["ours_psnr_mean"], reverse=True)
    lines = ["Ranking final por Ours PSNR", ""]
    for rank, row in enumerate(leaderboard, start=1):
        lines.append(
            f"{rank:02d}. PSNR={row['ours_psnr_mean']:.3f} | "
            f"SSIM={row['ours_ssim_mean']:.3f} | "
            f"LPIPS={row['ours_lpips_mean']:.3f} | "
            f"epochs={row['epochs']} | "
            f"params="
            f"learning_rate={row.get('learning_rate')}, "
            f"k_stages={row.get('k_stages')}, "
            f"hidden_channels={row.get('hidden_channels')} | "
            f"run_dir={row['run_dir']}"
        )
    (summary_dir / "leaderboard.txt").write_text("\n".join(lines), encoding="utf-8")

    print()
    print("Tuning finalizado.")
    print(f"CSV: {summary_dir / 'summary.csv'}")
    print(f"JSON: {summary_dir / 'summary.json'}")
    print(f"Ranking: {summary_dir / 'leaderboard.txt'}")


if __name__ == "__main__":
    main()
