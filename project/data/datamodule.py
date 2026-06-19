from __future__ import annotations

import math
import random
from pathlib import Path

from torch.utils.data import DataLoader

from project.config import ExperimentConfig
from project.data.dataset import ImageDataset


def list_image_paths(image_dir: Path) -> list[Path]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(image_dir.glob(pattern)))
    return sorted(paths)


def build_dataloaders(config: ExperimentConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    image_paths = list_image_paths(config.image_dir)
    validation_paths = list_image_paths(config.validation_dir) if config.validation_dir.exists() else []
    validation_set = {path.resolve() for path in validation_paths}
    image_paths = [path for path in image_paths if path.resolve() not in validation_set]

    if len(image_paths) < 2:
        raise ValueError(f"No hay suficientes imagenes en {config.image_dir}")
    if not validation_paths:
        raise ValueError(f"No hay imagenes en {config.validation_dir} para reportar metricas.")

    rng = random.Random(config.seed)
    shuffled = image_paths[:]
    rng.shuffle(shuffled)

    split_idx = max(1, min(len(shuffled) - 1, math.floor(len(shuffled) * config.train_split)))
    train_paths = shuffled[:split_idx]
    val_paths = shuffled[split_idx:]

    train_dataset = ImageDataset(
        train_paths,
        config.image_size,
        channel_mode=config.channel_mode,
        horizontal_flip=config.train_horizontal_flip,
    )
    val_dataset = ImageDataset(val_paths, config.image_size, channel_mode=config.channel_mode)
    report_dataset = ImageDataset(validation_paths, config.image_size, channel_mode=config.channel_mode)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    report_loader = DataLoader(
        report_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader, report_loader
