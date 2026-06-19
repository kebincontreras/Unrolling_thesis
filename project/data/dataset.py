from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        image_paths: list[Path],
        image_size: int,
        channel_mode: str = "rgb",
        horizontal_flip: bool = False,
    ) -> None:
        self.image_paths = image_paths
        self.image_size = image_size
        self.channel_mode = channel_mode
        self.horizontal_flip = horizontal_flip

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.image_paths[index]).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        if self.horizontal_flip and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        array = np.asarray(image, dtype=np.float32) / 255.0
        if self.channel_mode == "red":
            red = array[..., 0:1]
            tensor = torch.from_numpy(red).permute(2, 0, 1)
        else:
            tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor
