from __future__ import annotations

import io
import warnings
from contextlib import redirect_stdout

import lpips
import torch
from torch import nn


class LPIPSMetric(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        warnings.filterwarnings(
            "ignore",
            message="The parameter 'pretrained' is deprecated since 0.13",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13",
            category=UserWarning,
        )
        with redirect_stdout(io.StringIO()):
            self.metric = lpips.LPIPS(net="alex", verbose=False)
        self.metric.eval()
        for param in self.metric.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x_rgb = x.repeat(1, 3, 1, 1)
            y_rgb = y.repeat(1, 3, 1, 1)
        else:
            x_rgb = x
            y_rgb = y
        x_rgb = x_rgb * 2.0 - 1.0
        y_rgb = y_rgb * 2.0 - 1.0
        return self.metric(x_rgb, y_rgb).mean()
