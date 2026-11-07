"""Prediction metrics.

Speed-forecast MAE and RMSE over observed, crash-free targets, in mph and per
horizon step. Errors accumulate in running sums, so a year of roads never has
to fit in memory at once.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .events import BIN_MINUTES, Windows
from .models import predict


def masked_error(pred: Tensor, y: Tensor, mask: Tensor) -> Tensor:
    """Error at masked-in cells, zero elsewhere. Selected rather than multiplied:
    a NaN at a masked-out cell would otherwise poison the whole sum."""
    return torch.where(mask, pred - y, torch.zeros((), dtype=pred.dtype))


def masked_loss(pred: Tensor, y: Tensor, mask: Tensor, power: int = 1) -> Tensor:
    """Mean absolute (`power=1`) or squared (`power=2`) error over masked-in cells."""
    count = mask.sum()
    error = masked_error(pred, y, mask).abs().pow(power)
    return error.sum() / count if count else (pred * 0).sum()


#: --loss name -> exponent for `masked_loss`.
LOSSES = {"mae": 1, "mse": 2}


@torch.no_grad()
def evaluate(model, data: Windows, edge_index: Tensor, batch_size: int = 64) -> dict:
    """Forecast MAE and RMSE in mph, overall and per horizon step."""
    if isinstance(model, torch.nn.Module):
        model.eval()
    absolute, square, count = (torch.zeros(data.horizon, dtype=torch.float64) for _ in range(3))
    for batch in DataLoader(data, batch_size=batch_size):
        out, y = predict(model, batch, edge_index).double(), batch["y"].double()
        error = masked_error(out, y, batch["y_mask"]) * data.std
        absolute += error.abs().sum(dim=(0, 2))
        square += error.pow(2).sum(dim=(0, 2))
        count += batch["y_mask"].sum(dim=(0, 2))
    total = count.sum()
    if not total:
        raise ValueError(
            "no observed crash-free targets to evaluate on; lower --exposure-hops "
            "or check that the speeds table has observations on these roads"
        )
    # None, not 0.0, for a horizon step with no targets: they are not the same thing.
    per_step = (absolute / count.clamp(min=1)).tolist()
    return {
        "mae": float(absolute.sum() / total),
        "rmse": float((square.sum() / total).sqrt()),
        "mae_by_minutes": {
            str((step + 1) * BIN_MINUTES): round(value, 3) if count[step] else None
            for step, value in enumerate(per_step)
        },
        "n_targets": int(total),
    }
