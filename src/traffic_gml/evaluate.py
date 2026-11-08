"""Prediction metrics.

Speed-forecast MAE and RMSE over observed, crash-free targets, in mph and per
horizon step. Errors accumulate in running sums, so a year of roads never has
to fit in memory at once.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import Tensor

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  -- after the Agg backend is set
from torch.utils.data import DataLoader

from .data import BIN_MINUTES
from .events import Windows
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


@torch.no_grad()
def effects(model, data: Windows, edge_index: Tensor, batch_size: int = 64) -> pd.DataFrame:
    """Observed minus predicted no-accident speed, per window, road and step.

    Negative means slower than the counterfactual. One row per unmasked target,
    with `effect` and `observed` in mph. Forecast accuracy alone does not make
    this causal: `docs/methodology.md` sets out what else has to hold.
    """
    if isinstance(model, torch.nn.Module):
        model.eval()
    frames = []
    for batch in DataLoader(data, batch_size=batch_size):
        prediction = predict(model, batch, edge_index)
        mask = batch["y_mask"]
        where = mask.nonzero()
        frames.append(
            pd.DataFrame(
                {
                    "origin": batch["origin"][where[:, 0]].numpy(),
                    "minutes": (where[:, 1].numpy() + 1) * BIN_MINUTES,
                    "road": where[:, 2].numpy(),
                    "observed": (batch["y"] * data.std + data.mean)[mask].numpy(),
                    "predicted": (prediction * data.std + data.mean)[mask].numpy(),
                    "effect": ((batch["y"] - prediction) * data.std)[mask].numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


#: Validated categorical slots (light surface); markers carry identity too, so
#: the three low-contrast hues are never colour-alone. See the dataviz palette.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
MARKERS = ("o", "s", "^", "D", "v")
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def _axes(width: float = 7.0, height: float = 4.0, columns: int = 1):
    figure, axes = plt.subplots(1, columns, figsize=(width, height), facecolor=SURFACE)
    for axis in np.atleast_1d(axes):
        axis.set_facecolor(SURFACE)
        axis.grid(True, color="#e6e5e1", linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color("#d6d5d0")
        axis.tick_params(colors=MUTED, labelsize=9)
    return figure, axes


def _save(figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    return path


def coverage_figure(data: Windows, path: str | Path) -> Path:
    """Speed coverage over the study period, and across roads."""
    observed = np.isfinite(data.values.numpy())
    days = pd.Series(observed.mean(axis=1), index=data.times).resample("D").mean()
    figure, (left, right) = _axes(width=10.0, columns=2)
    left.plot(days.index, days.to_numpy(), color=SERIES[0], linewidth=2)
    left.set_title("Observed speed bins per day", color=INK, fontsize=11, loc="left")
    left.set_ylabel("fraction observed", color=MUTED, fontsize=9)
    left.set_ylim(0, 1)
    figure.autofmt_xdate()
    right.hist(observed.mean(axis=0), bins=24, color=SERIES[0], edgecolor=SURFACE)
    right.set_title("Roads by coverage", color=INK, fontsize=11, loc="left")
    right.set_xlabel("fraction of bins observed", color=MUTED, fontsize=9)
    right.set_ylabel("roads", color=MUTED, fontsize=9)
    return _save(figure, path)


def comparison_figure(metrics: dict[str, dict], path: str | Path) -> Path:
    """Forecast MAE by horizon, one line per model."""
    figure, axis = _axes()
    for i, (name, values) in enumerate(metrics.items()):
        steps = {int(k): v for k, v in values["mae_by_minutes"].items() if v is not None}
        axis.plot(
            list(steps),
            list(steps.values()),
            label=name,
            color=SERIES[i % len(SERIES)],
            marker=MARKERS[i % len(MARKERS)],
            markersize=5,
            linewidth=2,
        )
    axis.set_title("No-accident forecast error by horizon", color=INK, fontsize=11, loc="left")
    axis.set_xlabel("minutes ahead", color=MUTED, fontsize=9)
    axis.set_ylabel("MAE (mph)", color=MUTED, fontsize=9)
    axis.legend(frameon=False, labelcolor=INK, fontsize=9)
    return _save(figure, path)


def effect_figure(event: pd.DataFrame, control: pd.DataFrame, path: str | Path) -> Path:
    """Observed minus predicted speed after a crash, against matched controls."""
    figure, axis = _axes()
    axis.axhline(0, color="#b9b8b2", linewidth=1)
    for i, (name, frame) in enumerate((("crash windows", event), ("matched controls", control))):
        by_step = frame.groupby("minutes").effect.agg(["mean", "sem"])
        axis.errorbar(
            by_step.index,
            by_step["mean"],
            yerr=1.96 * by_step["sem"].fillna(0),
            label=name,
            color=SERIES[i],
            marker=MARKERS[i],
            markersize=5,
            linewidth=2,
            capsize=3,
        )
    axis.set_title(
        "Speed effect after a crash, with 95% intervals", color=INK, fontsize=11, loc="left"
    )
    axis.set_xlabel("minutes after the crash", color=MUTED, fontsize=9)
    axis.set_ylabel("observed − predicted (mph)", color=MUTED, fontsize=9)
    axis.legend(frameon=False, labelcolor=INK, fontsize=9)
    return _save(figure, path)
