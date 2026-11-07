"""Non-graph speed prediction baselines.

Every model returns ``y``'s shape, so training and metrics treat them
interchangeably.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from .events import WEEK_SLOTS, Windows


def last_observed(x: Tensor, x_mask: Tensor) -> Tensor:
    """Most recent observed speed per road, ``[batch, roads]``; zero if never seen."""
    speed = x[..., 0]
    newest = x_mask.shape[1] - 1 - torch.flip(x_mask, [1]).float().argmax(dim=1)
    value = speed.gather(1, newest.unsqueeze(1)).squeeze(1)
    return torch.where(x_mask.any(dim=1), value, torch.zeros_like(value))


class Persistence:
    """Carry the most recent observed speed across the horizon."""

    def __init__(self, data: Windows):
        self.horizon = data.horizon

    def fit(self, data: Windows) -> Persistence:
        return self

    @torch.no_grad()
    def predict(self, batch: dict[str, Tensor]) -> Tensor:
        held = last_observed(batch["x"], batch["x_mask"])
        return held.unsqueeze(1).expand(-1, self.horizon, -1)


class HistoricalMean:
    """Mean speed for that road at that slot of the week."""

    def __init__(self, data: Windows):
        self.horizon, self.slot = data.horizon, data.slot
        self.table: Tensor | None = None

    def fit(self, data: Windows) -> HistoricalMean:
        values = data.values.numpy()
        steps = np.arange(1, self.horizon + 1)
        bins = np.unique((data.origins.numpy()[:, None] + steps).ravel())
        # Crash-free cells only: otherwise the no-accident table contains the accident.
        seen = np.isfinite(values[bins]) & data.clean.numpy()[bins]
        filled = np.where(seen, values[bins], 0.0)
        slots = data.slot.numpy()[bins]
        total = np.zeros((WEEK_SLOTS, values.shape[1]))
        count = np.zeros_like(total)
        np.add.at(total, slots, filled)
        np.add.at(count, slots, seen)
        # Slots never observed crash-free fall back to the overall crash-free mean.
        overall = filled.sum() / max(seen.sum(), 1)
        table = np.divide(total, count, out=np.full_like(total, overall), where=count > 0)
        self.table = torch.from_numpy(data.encode(table).astype("float32"))
        return self

    @torch.no_grad()
    def predict(self, batch: dict[str, Tensor]) -> Tensor:
        steps = torch.arange(1, self.horizon + 1)
        bins = batch["origin"].unsqueeze(1) + steps.unsqueeze(0)
        return self.table[self.slot[bins]]


BASELINES = {"persistence": Persistence, "historical_mean": HistoricalMean}


def build_model(name: str, data: Windows, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
    if name in BASELINES:
        return BASELINES[name](data)
    raise ValueError(f"unknown model {name!r}; choose from {(*BASELINES,)}")


def predict(model, batch: dict[str, Tensor], edge_index: Tensor) -> Tensor:
    if isinstance(model, nn.Module):
        return model(batch["x"], batch["x_mask"], edge_index)
    return model.predict(batch)
