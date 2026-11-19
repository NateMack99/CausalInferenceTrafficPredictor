"""Non-graph baselines and basic graph forecasters.

Every model returns ``y``'s shape, so training and metrics treat them
interchangeably.

``mlp`` has ``gcn``'s trunk and parameter shapes with no edges, so that pair
differs only by the graph and not by capacity. ``sage`` has twice the
per-layer parameters (separate root and neighbour weights), so read the graph
effect off ``gcn`` against ``mlp``, not off ``sage``.

``stgcn`` adds a GRU down the history, and ``tgru`` is ``stgcn`` without
edges; that is the matched pair for the spatiotemporal comparison.

PyTorch Geometric convolutions act on the second-to-last dimension, so a whole
batch of window snapshots shares one ``edge_index`` and runs in a single call.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric.nn import GCNConv, SAGEConv

from .events import WEEK_SLOTS, Windows


def resolve_device(choice: str = "cpu") -> torch.device:
    """`auto` picks cuda, then mps, then cpu."""
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def on_device(batch: dict, device: torch.device) -> dict:
    """Move a window batch; `origin` stays put since only indexing uses it."""
    return {k: v.to(device) if k != "origin" else v for k, v in batch.items()}


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


class _NoGraph(nn.Linear):
    """Stand-in for a convolution that ignores the edges."""

    def forward(self, x: Tensor, edge_index: Tensor | None = None) -> Tensor:
        return super().forward(x)


class Forecaster(nn.Module):
    """Per-road encoder over the flattened history, message passing, horizon head."""

    def __init__(
        self,
        data: Windows,
        hidden: int = 64,
        layers: int = 2,
        conv: type[nn.Module] = GCNConv,
        dropout: float = 0.1,
    ):
        super().__init__()
        # +1 channel for the observation mask, so a gap is explicit to the model.
        self.encoder = nn.Linear(data.history * (data.num_features + 1), hidden)
        self.layers = nn.ModuleList(conv(hidden, hidden) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, data.horizon)

    def forward(self, x: Tensor, x_mask: Tensor, edge_index: Tensor) -> Tensor:
        batch, history, roads, _ = x.shape
        features = torch.cat([x, x_mask.unsqueeze(-1).to(x.dtype)], dim=-1)
        flat = features.permute(0, 2, 1, 3).reshape(batch, roads, -1)
        h = torch.relu(self.encoder(flat))
        for layer in self.layers:
            h = h + self.dropout(torch.relu(layer(h, edge_index)))
        return self.head(h).permute(0, 2, 1)


class SpatioTemporal(nn.Module):
    """Per-bin encoder, a GRU down the history, message passing, horizon head.

    `Forecaster` flattens `history` into per-road features, so time only exists
    in the weight layout and the model cannot share what it learns about a
    trend across offsets. This keeps the time axis and runs one GRU down it,
    then passes messages on the final hidden state.

    `tgru` is this same model with the edges removed, so that pair isolates the
    graph at matched capacity exactly as `mlp` does for `gcn`. Read the
    spatiotemporal graph effect off `stgcn` against `tgru`, never against
    `mlp`, which differs in both the edges and the temporal encoder.
    """

    def __init__(
        self,
        data: Windows,
        hidden: int = 64,
        layers: int = 2,
        conv: type[nn.Module] = GCNConv,
        dropout: float = 0.1,
    ):
        super().__init__()
        # +1 channel for the observation mask, as in `Forecaster`.
        self.encoder = nn.Linear(data.num_features + 1, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.layers = nn.ModuleList(conv(hidden, hidden) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, data.horizon)

    def forward(self, x: Tensor, x_mask: Tensor, edge_index: Tensor) -> Tensor:
        batch, history, roads, _ = x.shape
        features = torch.cat([x, x_mask.unsqueeze(-1).to(x.dtype)], dim=-1)
        # The GRU wants [sequences, time, channels], so roads fold into the batch
        # and every road shares one recurrent cell, as the conv layers share weights.
        flat = features.permute(0, 2, 1, 3).reshape(batch * roads, history, -1)
        h = torch.relu(self.encoder(flat))
        h = self.gru(h)[0][:, -1].reshape(batch, roads, -1)
        for layer in self.layers:
            h = h + self.dropout(torch.relu(layer(h, edge_index)))
        return self.head(h).permute(0, 2, 1)


BASELINES = {"persistence": Persistence, "historical_mean": HistoricalMean}
CONVS = {"mlp": _NoGraph, "gcn": GCNConv, "sage": SAGEConv}
#: Spatiotemporal pair: `tgru` is `stgcn` without edges, at matched capacity.
TEMPORAL = {"tgru": _NoGraph, "stgcn": GCNConv}


def build_model(name: str, data: Windows, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
    if name in BASELINES:
        return BASELINES[name](data)
    if name in CONVS:
        return Forecaster(data, hidden, layers, CONVS[name], dropout)
    if name in TEMPORAL:
        return SpatioTemporal(data, hidden, layers, TEMPORAL[name], dropout)
    raise ValueError(f"unknown model {name!r}; choose from {(*BASELINES, *CONVS, *TEMPORAL)}")


def predict(model, batch: dict[str, Tensor], edge_index: Tensor) -> Tensor:
    if isinstance(model, nn.Module):
        return model(batch["x"], batch["x_mask"], edge_index)
    return model.predict(batch)
