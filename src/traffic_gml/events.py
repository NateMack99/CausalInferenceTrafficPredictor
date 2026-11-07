"""Model inputs and prediction targets: crash-free forecasting windows.

PR #4 owns the full accident and matched-control construction; this is the
crash-free slice PR #3 needs. Table loading, speed alignment and
`build_edge_index` belong in `data.py`/`graph.py` and live here only so PR #3
touches neither file while PR #1 and PR #2 are being written.

`docs/architecture.md` owns the batch shapes and the asymmetries behind them.
The rule it leaves to the code: a crash contaminates its own road and every
road within `exposure_hops` for `horizon` bins after it is reported, a road is
dropped from a window's *targets* if any contaminated cell falls in its lag or
target span, and contaminated cells are kept out of the historical-mean table
and the normalisation statistics. Fitting them would put the slowdown inside
the very counterfactual it is measured against.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import Tensor
from torch.utils.data import Dataset

FREQ = "10min"
BIN_MINUTES = 10
BINS_PER_DAY = 24 * 60 // BIN_MINUTES
WEEK_SLOTS = 7 * BINS_PER_DAY
#: Bins are stored in UTC per `docs/data.md`; rush hour is local.
LOCAL_TZ = "America/Chicago"
#: Per-bin feature channels, before any static road columns.
FEATURES = ("speed", "tod_sin", "tod_cos", "dow_sin", "dow_cos")
#: Required columns of the processed tables, per `docs/data.md`.
TABLES: dict[str, tuple[str, ...]] = {
    "speeds": ("segment_id", "time", "speed"),
    "crashes": ("crash_record_id", "crash_time", "segment_id"),
    "segments": ("segment_id",),
    "edges": ("src", "dst"),
}


def load_tables(path: str | Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for name, columns in TABLES.items():
        file = Path(path) / f"{name}.parquet"
        if not file.exists():
            raise FileNotFoundError(
                f"{file} not found. Point --data at data/processed, or write mock "
                f"tables with scripts/make_mock_data.py."
            )
        tables[name] = pd.read_parquet(file)
        missing = [c for c in columns if c not in tables[name].columns]
        if missing:
            raise ValueError(f"{file} is missing columns {missing}")
    return tables


def align_grid(
    speeds: pd.DataFrame, segment_ids: np.ndarray
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Speeds on a gap-free grid: `[bins, roads]` in mph, NaN where missing."""
    frame = speeds[["segment_id", "time", "speed"]].copy()
    frame["time"] = pd.to_datetime(frame["time"]).dt.floor(FREQ)
    frame["speed"] = frame["speed"].astype("float32").where(frame["speed"] >= 0)
    times = pd.date_range(frame["time"].min(), frame["time"].max(), freq=FREQ)
    wide = (
        frame.groupby(["time", "segment_id"])["speed"]
        .mean()
        .unstack("segment_id")
        .reindex(index=times, columns=segment_ids)
    )
    return wide.to_numpy("float32", copy=True), times


def build_edge_index(edges: pd.DataFrame, segment_ids: np.ndarray) -> Tensor:
    """`[2, edges]` in node-index order; unknown segments and self loops dropped."""
    node = {segment: index for index, segment in enumerate(segment_ids)}
    pairs = {
        (node[src], node[dst])
        for src, dst in zip(edges["src"], edges["dst"], strict=True)
        if src in node and dst in node and src != dst
    }
    if not pairs:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(sorted(pairs), dtype=torch.long).t().contiguous()


def neighbourhood(edge_index: Tensor, num_roads: int, hops: int) -> sp.csr_matrix:
    """Boolean `[roads, roads]` reachability within `hops` undirected hops.

    Undirected and inclusive of self: this decides which roads a crash
    contaminates, where over-reaching is the safe direction to err.
    """
    adjacency = sp.csr_matrix(
        (np.ones(edge_index.size(1), dtype=bool), (edge_index[0].numpy(), edge_index[1].numpy())),
        shape=(num_roads, num_roads),
    )
    both = (adjacency + adjacency.T).astype(bool)
    reach = sp.identity(num_roads, dtype=bool, format="csr")
    for _ in range(hops):
        reach = (reach + reach @ both).astype(bool)
    return reach


def local_time(times: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Bins in local time. Naive input is taken as UTC, per `docs/data.md`.

    Getting this wrong puts every clock feature hours out of phase, so the run
    summary prints the local span it inferred.
    """
    utc = times.tz_localize("UTC") if times.tz is None else times
    return utc.tz_convert(LOCAL_TZ)


def clock_features(times: pd.DatetimeIndex) -> np.ndarray:
    """Cyclic time-of-day and day-of-week in local time, `[bins, 4]`."""
    local = local_time(times)
    day = 2 * np.pi * (np.asarray(local.hour) * 60 + np.asarray(local.minute)) / 1440
    week = 2 * np.pi * np.asarray(local.day_of_week) / 7
    return np.stack([np.sin(day), np.cos(day), np.sin(week), np.cos(week)], 1).astype("float32")


def week_slot(times: pd.DatetimeIndex) -> np.ndarray:
    """Which of the week's 1008 ten-minute slots each bin falls in, local time."""
    local = local_time(times)
    per_hour = 60 // BIN_MINUTES
    into_day = np.asarray(local.hour) * per_hour + np.asarray(local.minute) // BIN_MINUTES
    return (np.asarray(local.day_of_week) * BINS_PER_DAY + into_day).astype("int64")


def encode_static(
    segments: pd.DataFrame, segment_ids: np.ndarray, columns: list[str] | tuple[str, ...]
) -> np.ndarray:
    """Standardised static road features, `[roads, features]`; empty when unused."""
    if not columns:
        return np.zeros((len(segment_ids), 0), dtype="float32")
    missing = [c for c in columns if c not in segments.columns]
    if missing:
        raise ValueError(f"segments table has no columns {missing}")
    table = segments.set_index("segment_id").reindex(segment_ids)[list(columns)]
    values = np.nan_to_num(table.to_numpy("float32"))
    spread = values.std(0)
    return ((values - values.mean(0)) / np.where(spread > 0, spread, 1.0)).astype("float32")


class Windows(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        times: pd.DatetimeIndex,
        static: np.ndarray,
        clean: np.ndarray,
        origins: np.ndarray,
        usable: np.ndarray,
        history: int,
        horizon: int,
        norm: tuple[float, float],
    ):
        self.values = torch.from_numpy(values)
        self.times = times
        self.clock = torch.from_numpy(clock_features(times))
        self.slot = torch.from_numpy(week_slot(times))
        self.static = torch.from_numpy(static)
        self.clean = torch.from_numpy(clean)
        self.origins = torch.from_numpy(origins)
        self.usable = torch.from_numpy(usable)
        self.history, self.horizon = history, horizon
        self.mean, self.std = norm
        self.num_roads = values.shape[1]
        self.num_features = len(FEATURES) + static.shape[1]

    def encode(self, speed):  # Tensor or ndarray, both used
        return (speed - self.mean) / self.std

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        t = int(self.origins[i])
        lags = self.values[t - self.history + 1 : t + 1]
        future = self.values[t + 1 : t + 1 + self.horizon]
        x_mask, y_seen = torch.isfinite(lags), torch.isfinite(future)
        clock = self.clock[t - self.history + 1 : t + 1].unsqueeze(1)
        return {
            "x": torch.cat(
                [
                    torch.where(x_mask, self.encode(lags), 0.0).unsqueeze(-1),
                    clock.expand(self.history, self.num_roads, clock.shape[-1]),
                    self.static.expand(self.history, *self.static.shape),
                ],
                dim=-1,
            ),
            "x_mask": x_mask,
            "y": torch.where(y_seen, self.encode(future), 0.0),
            "y_mask": y_seen & self.usable[i].unsqueeze(0),
            "origin": self.origins[i],
        }


def _span_sum(cumulative: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Inclusive `low..high` sums down the bin axis, vectorised over origins."""
    before = np.where((low > 0)[:, None], cumulative[np.maximum(low - 1, 0)], 0)
    return cumulative[high] - before


def crash_free_roads(
    values: np.ndarray, dirty: np.ndarray, history: int, horizon: int, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    """Window origins, and the `[windows, roads]` mask of usable roads."""
    origins = np.arange(history - 1, values.shape[0] - horizon, stride, dtype=np.int64)
    if origins.size == 0:
        raise ValueError("speed history is shorter than one window")
    crashed = np.cumsum(dirty, axis=0, dtype=np.int32)
    observed = np.cumsum(np.isfinite(values), axis=0, dtype=np.int32)
    usable = (_span_sum(crashed, origins - history + 1, origins + horizon) == 0) & (
        _span_sum(observed, origins + 1, origins + horizon) > 0
    )
    keep = usable.any(axis=1)
    return origins[keep], usable[keep]


def split_windows(count: int, fracs: tuple[float, float, float], gap: int) -> dict[str, np.ndarray]:
    """Chronological split, dropping `gap` *windows* at each boundary.

    `gap` is in windows, not bins, so the caller converts: a window every
    `stride` bins needs `ceil((history + horizon) / stride)` of them to cover
    the span that would otherwise straddle the boundary.
    """
    if not np.isclose(sum(fracs), 1.0):
        raise ValueError(f"split fractions must sum to 1, got {fracs}")
    train_end = int(count * fracs[0])
    val_end = train_end + int(count * fracs[1])
    splits = {
        "train": np.arange(0, train_end - gap),
        "val": np.arange(train_end, val_end - gap),
        "test": np.arange(val_end, count),
    }
    if any(index.size == 0 for index in splits.values()):
        raise ValueError(
            f"{count} windows cannot fill three splits with a gap of {gap}; "
            f"lower --history/--horizon/--stride or use a longer speed history"
        )
    return splits


def crash_cells(
    crashes: pd.DataFrame,
    times: pd.DatetimeIndex,
    segment_ids: np.ndarray,
    edge_index: Tensor,
    hops: int,
    horizon: int,
) -> np.ndarray:
    """`[bins, roads]` bool: cells a crash contaminates, spread `hops` across the
    graph and `horizon` bins forward. Rows with no usable time or segment are
    ignored -- a `NaT` must not become a crash at bin 0.
    """
    when = pd.to_datetime(crashes["crash_time"]).dt.floor(FREQ)
    dated = when.notna().to_numpy()
    bins = np.full(len(when), -1, dtype=np.int64)
    bins[dated] = (when[dated] - times[0]) // pd.Timedelta(FREQ)
    road_of = crashes["segment_id"].to_numpy()
    roads = np.minimum(np.searchsorted(segment_ids, road_of), len(segment_ids) - 1)
    on_grid = (segment_ids[roads] == road_of) & (bins >= 0) & (bins < len(times))

    lasts = np.arange(horizon + 1)
    rows = (bins[on_grid][:, None] + lasts).ravel()
    cols = np.repeat(roads[on_grid], lasts.size)
    inside = rows < len(times)
    hit = sp.csr_matrix(
        (np.ones(int(inside.sum()), dtype=bool), (rows[inside], cols[inside])),
        shape=(len(times), len(segment_ids)),
    )
    return (hit @ neighbourhood(edge_index, len(segment_ids), hops)).toarray()


def build_windows(
    tables: dict[str, pd.DataFrame],
    history: int = 12,
    horizon: int = 6,
    exposure_hops: int = 2,
    stride: int = 1,
    static_features: list[str] | tuple[str, ...] = (),
) -> tuple[dict[str, Windows], Tensor]:
    """Crash-free train/validation/test window sets, and the road `edge_index`."""
    segment_ids = np.sort(tables["segments"]["segment_id"].unique())
    values, times = align_grid(tables["speeds"], segment_ids)
    edges = build_edge_index(tables["edges"], segment_ids)
    if len(tables["edges"]) and not edges.size(1):
        raise ValueError(
            f"none of the {len(tables['edges'])} edges name two known segments; "
            "check that edges.src/dst use the same ids and dtype as segments.segment_id"
        )

    dirty = crash_cells(tables["crashes"], times, segment_ids, edges, exposure_hops, horizon)
    clean = ~dirty
    origins, usable = crash_free_roads(values, dirty, history, horizon, stride)
    splits = split_windows(len(origins), (0.7, 0.15, 0.15), -(-(history + horizon) // stride))

    # Normalise on crash-free cells of the training split only: later bins would leak,
    # and crashed cells would make the fill value for a missing speed crash-depressed.
    cut = int(origins[splits["train"]][-1]) + 1
    seen = values[:cut][clean[:cut]]
    seen = seen[np.isfinite(seen)]
    if not seen.size:
        raise ValueError("no crash-free observed speeds in the training split to normalise with")
    spread = float(seen.std())
    norm = (float(seen.mean()), spread if spread > 0 else 1.0)

    static = encode_static(tables["segments"], segment_ids, static_features)
    sets = {
        name: Windows(
            values, times, static, clean, origins[index], usable[index], history, horizon, norm
        )
        for name, index in splits.items()
    }
    return sets, edges
