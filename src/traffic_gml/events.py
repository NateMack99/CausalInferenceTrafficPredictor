"""Model inputs and prediction targets: crash-free forecasting windows.

Table loading and speed alignment live in `data.py`, graph tensors in
`graph.py`; this module turns them into the windows the models are fed.

`docs/architecture.md` owns the batch shapes and the asymmetries behind them.
The rule it leaves to the code: a crash contaminates its own road and every
road within `exposure_hops` for `horizon` bins after it is reported, a road is
dropped from a window's *targets* if any contaminated cell falls in its lag or
target span, and contaminated cells are kept out of the historical-mean table
and the normalisation statistics. Fitting them would put the slowdown inside
the very counterfactual it is measured against.
"""

from __future__ import annotations

import copy
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .data import BIN_MINUTES, FREQ, TABLES, align_grid, load_speed_dataset, local_time
from .graph import build_edge_index, neighbourhood

BINS_PER_DAY = 24 * 60 // BIN_MINUTES
WEEK_SLOTS = 7 * BINS_PER_DAY
#: Per-bin feature channels, before any static road columns.
FEATURES = ("speed", "tod_sin", "tod_cos", "dow_sin", "dow_cos")


def processed_tables(
    speeds_dir: str | Path,
    segment_graphml: str | Path,
    crash_csv: str | Path,
    window: tuple[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Join the speed, crash and road-graph outputs into the model contract.

    `segment_graphml` and `crash_csv` are what `segments.analyze` writes beside
    the road graph. Three judgement calls, all of them narrowing:

    - only crashes with `match_status == "matched"` become events, since an
      unmatched crash has no road to attribute a slowdown to;
    - only crashes inside the speed window are kept, because a crash outside it
      cannot contaminate any window built from these speeds;
    - only segments with an observed speed in the window become nodes, so the
      dense `[bins, roads]` grid is not mostly empty columns. Edges to a dropped
      segment go with it, which removes paths that ran through unobserved road.
    """
    speeds = load_speed_dataset(speeds_dir, window)
    observed = speeds.loc[speeds["speed"].notna(), "segment_id"].unique()

    graph = nx.read_graphml(segment_graphml, node_type=int)
    keep = sorted(set(observed) & set(graph.nodes))
    if not keep:
        raise ValueError("no segment has both an observed speed and a node in the road graph")
    graph = graph.subgraph(keep)

    segments = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient="index")
    segments.index.name = "segment_id"
    segments = segments.reset_index()

    # The segment graph is undirected, so emit both directions: PyG message
    # passing follows edge_index as given, and one direction would starve half
    # the neighbourhood.
    both = {(int(u), int(v)) for u, v in graph.edges()}
    both |= {(v, u) for u, v in both}
    edges = pd.DataFrame(sorted(both), columns=["src", "dst"])

    crashes = pd.read_csv(crash_csv)
    crashes = crashes.loc[crashes["match_status"] == "matched"].copy()
    crashes["crash_time"] = pd.to_datetime(crashes["timestamp"])
    crashes = crashes.loc[crashes["segment_id"].notna()]
    crashes["segment_id"] = crashes["segment_id"].astype("int64")
    crashes = crashes.loc[crashes["segment_id"].isin(keep)]
    if window is not None:
        # Upper bound only: contamination runs forward, so a crash just before
        # the window still touches the first bins, and `crash_cells` clips any
        # whose wake lands entirely outside.
        crashes = crashes.loc[crashes["crash_time"] < pd.Timestamp(window[1])]

    tables = {
        "speeds": speeds.loc[speeds["segment_id"].isin(keep)].reset_index(drop=True),
        "crashes": crashes[list(TABLES["crashes"])].reset_index(drop=True),
        "segments": segments,
        "edges": edges,
    }
    for name, columns in TABLES.items():
        missing = [c for c in columns if c not in tables[name].columns]
        if missing:
            raise ValueError(f"joined {name} table is missing {missing}")
    return tables


def write_processed(tables: dict[str, pd.DataFrame], out: str | Path) -> Path:
    """Write the four contract tables so `--data` can point at them."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(out / f"{name}.parquet", index=False)
    return out


def event_datasets(
    reference: Windows,
    crashes: pd.DataFrame,
    segment_ids: np.ndarray,
    controls: int = 1,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, Windows, Windows]:
    """Before/after windows for each matched crash, plus accident-free controls.

    The event origin is the last bin before the crash, so `x` is entirely
    pre-crash and `y` spans the crash itself. Only the crash's own road is
    unmasked, so a window scores that road and nothing else.

    Both arms are held to the same standard, which is what makes the control a
    placebo rather than a flattering one:

    - a control is the same road at the same slot of the week in another week,
      crash-free across its whole span, **and never a window the model trained
      on**. Otherwise the control arm reports in-sample residuals against the
      event arm's out-of-sample ones, and part of the gap between them is just
      the model fitting its own training data.
    - an event's pre-crash lag must be crash-free too, so an earlier crash on
      the same road cannot depress the history the forecast is made from.

    Returns one row per crash, one row per (crash, control) pair, and the two
    window sets. The event set holds one window per crash: repeating it once
    per control would shrink the intervals without adding any information.
    """
    history, horizon = reference.history, reference.horizon
    values, clean = reference.values.numpy(), reference.clean.numpy()
    slot = reference.slot.numpy()
    n_bins = values.shape[0]
    times = reference.times
    if len(segment_ids) != reference.num_roads:
        raise ValueError("segment_ids does not match the grid these windows were built on")

    trained = np.zeros(values.shape, dtype=bool)
    trained[reference.origins.numpy()] = reference.usable.numpy()

    bins, roads, on_grid = crash_index(crashes, times, segment_ids)
    rng = np.random.default_rng(seed)

    events, pairs = [], []
    for record, crash_bin, road, known in zip(
        crashes["crash_record_id"], bins, roads, on_grid, strict=True
    ):
        origin = int(crash_bin) - 1
        if not known or origin - history + 1 < 0 or origin + horizon >= n_bins:
            continue
        lag = slice(origin - history + 1, origin + 1)
        if not clean[lag, road].all() or not np.isfinite(values[lag, road]).any():
            continue

        usable_controls = [
            int(t)
            for t in np.flatnonzero(slot == slot[origin])
            if abs(t - origin) >= history + horizon
            and t - history + 1 >= 0
            and t + horizon < n_bins
            and not trained[t, road]
            and clean[t - history + 1 : t + horizon + 1, road].all()
            and np.isfinite(values[t + 1 : t + 1 + horizon, road]).any()
        ]
        if not usable_controls:
            continue

        shared = {
            "crash_record_id": record,
            "segment_id": int(segment_ids[road]),
            "road": int(road),
        }
        events.append({**shared, "crash_time": times[int(crash_bin)], "origin": origin})
        chosen = rng.choice(
            usable_controls, size=min(controls, len(usable_controls)), replace=False
        )
        pairs += [{**shared, "origin": int(t)} for t in np.atleast_1d(chosen)]

    table, control_table = pd.DataFrame(events), pd.DataFrame(pairs)
    if table.empty:
        raise ValueError(
            "no crash has both a crash-free pre-crash history and a held-out control; "
            "widen the speed window or lower --history/--horizon"
        )

    def windows(frame: pd.DataFrame) -> Windows:
        only = np.zeros((len(frame), reference.num_roads), dtype=bool)
        only[np.arange(len(frame)), frame["road"].to_numpy()] = True
        return reference.at(frame["origin"].to_numpy(), only)

    return table, control_table, windows(table), windows(control_table)


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

    def at(self, origins: np.ndarray, usable: np.ndarray) -> Windows:
        """Same grid and statistics, different windows — for event/control sets."""
        other = copy.copy(self)
        other.origins = torch.from_numpy(np.asarray(origins, dtype=np.int64))
        other.usable = torch.from_numpy(np.asarray(usable, dtype=bool))
        return other

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


def crash_index(
    crashes: pd.DataFrame, times: pd.DatetimeIndex, segment_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each crash's (bin, road), and a mask of the ones that land on this grid.

    A crash on a segment that is not a road here has to be rejected, not snapped
    to a neighbour: `searchsorted` alone returns the insertion point, so segment
    25 against roads [10, 20, 30, 40] would come back as road 30 and the effect
    would be measured on a road that never had the crash. A `NaT` crash time
    must not become bin 0 either. A crash *before* the grid keeps its negative
    bin, so `crash_cells` can still mask the part of its wake that lands inside.
    """
    when = pd.to_datetime(crashes["crash_time"]).dt.floor(FREQ)
    dated = when.notna().to_numpy()
    bins = np.full(len(when), -1, dtype=np.int64)
    bins[dated] = (when[dated] - times[0]) // pd.Timedelta(FREQ)
    road_of = crashes["segment_id"].to_numpy()
    roads = np.minimum(np.searchsorted(segment_ids, road_of), len(segment_ids) - 1)
    on_grid = (segment_ids[roads] == road_of) & dated & (bins < len(times))
    return bins, roads, on_grid


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
    bins, roads, on_grid = crash_index(crashes, times, segment_ids)
    lasts = np.arange(horizon + 1)
    rows = (bins[on_grid][:, None] + lasts).ravel()
    cols = np.repeat(roads[on_grid], lasts.size)
    inside = (rows >= 0) & (rows < len(times))
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
