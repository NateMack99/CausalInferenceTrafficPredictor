"""Build one speed-observed road graph for independent crash and speed windows."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import pyarrow.dataset as ds
import scipy.sparse as sp
import torch
from pyproj import Transformer
from torch import Tensor

DATA = Path(__file__).resolve().parents[2] / "data"
OUTPUT = Path(__file__).resolve().parents[2] / "outputs"
EDGE_TYPES = {
    "segment_id": int,
    "coverage_fraction": float,
    "max_speed_mph": float,
    "speed_mph": float,
    "speed_count": int,
    "congestion": float,
    "crash_count": int,
}


def load_graph(path: str | Path) -> nx.MultiDiGraph:
    return ox.load_graphml(path, edge_dtypes=EDGE_TYPES)


def _window(
    path: Path, window: tuple[str | datetime, str | datetime]
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    start, end = map(pd.Timestamp, window)
    if pd.isna(start) or pd.isna(end) or start.tzinfo or end.tzinfo or start >= end:
        raise ValueError("Use a nonempty [start, end) window with timezone-naive timestamps")
    dataset = ds.dataset(path, format="parquet", partitioning="hive")
    selection = (
        (ds.field("date") >= start.date().isoformat())
        & (ds.field("date") <= end.date().isoformat())
        & (ds.field("timestamp") >= start.to_pydatetime())
        & (ds.field("timestamp") < end.to_pydatetime())
    )
    return dataset.to_table(filter=selection).to_pandas(), start, end


def build_graph(
    crash_window: tuple[str | datetime, str | datetime],
    speed_window: tuple[str | datetime, str | datetime],
    only_80_percent: bool = False,
    *,
    data_dir: str | Path = DATA,
    output_dir: str | Path = OUTPUT,
) -> nx.MultiDiGraph:
    """Save and return a road graph; windows are (start, end), with end excluded.

    Speeds are window means in mph. Congestion is 1-speed/historical maximum,
    clipped to [0,1], for visualization. Missing speeds remain NaN. The 80% flag
    uses full-reference-period bin availability, not availability in this window.
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    crashes, c0, c1 = _window(data_dir / "parquet/chicago-crashes", crash_window)
    if not {"latitude", "longitude"} <= set(crashes.columns):
        raise ValueError(
            "Crash data lacks coordinates. Run bash scripts/fetch_data.sh --crashes-only first."
        )
    speeds, s0, s1 = _window(data_dir / "parquet/chicago-speeds", speed_window)
    graph = load_graph(data_dir / "speed_road_network.graphml")
    if only_80_percent:
        graph = graph.edge_subgraph(
            [
                (u, v, k)
                for u, v, k, d in graph.edges(keys=True, data=True)
                if d["coverage_fraction"] >= 0.8
            ]
        ).copy()
    if not graph.number_of_edges():
        raise ValueError("No roads in the selected reference subset")
    means = speeds.groupby("segment_id").speed.agg(["mean", "count"])
    for _, _, _, edge in graph.edges(keys=True, data=True):
        sid = edge["segment_id"]
        speed = float(means.at[sid, "mean"]) if sid in means.index else float("nan")
        maximum = edge["max_speed_mph"]
        edge.update(
            speed_mph=speed,
            speed_count=int(means.at[sid, "count"]) if sid in means.index else 0,
            congestion=float(np.clip(1 - speed / maximum, 0, 1)) if maximum > 0 else float("nan"),
            crash_count=0,
            crash_times_json="[]",
        )

    audit = crashes.reset_index(drop=True).copy()
    lat, lon = (
        pd.to_numeric(audit.latitude, errors="coerce"),
        pd.to_numeric(audit.longitude, errors="coerce"),
    )
    valid = lat.between(-90, 90) & lon.between(-180, 180) & ~((lat == 0) & (lon == 0))
    audit["match_status"] = "invalid_coordinates"
    for field in ("edge_u", "edge_v", "edge_key"):
        audit[field] = pd.Series(pd.NA, index=audit.index, dtype="Int64")
    audit["distance_m"] = np.nan
    if valid.any():
        # Match to the FULL map before filtering: an unmonitored road's crash
        # must not be reassigned to the nearest monitored road instead.
        full = ox.project_graph(ox.load_graphml(data_dir / "chicago_road_network.graphml"))
        x, y = Transformer.from_crs("EPSG:4326", full.graph["crs"], always_xy=True).transform(
            lon[valid], lat[valid]
        )
        finite = np.isfinite(x) & np.isfinite(y)
        indexes = audit.index[valid][finite]
        if len(indexes):
            edges, distances = ox.distance.nearest_edges(
                full, np.asarray(x)[finite], np.asarray(y)[finite], return_dist=True
            )
            for row, edge, distance in zip(indexes, edges, distances, strict=True):
                edge = tuple(int(n) for n in edge)
                audit.loc[row, ["edge_u", "edge_v", "edge_key"]] = edge
                audit.at[row, "distance_m"] = float(distance)
                status = "too_far" if distance > 50 else "outside_subgraph"
                if distance <= 50 and graph.has_edge(*edge):
                    status = "matched"
                audit.at[row, "match_status"] = status
    matched = audit.loc[audit.match_status == "matched"]
    for edge, events in matched.groupby(["edge_u", "edge_v", "edge_key"]):
        graph.edges[tuple(int(n) for n in edge)].update(
            crash_count=len(events),
            crash_times_json=json.dumps([t.isoformat() for t in events.timestamp]),
        )
    created = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    subset = "80pct" if only_80_percent else "all"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"graph_{s0:%Y%m%dT%H%M%S}_{s1:%Y%m%dT%H%M%S}_{subset}_{created}.graphml"
    graph.graph.update(
        crash_start=c0.isoformat(),
        crash_end=c1.isoformat(),
        speed_start=s0.isoformat(),
        speed_end=s1.isoformat(),
        subset=subset,
        crashes_in_window=len(audit),
        matched_crashes=len(matched),
        crash_status_counts_json=json.dumps(audit.match_status.value_counts().to_dict()),
        crash_direction_resolved=False,
        output_path=str(path.resolve()),
    )
    ox.save_graphml(graph, filepath=path)
    audit.to_parquet(path.with_name(path.stem + "_crashes.parquet"), index=False)
    return graph


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
