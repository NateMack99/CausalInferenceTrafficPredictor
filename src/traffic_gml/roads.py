"""Build the reusable speed-observed road subgraph after fetching data."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import pyarrow.dataset as ds
from shapely.geometry import LineString
from shapely.strtree import STRtree

DATA = Path(__file__).resolve().parents[2] / "data"


def _street(name: str) -> str:
    words = re.sub(r"[^a-z0-9 ]", " ", name.lower()).split()
    if words and words[0] in {"north", "south", "east", "west", "n", "s", "e", "w"}:
        words = words[1:]
    if words and words[-1] in {
        "avenue",
        "ave",
        "street",
        "st",
        "road",
        "rd",
        "boulevard",
        "blvd",
        "drive",
        "dr",
        "parkway",
        "pkwy",
        "court",
        "ct",
        "place",
        "pl",
    }:
        words = words[:-1]
    return " ".join(words)


def prepare_roads(data_dir: str | Path = DATA) -> nx.MultiDiGraph:
    """Save speed_segments.parquet, speed_roads.parquet, and the starter GraphML."""
    data_dir = Path(data_dir)
    source = data_dir / "parquet/chicago-speeds"
    manifest = json.loads((source / "_conversion.json").read_text())
    days = (
        pd.Timestamp(manifest["last_timestamp"]).normalize()
        - pd.Timestamp(manifest["first_timestamp"]).normalize()
    ).days + 1
    bins, maxima, counts = defaultdict(int), defaultdict(float), defaultdict(int)
    dataset = ds.dataset(source, format="parquet", partitioning="hive")
    for batch in dataset.to_batches(
        columns=["timestamp", "segment_id", "speed"],
        filter=ds.field("speed_available"),
        batch_size=500_000,
    ):
        frame = batch.to_pandas()
        frame["day"] = frame.timestamp.dt.normalize()
        frame["slot"] = frame.timestamp.dt.hour * 6 + frame.timestamp.dt.minute // 10
        for key, slots in frame.groupby(["segment_id", "day"]).slot.unique().items():
            bins[key] |= sum(1 << int(slot) for slot in slots)
        for sid, group in frame.groupby("segment_id"):
            maxima[int(sid)] = max(maxima[int(sid)], float(group.speed.max()))
            counts[int(sid)] += len(group)
    covered = defaultdict(int)
    for (sid, _), mask in bins.items():
        covered[int(sid)] += mask.bit_count()
    reference = pd.DataFrame(
        [
            {
                "segment_id": sid,
                "valid_speed_count": counts[sid],
                "max_speed_mph": maxima[sid],
                "coverage_fraction": covered[sid] / (days * 144),
            }
            for sid in sorted(counts)
        ]
    )
    if reference.empty:
        raise ValueError("No valid speeds in the Parquet dataset")
    metadata = pd.DataFrame(json.loads((data_dir / "chicago-traffic-segments.json").read_text()))
    metadata["segment_id"] = metadata.segmentid.astype(int)
    reference = reference.merge(
        metadata[["segment_id", "street", "_direction", "_fromst", "_tost"]],
        on="segment_id",
        how="left",
        validate="one_to_one",
    ).rename(columns={"_direction": "direction", "_fromst": "from_street", "_tost": "to_street"})
    graph = ox.load_graphml(data_dir / "chicago_road_network.graphml")
    projected = ox.project_graph(graph)
    metadata = metadata.loc[metadata.segment_id.isin(reference.segment_id)].copy()
    metadata["geometry"] = [
        LineString(
            [
                (float(row.start_lon), float(row._lif_lat)),
                (float(row._lit_lon), float(row._lit_lat)),
            ]
        )
        for _, row in metadata.iterrows()
    ]
    segments = gpd.GeoDataFrame(metadata, geometry="geometry", crs="EPSG:4326").to_crs(
        projected.graph["crs"]
    )
    lines = list(segments.geometry)
    tree = STRtree(lines)
    matches = []
    for (u, v, key), edge in ox.graph_to_gdfs(projected, nodes=False).iterrows():
        name = edge.get("name", "")
        names = {
            _street(n) for n in (name if isinstance(name, list) else [name]) if isinstance(n, str)
        } - {""}
        vector = np.array(
            [
                projected.nodes[v]["x"] - projected.nodes[u]["x"],
                projected.nodes[v]["y"] - projected.nodes[u]["y"],
            ]
        )
        points = [edge.geometry.interpolate(f, normalized=True) for f in np.linspace(0.1, 0.9, 5)]
        candidates = []
        for i in tree.query(edge.geometry.buffer(40)):
            line = lines[i]
            direction = np.array(line.coords[-1]) - np.array(line.coords[0])
            scale = np.linalg.norm(vector) * np.linalg.norm(direction)
            if not scale or np.dot(vector, direction) / scale < np.cos(np.pi / 4):
                continue
            if _street(segments.iloc[i].street) not in names:
                continue
            distances = np.array([line.distance(point) for point in points])
            support = float((distances <= 40).mean())
            if support >= 0.6:
                candidates.append(
                    (-support, float(distances.mean()), int(segments.iloc[i].segment_id))
                )
        if candidates:
            _, offset, sid = min(candidates)
            matches.append(
                {
                    "edge_u": u,
                    "edge_v": v,
                    "edge_key": key,
                    "segment_id": sid,
                    "match_distance_m": offset,
                }
            )
    roads = pd.DataFrame(matches).merge(reference, on="segment_id", validate="many_to_one")
    starter = graph.edge_subgraph(
        [(r.edge_u, r.edge_v, r.edge_key) for r in roads.itertuples()]
    ).copy()
    for row in roads.itertuples():
        starter.edges[row.edge_u, row.edge_v, row.edge_key].update(
            segment_id=int(row.segment_id),
            coverage_fraction=float(row.coverage_fraction),
            max_speed_mph=float(row.max_speed_mph),
        )
    starter.graph["speed_reference_period"] = (
        f"{manifest['first_timestamp']}/{manifest['last_timestamp']}"
    )
    reference.to_parquet(data_dir / "speed_segments.parquet", index=False)
    reference.loc[reference.coverage_fraction >= 0.8].to_csv(
        data_dir / "dense_80pct_segments.csv", index=False
    )
    roads.to_parquet(data_dir / "speed_roads.parquet", index=False)
    ox.save_graphml(starter, filepath=data_dir / "speed_road_network.graphml")
    print(
        f"Saved {len(reference)} observed segments and {len(roads)} matched road edges "
        f"in {data_dir}"
    )
    return starter


def download_street_graph(data_dir: str | Path = DATA) -> nx.MultiDiGraph:
    """Download Chicago's directed driving network into the local data folder."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    graph = ox.graph_from_place("Chicago, Illinois, USA", network_type="drive")
    ox.save_graphml(graph, filepath=data_dir / "chicago_road_network.graphml")
    return graph
