"""Segment-level graph analysis built on top of `build_graph`.

`build_graph` already snaps every crash to the nearest road in the *full*
Chicago network (50 m limit) and writes an audit table beside the GraphML. This
module reuses that match rather than redoing it, then adds:

1. a NetworkX graph whose nodes are speed segments, linked when their roads meet
   at an OSM intersection;
2. a per-crash table giving the road segment each accident belongs to;
3. degree and betweenness centrality for every segment;
4. an interactive Folium map of roads (coloured by betweenness) and crashes.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path

import branca.colormap as cm
import folium
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from folium.plugins import HeatMap

from .roads import DATA

TEXT_COLUMNS = ("street", "direction", "from_street", "to_street")
CRASH_COLUMNS = (
    "crash_record_id",
    "timestamp",
    "latitude",
    "longitude",
    "match_status",
    "distance_m",
    "segment_id",
    "edge_u",
    "edge_v",
    "edge_key",
)
#: Above this many segments, betweenness is estimated from a sample of sources.
EXACT_LIMIT = 5_000
DEFAULT_SAMPLE = 1_000


# --------------------------------------------------------------------------
# 1. Segment graph: nodes = speed segments, edges = shared intersection
# --------------------------------------------------------------------------
def build_segment_graph(road: nx.MultiDiGraph, reference: pd.DataFrame | None = None) -> nx.Graph:
    """Link two segments when any of their OSM edges touch the same intersection.

    A segment can span several OSM edges, so its length is their sum. The link
    weight is the mean of the two segment lengths in metres (a midpoint-to-
    midpoint travel estimate), used as the distance for betweenness.
    """
    segments: dict[int, dict] = {}
    touching: dict[int, set[int]] = defaultdict(set)  # OSM node -> segment ids
    for u, v, edge in road.edges(data=True):
        if edge.get("segment_id") is None:
            continue
        sid = int(edge["segment_id"])
        segment = segments.setdefault(
            sid,
            {
                "length_m": 0.0,
                "speed_mph": float(edge.get("speed_mph", np.nan)),
                "congestion": float(edge.get("congestion", np.nan)),
                "max_speed_mph": float(edge.get("max_speed_mph", np.nan)),
                "coverage_fraction": float(edge.get("coverage_fraction", np.nan)),
            },
        )
        segment["length_m"] += float(edge.get("length", 0.0))
        touching[u].add(sid)
        touching[v].add(sid)
    if not segments:
        raise ValueError("The road graph has no edges with a segment_id")

    names: dict[int, dict] = {}
    if reference is not None:
        keep = [c for c in TEXT_COLUMNS if c in reference.columns]
        names = (
            reference.drop_duplicates("segment_id")
            .set_index("segment_id")[keep]
            .fillna("")
            .to_dict("index")
        )

    G = nx.Graph()
    for sid, attrs in segments.items():
        text = {c: str(names.get(sid, {}).get(c, "")) for c in TEXT_COLUMNS}
        G.add_node(sid, **attrs, **text)

    junctions = 0
    for sids in touching.values():
        if len(sids) < 2:
            continue
        junctions += 1
        for a, b in combinations(sorted(sids), 2):
            if G.has_edge(a, b):  # they meet at more than one intersection
                G[a][b]["shared"] += 1
                continue
            weight = (G.nodes[a]["length_m"] + G.nodes[b]["length_m"]) / 2
            G.add_edge(a, b, weight=max(weight, 1.0), shared=1)
    G.graph["junctions"] = junctions
    return G


# --------------------------------------------------------------------------
# 2. Crash -> segment, reusing build_graph's audit table
# --------------------------------------------------------------------------
def match_crashes_to_segments(road: nx.MultiDiGraph) -> pd.DataFrame:
    """One row per crash in the window, with the segment it matched (or <NA>).

    `build_graph` records the matched OSM edge (edge_u, edge_v, edge_key) for each
    crash; the edge's segment_id says which speed segment that is. Crashes with
    status other than "matched" (too_far, outside_subgraph, invalid_coordinates)
    keep their status and get no segment.
    """
    output = Path(road.graph["output_path"])
    audit_path = output.with_name(output.stem + "_crashes.parquet")
    if not audit_path.exists():
        raise FileNotFoundError(f"{audit_path} not found; rebuild the graph with build_graph")
    audit = pd.read_parquet(audit_path)

    segment_of = {
        (int(u), int(v), int(k)): int(d["segment_id"])
        for u, v, k, d in road.edges(keys=True, data=True)
        if d.get("segment_id") is not None
    }
    hit = audit.loc[audit.match_status == "matched"]
    found = [
        segment_of.get((int(u), int(v), int(k)))
        for u, v, k in zip(hit.edge_u, hit.edge_v, hit.edge_key, strict=True)
    ]
    audit["segment_id"] = pd.Series(found, index=hit.index, dtype="Int64")
    return audit[[c for c in CRASH_COLUMNS if c in audit.columns]]


# --------------------------------------------------------------------------
# 3. Centrality
# --------------------------------------------------------------------------
def add_centrality(G: nx.Graph, k: int | None = None, seed: int = 42) -> None:
    """Degree, degree centrality and betweenness, stored as node attributes.

    Betweenness uses the segment-length weights, so it ranks segments by how many
    shortest *distance* paths pass through them. Exact for up to 5,000 segments;
    beyond that it samples `k` source nodes (default 1,000).
    """
    n = G.number_of_nodes()
    if n == 0:
        raise ValueError("Cannot compute centrality on an empty graph")
    if k is None and n > EXACT_LIMIT:
        k = DEFAULT_SAMPLE
    k = min(k, n) if k else None
    nx.set_node_attributes(G, dict(G.degree()), "degree")
    nx.set_node_attributes(G, nx.degree_centrality(G), "degree_centrality")
    nx.set_node_attributes(
        G,
        nx.betweenness_centrality(G, k=k, weight="weight", normalized=True, seed=seed),
        "betweenness",
    )


def add_crash_counts(G: nx.Graph, crashes: pd.DataFrame) -> None:
    counts = crashes["segment_id"].dropna().astype("int64").value_counts()
    nx.set_node_attributes(G, {s: int(counts.get(s, 0)) for s in G}, "crashes")


def segment_table(G: nx.Graph) -> pd.DataFrame:
    table = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient="index")
    table.index.name = "segment_id"
    return table.sort_values("betweenness", ascending=False)


# --------------------------------------------------------------------------
# 4. Map
# --------------------------------------------------------------------------
def make_map(
    road: nx.MultiDiGraph,
    table: pd.DataFrame,
    crashes: pd.DataFrame,
    path: Path,
    max_points: int = 5_000,
) -> Path:
    """Roads coloured/sized by betweenness, plus matched and unmatched crashes."""
    edges = ox.graph_to_gdfs(road, nodes=False).reset_index(drop=True)
    edges = edges.loc[edges["segment_id"].notna(), ["segment_id", "geometry"]]
    edges["segment_id"] = edges["segment_id"].astype("int64")
    fields = [
        c
        for c in (*TEXT_COLUMNS, "betweenness", "degree", "crashes", "congestion")
        if c in table.columns
    ]
    edges = edges.merge(table[fields].reset_index(), on="segment_id", how="left")
    for column in ("betweenness", "congestion"):
        if column in edges:
            edges[column] = edges[column].round(5)

    west, south, east, north = edges.total_bounds
    fmap = folium.Map(
        location=[(south + north) / 2, (west + east) / 2],
        zoom_start=11,
    )
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Street_Map/MapServer/tile/{z}/{y}/{x}"
        ),
        name="Esri World Street Map",
        attr="Tiles &copy; Esri &mdash; Sources: Esri, HERE, Garmin, USGS, NPS",
        max_zoom=19,
    ).add_to(fmap)

    vmax = max(float(edges["betweenness"].quantile(0.99)), 1e-12)
    cmap = cm.LinearColormap(
        ["#2c7bb6", "#ffffbf", "#d7191c"], vmin=0, vmax=vmax, caption="Betweenness centrality"
    )
    cmap.add_to(fmap)

    def style(feature: dict) -> dict:
        value = min(feature["properties"].get("betweenness") or 0.0, vmax)
        return {"color": cmap(value), "weight": 1.5 + 5 * value / vmax, "opacity": 0.9}

    folium.GeoJson(
        edges.to_json(),
        name="Roads (betweenness)",
        style_function=style,
        tooltip=folium.GeoJsonTooltip(fields=["segment_id", *fields]),
    ).add_to(fmap)

    plotted = crashes.dropna(subset=["latitude", "longitude"])
    plotted = plotted[plotted.match_status != "invalid_coordinates"]
    matched = plotted[plotted.match_status == "matched"]
    unmatched = plotted[plotted.match_status != "matched"]

    def markers(frame: pd.DataFrame, name: str, color: str, show: bool) -> None:
        group = folium.FeatureGroup(name=name, show=show)
        sample = frame.sample(min(len(frame), max_points), random_state=1)
        for row in sample.itertuples():
            segment = "" if pd.isna(row.segment_id) else f"segment {int(row.segment_id)}<br>"
            folium.CircleMarker(
                (row.latitude, row.longitude),
                radius=2.5,
                color=color,
                weight=0.5,
                fill=True,
                fill_opacity=0.6,
                tooltip=f"{segment}{row.match_status}<br>{getattr(row, 'timestamp', '')}",
            ).add_to(group)
        group.add_to(fmap)

    markers(matched, f"Crashes matched to a segment (n={len(matched):,})", "#111111", True)
    markers(unmatched, f"Crashes not matched (n={len(unmatched):,})", "#e08214", False)
    if len(matched):
        heat = matched.sample(min(len(matched), 50_000), random_state=1)
        HeatMap(
            heat[["latitude", "longitude"]].to_numpy().tolist(),
            radius=8,
            name="Crash heatmap",
            show=False,
        ).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save(path)
    return path


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def analyze(
    road: nx.MultiDiGraph,
    data_dir: str | Path = DATA,
    output_dir: str | Path | None = None,
    k: int | None = None,
    max_crash_points: int = 5_000,
    top: int = 10,
) -> nx.Graph:
    """Run all four steps on a graph from `build_graph`; write files beside it."""
    graph_path = Path(road.graph["output_path"])
    out = Path(output_dir) if output_dir else graph_path.parent
    out.mkdir(parents=True, exist_ok=True)
    stem = graph_path.stem

    reference_path = Path(data_dir) / "speed_segments.parquet"
    reference = pd.read_parquet(reference_path) if reference_path.exists() else None

    G = build_segment_graph(road, reference)
    print(
        f"Segment graph: {G.number_of_nodes():,} segments, {G.number_of_edges():,} links "
        f"at {G.graph['junctions']:,} intersections, "
        f"{nx.number_connected_components(G):,} components, "
        f"{sum(1 for _ in nx.isolates(G)):,} isolated"
    )

    crashes = match_crashes_to_segments(road)
    print(
        "Crash matching: "
        + ", ".join(
            f"{status}={count:,}" for status, count in crashes.match_status.value_counts().items()
        )
    )
    add_crash_counts(G, crashes)
    add_centrality(G, k=k)

    table = segment_table(G)
    show = [
        c
        for c in ("street", "from_street", "to_street", "betweenness", "degree", "crashes")
        if c in table.columns
    ]
    print(f"\nTop {top} segments by betweenness:")
    print(table[show].head(top).to_string())

    nx.write_graphml(G, out / f"{stem}_segments.graphml")
    table.to_csv(out / f"{stem}_segment_metrics.csv")
    crashes.to_csv(out / f"{stem}_crash_segments.csv", index=False)
    map_path = make_map(road, table, crashes, out / f"{stem}_map.html", max_crash_points)
    print(f"\nWrote segment graph, metrics, crash matches and map to {out}\n{map_path}")
    return G
