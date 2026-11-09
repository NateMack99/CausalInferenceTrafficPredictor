"""Offline checks for the crash-to-segment match and the segment graph.

Every headline number downstream -- the matched-crash count, the per-segment
crash counts, the event windows `events.processed_tables` builds -- is read off
`match_crashes_to_segments`, so its statuses, its segment ids and its columns
are checked against a hand-built network small enough to reason about by hand.
"""

from __future__ import annotations

import networkx as nx
import osmnx as ox
import pandas as pd
import pytest
from shapely.geometry import LineString

from traffic_gml.data import csv_to_parquet
from traffic_gml.graph import build_graph
from traffic_gml.segments import (
    CRASH_COLUMNS,
    add_centrality,
    add_crash_counts,
    build_segment_graph,
    match_crashes_to_segments,
    segment_table,
)

#: (lon, lat). 1-2-3-4-5-6 runs east along one latitude, 1-7-3 is a parallel
#: detour, 8-9 is monitored but disconnected and 10-11 is not monitored at all.
COORDINATES = {
    1: (-87.6300, 41.8800),
    2: (-87.6250, 41.8800),
    3: (-87.6200, 41.8800),
    4: (-87.6150, 41.8800),
    5: (-87.6100, 41.8800),
    6: (-87.6050, 41.8800),
    7: (-87.6250, 41.8830),
    8: (-87.6300, 41.8700),
    9: (-87.6200, 41.8700),
    10: (-87.6300, 41.8900),
    11: (-87.6200, 41.8900),
}
#: OSM edge -> (length in metres, speed segment). A segment can span several
#: edges, and (10, 11) has no segment so it stays out of the monitored subgraph.
EDGES = {
    (1, 2): (400.0, 10),
    (2, 3): (400.0, 10),
    (3, 4): (300.0, 20),
    (4, 5): (300.0, 20),
    (5, 6): (200.0, 40),
    (1, 7): (500.0, 50),
    (7, 3): (500.0, 50),
    (8, 9): (300.0, 30),
    (10, 11): (900.0, None),
}
#: (record id, time, lat, lon, expected status, expected segment). Repeating a
#: record is deliberate: duplicate rows reach the real portal extracts.
CRASHES = (
    ("C-ON-10", "01/01/2025 12:00:00 PM", 41.8800, -87.6280, "matched", 10),
    ("C-DUP", "01/01/2025 12:10:00 PM", 41.8800, -87.6280, "matched", 10),
    ("C-DUP", "01/01/2025 12:10:00 PM", 41.8800, -87.6280, "matched", 10),
    ("C-ON-20", "01/01/2025 12:20:00 PM", 41.8800, -87.6175, "matched", 20),
    ("C-FAR", "01/01/2025 12:30:00 PM", 41.8800, -87.5000, "too_far", None),
    ("C-UNMONITORED", "01/01/2025 12:40:00 PM", 41.8900, -87.6250, "outside_subgraph", None),
    ("C-NO-COORDS", "01/01/2025 12:50:00 PM", "", "", "invalid_coordinates", None),
    ("C-NULL-ISLAND", "01/01/2025 12:55:00 PM", 0.0, 0.0, "invalid_coordinates", None),
)
#: segment_id -> summed OSM edge length, i.e. what `build_segment_graph` nodes
#: should carry.
LENGTHS = {10: 800.0, 20: 600.0, 30: 300.0, 40: 200.0, 50: 1000.0}


@pytest.fixture(scope="module")
def road(tmp_path_factory):
    """A graph straight from `build_graph`: the only input `segments.py` takes.

    Written to disk and reloaded the way the real run does, because
    `match_crashes_to_segments` reads the crash audit parquet beside the
    GraphML rather than anything held in memory.
    """
    data = tmp_path_factory.mktemp("segments")
    full = nx.MultiDiGraph(crs="EPSG:4326", simplified=True)
    for node, (x, y) in COORDINATES.items():
        full.add_node(node, x=x, y=y)
    for (u, v), (length, _) in EDGES.items():
        full.add_edge(
            u, v, key=0, length=length, geometry=LineString([COORDINATES[u], COORDINATES[v]])
        )
    ox.save_graphml(full, filepath=data / "chicago_road_network.graphml")

    monitored = full.edge_subgraph([(u, v, 0) for u, v in EDGES if EDGES[u, v][1]]).copy()
    for (u, v), (_, segment) in EDGES.items():
        if segment:
            monitored.edges[u, v, 0].update(
                segment_id=segment, coverage_fraction=0.9, max_speed_mph=40.0
            )
    ox.save_graphml(monitored, filepath=data / "speed_road_network.graphml")

    segments = sorted(LENGTHS)
    speeds = pd.DataFrame(
        {
            "TIME": ["01/01/2025 08:00:00 AM", "01/01/2025 08:30:00 AM"] * len(segments),
            "SEGMENT_ID": [s for s in segments for _ in range(2)],
            "SPEED": [20, 30] * len(segments),
        }
    )
    crashes = pd.DataFrame(
        [row[:4] for row in CRASHES],
        columns=["CRASH_RECORD_ID", "CRASH_DATE", "LATITUDE", "LONGITUDE"],
    )
    for name, kind, frame in [
        ("chicago-speeds", "speeds", speeds),
        ("chicago-crashes", "crashes", crashes),
    ]:
        frame.to_csv(data / f"{name}.csv", index=False)
        csv_to_parquet(data / f"{name}.csv", kind=kind)
    return build_graph(
        ("2025-01-01 12:00", "2025-01-01 13:00"),
        ("2025-01-01 08:00", "2025-01-01 09:00"),
        data_dir=data,
        output_dir=data / "output",
    )


@pytest.fixture(scope="module")
def crashes(road):
    return match_crashes_to_segments(road)


@pytest.fixture(scope="module")
def reference():
    """Stand-in for `data/speed_segments.parquet`, which supplies street names."""
    return pd.DataFrame(
        {
            "segment_id": sorted(LENGTHS),
            "street": ["Madison", "Monroe", "Lake", "Adams", "Jackson"],
            "direction": ["EB"] * len(LENGTHS),
        }
    )


def test_crash_table_keeps_one_row_per_crash_with_the_downstream_columns(crashes):
    """`events.processed_tables` reads this frame back from CSV by column name,
    and the coverage funnel counts its rows, so a dropped row or a renamed
    column silently moves every number in the report."""
    assert list(crashes.columns) == list(CRASH_COLUMNS)
    assert {"segment_id", "match_status", "timestamp", "crash_record_id"} <= set(crashes.columns)
    assert len(crashes) == len(CRASHES)
    assert crashes.match_status.value_counts().to_dict() == {
        "matched": 4,
        "invalid_coordinates": 2,
        "too_far": 1,
        "outside_subgraph": 1,
    }
    assert crashes.segment_id.dtype == "Int64"
    assert crashes.timestamp.is_monotonic_increasing


def test_a_crash_is_matched_to_the_segment_it_sits_on(crashes):
    """A crash attributed to the wrong road puts its slowdown on the wrong
    node, which no later stage can detect or undo."""
    expected = sorted((record, status, segment) for record, _, _, _, status, segment in CRASHES)
    found = sorted(
        (
            row.crash_record_id,
            row.match_status,
            None if pd.isna(row.segment_id) else int(row.segment_id),
        )
        for row in crashes.itertuples()
    )
    assert found == expected


def test_an_unmatched_crash_never_borrows_a_nearby_segment(crashes):
    """Snapping a crash 9 km to the nearest road, or placing one with no
    coordinates at all, would invent events on roads nothing happened on."""
    assert crashes.loc[crashes.match_status != "matched", "segment_id"].isna().all()

    far = crashes.loc[crashes.match_status == "too_far"].squeeze()
    assert far.distance_m > 50  # build_graph's threshold, the reason for the status
    assert not pd.isna(far.edge_u)  # a nearest edge was found and still refused

    invalid = crashes.loc[crashes.match_status == "invalid_coordinates"]
    assert invalid.distance_m.isna().all()
    assert invalid[["edge_u", "edge_v", "edge_key"]].isna().all().all()
    assert invalid.latitude.isna().sum() == 1  # the blank pair
    assert (invalid.latitude.fillna(1.0) == 0.0).sum() == 1  # and the (0, 0) pair


def test_repeated_crash_records_stay_separate_rows(crashes):
    """This frame is the raw audit, not a deduplicated event list: the same
    record twice stays twice. `events.event_datasets` is what collapses one
    disturbance per (bin, road), and it can only do that if nothing upstream
    has quietly dropped or merged rows first."""
    duplicates = crashes.loc[crashes.crash_record_id == "C-DUP"]
    assert len(duplicates) == 2
    assert duplicates.segment_id.tolist() == [10, 10]
    assert duplicates.timestamp.nunique() == 1


def test_segments_are_nodes_and_shared_intersections_are_links(road, reference):
    """Nodes are speed segments, not OSM edges: a segment spanning several edges
    must collapse to one node whose length is their sum, or the per-segment
    speeds and crash counts land on partial roads."""
    G = build_segment_graph(road, reference)

    assert sorted(G) == sorted(LENGTHS)
    assert {s: G.nodes[s]["length_m"] for s in G} == LENGTHS
    assert G.nodes[10]["street"] == "Madison"
    assert G.nodes[10]["from_street"] == ""  # absent from the reference, never NaN
    assert G.nodes[10]["speed_mph"] == 25.0  # the window mean of 20 and 30

    # 10 and 50 are parallel streets meeting at OSM nodes 1 and 3: one link that
    # records both. 30 touches no other segment and must stay a node regardless.
    links = sorted(tuple(sorted(link)) for link in G.edges())
    assert links == [(10, 20), (10, 50), (20, 40), (20, 50)]
    assert G[10][50]["shared"] == 2
    assert G[10][20]["shared"] == 1
    assert G[10][20]["weight"] == pytest.approx((LENGTHS[10] + LENGTHS[20]) / 2)
    assert G.graph["junctions"] == 3  # OSM nodes 1, 3 and 5
    assert list(nx.isolates(G)) == [30]

    bare = nx.MultiDiGraph()
    bare.add_edge(1, 2, key=0, length=10.0)
    with pytest.raises(ValueError, match="segment_id"):
        build_segment_graph(bare)


def test_centrality_ranks_the_connector_and_repeats_under_a_seed(road):
    """Betweenness is a reported road feature and a model input, so it has to be
    reproducible: the sampled estimate must depend only on `seed`."""
    G = build_segment_graph(road)
    add_centrality(G)

    # 40 is reachable only through 20, so 20 is the one segment that lies on a
    # shortest path between two others.
    assert [s for s in G if G.nodes[s]["betweenness"] > 0] == [20]
    assert G.nodes[20]["degree"] == 3
    assert G.nodes[30]["degree"] == 0
    assert G.nodes[30]["degree_centrality"] == 0.0

    def sampled(seed: int) -> dict[int, float]:
        other = build_segment_graph(road)
        add_centrality(other, k=2, seed=seed)
        return {s: other.nodes[s]["betweenness"] for s in other}

    assert sampled(11) == sampled(11)


def test_crash_counts_land_on_every_node_including_the_empty_ones(road, crashes):
    """A segment with no crash must read 0, not be missing: `segment_table`
    writes these straight into the metrics CSV and the report."""
    G = build_segment_graph(road)
    add_crash_counts(G, crashes)

    # Three rows matched segment 10, two of them the duplicated record.
    assert {s: G.nodes[s]["crashes"] for s in G} == {10: 3, 20: 1, 30: 0, 40: 0, 50: 0}
    assert sum(G.nodes[s]["crashes"] for s in G) == (crashes.match_status == "matched").sum()


def test_segment_table_is_one_ranked_row_per_segment(road, crashes, reference):
    G = build_segment_graph(road, reference)
    add_crash_counts(G, crashes)
    add_centrality(G)
    table = segment_table(G)

    assert table.index.name == "segment_id"
    assert sorted(table.index) == sorted(LENGTHS)
    assert {
        "street",
        "length_m",
        "speed_mph",
        "congestion",
        "betweenness",
        "degree",
        "degree_centrality",
        "crashes",
    } <= set(table.columns)
    assert table.betweenness.is_monotonic_decreasing
    assert table.index[0] == 20
