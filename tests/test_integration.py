"""Offline checks for the join and the crash-effect estimate.

The join is checked against a hand-built stand-in for what the data and graph
PRs write; the effect is checked against the shock the mock injects, which the
models never see because crash windows are masked out of training.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds
import pytest

from traffic_gml.data import FREQ
from traffic_gml.evaluate import effects
from traffic_gml.events import build_windows, event_datasets, processed_tables
from traffic_gml.mock import make_mock
from traffic_gml.models import build_model

HISTORY, HORIZON = 6, 3


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """A miniature of what `csv_to_parquet` and `segments.analyze` write."""
    root = tmp_path_factory.mktemp("pipeline")
    times = pd.date_range("2025-03-01", periods=300, freq=FREQ)
    ids = [10, 20, 30, 40]
    frame = pd.DataFrame(
        {
            "timestamp": np.repeat(times, len(ids)),
            "segment_id": np.tile(ids, len(times)),
            "speed": np.full(len(times) * len(ids), 30.0, dtype="float32"),
            "speed_available": True,
        }
    )
    frame.loc[frame.segment_id == 40, "speed"] = None  # a road the graph has but speeds do not
    frame["date"] = frame.timestamp.dt.date
    speeds_dir = root / "speeds"
    pds.write_dataset(
        pa.Table.from_pandas(frame, preserve_index=False),
        speeds_dir,
        format="parquet",
        partitioning=pds.partitioning(pa.schema([("date", pa.date32())]), flavor="hive"),
    )

    graph = nx.Graph()
    for sid in ids:
        graph.add_node(sid, betweenness=0.5, degree=2, length_m=100.0)
    graph.add_edges_from([(10, 20), (20, 30), (30, 40)])
    graphml = root / "segments.graphml"
    nx.write_graphml(graph, graphml)

    crashes = pd.DataFrame(
        {
            "crash_record_id": ["A", "B", "C"],
            "timestamp": [times[100], times[120], times[140]],
            "match_status": ["matched", "too_far", "matched"],
            "segment_id": [20, pd.NA, 40],
        }
    )
    crash_csv = root / "crash_segments.csv"
    crashes.to_csv(crash_csv, index=False)
    return speeds_dir, graphml, crash_csv


def test_join_filters_to_matched_crashes_and_observed_roads(pipeline):
    speeds_dir, graphml, crash_csv = pipeline
    tables = processed_tables(speeds_dir, graphml, crash_csv)

    # Segment 40 has a graph node but no observed speed, so it is not a road.
    assert sorted(tables["segments"].segment_id) == [10, 20, 30]
    assert set(tables["speeds"].segment_id) == {10, 20, 30}
    # Its edge goes with it; the rest appear in both directions for message passing.
    assert set(map(tuple, tables["edges"].to_numpy())) == {(10, 20), (20, 10), (20, 30), (30, 20)}
    # Only "matched" crashes become events, and only on roads that survived.
    assert tables["crashes"].crash_record_id.tolist() == ["A"]
    assert tables["crashes"].segment_id.tolist() == [20]
    assert tables["segments"].columns.str.contains("betweenness").any()


def test_join_window_bounds_are_half_open(pipeline):
    speeds_dir, graphml, crash_csv = pipeline
    tables = processed_tables(
        speeds_dir, graphml, crash_csv, window=("2025-03-01 00:00", "2025-03-01 02:00")
    )
    assert tables["speeds"].time.min() == pd.Timestamp("2025-03-01 00:00")
    assert tables["speeds"].time.max() == pd.Timestamp("2025-03-01 01:50")
    assert tables["crashes"].empty  # crash A is at bin 100, outside this window


@pytest.fixture(scope="module")
def events():
    tables = make_mock(rows=4, cols=4, days=30, crashes=120, seed=0)
    sets, edge_index = build_windows(tables, history=HISTORY, horizon=HORIZON)
    ids = np.sort(tables["segments"].segment_id.unique())
    table, event, control = event_datasets(sets["train"], tables["crashes"], ids, seed=0)
    return tables, sets, edge_index, table, event, control


def test_event_windows_are_pre_crash_and_controls_are_crash_free(events):
    tables, sets, _, table, event, control = events
    clean = sets["train"].clean.numpy()
    for row in table.head(40).itertuples():
        # x ends the bin before the crash, so no post-crash speed is ever an input.
        assert row.event_origin == row.crash_bin - 1
        span = slice(row.control_origin - HISTORY + 1, row.control_origin + HORIZON + 1)
        assert clean[span, row.road].all()
        assert sets["train"].slot[row.control_origin] == sets["train"].slot[row.event_origin]
    # Each window scores exactly the one road the crash happened on.
    assert event.usable.sum(dim=1).unique().tolist() == [1]
    assert control.usable.sum(dim=1).unique().tolist() == [1]


def test_effects_recover_the_injected_shock_and_controls_stay_flat(events):
    tables, sets, edge_index, _, event, control = events
    model = build_model("historical_mean", sets["train"]).fit(sets["train"])
    crash = effects(model, event, edge_index).groupby("minutes").effect.mean()
    flat = effects(model, control, edge_index).groupby("minutes").effect.mean()

    # mock injects 8.0 mph decaying by 0.7 per bin on the crash road.
    assert crash.iloc[0] < -5.0
    assert crash.is_monotonic_increasing  # the slowdown decays back toward zero
    assert abs(flat).max() < 0.5  # matched controls show no effect
