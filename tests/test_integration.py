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
from traffic_gml.evaluate import coverage_funnel, effects, grid_coverage
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

    # [start, end) on both the partition filter and the timestamp filter.
    windowed = processed_tables(
        speeds_dir, graphml, crash_csv, window=("2025-03-01 00:00", "2025-03-01 02:00")
    )
    assert windowed["speeds"].time.min() == pd.Timestamp("2025-03-01 00:00")
    assert windowed["speeds"].time.max() == pd.Timestamp("2025-03-01 01:50")
    assert windowed["crashes"].empty


@pytest.fixture(scope="module")
def events():
    tables = make_mock(rows=4, cols=4, days=30, crashes=120, seed=0)
    sets, edge_index = build_windows(tables, history=HISTORY, horizon=HORIZON)
    ids = np.sort(tables["segments"].segment_id.unique())
    table, controls, event, control = event_datasets(sets["train"], tables["crashes"], ids, seed=0)
    return tables, sets, edge_index, table, controls, event, control


def test_event_windows_are_pre_crash_and_controls_are_held_out(events):
    """Both arms must be out-of-sample, or the control is not a placebo."""
    _, sets, _, table, controls, event, control = events
    train = sets["train"]
    times = train.times
    trained = np.zeros(train.values.shape, dtype=bool)
    trained[train.origins.numpy()] = train.usable.numpy()

    for row in table.head(40).itertuples():
        # Checked against wall-clock time, not against the arithmetic that built it:
        # the last input bin must end at or before the crash.
        assert times[row.origin] < row.crash_time
        assert not trained[row.origin, row.road]
    for row in controls.head(40).itertuples():
        # A control the model was fitted on would report in-sample residuals.
        assert not trained[row.origin, row.road]

    # One window per crash; repeating it per control would fake the intervals.
    assert len(event) == table.crash_record_id.nunique()
    assert len(control) == len(controls)
    # Each window scores exactly the one road the crash happened on.
    assert event.usable.sum(dim=1).unique().tolist() == [1]
    assert control.usable.sum(dim=1).unique().tolist() == [1]


def test_effects_recover_the_injected_shock_and_controls_stay_flat(events):
    _, sets, edge_index, _, _, event, control = events
    model = build_model("historical_mean", sets["train"]).fit(sets["train"])
    crash = effects(model, event, edge_index).groupby("minutes").effect.mean()
    flat = effects(model, control, edge_index).groupby("minutes").effect.mean()

    # mock injects 8.0 mph decaying by 0.7 per bin for 6 bins; crash_cells masks
    # horizon + 1 of them, so at HORIZON=3 the last two bins do reach training.
    assert crash.iloc[0] < -5.0
    assert crash.is_monotonic_increasing  # the slowdown decays back toward zero
    assert abs(flat).max() < 0.5  # matched controls show no effect


def test_coverage_funnel_never_gains_crashes_between_stages(pipeline):
    """Every stage must narrow the same population of crashes.

    The audit table holds all three crashes; the joined `crashes` table holds
    only the one that matched a monitored road. Counting the early stages from
    one and the later stages from the other reported more survivors than the
    stage above, which is how a retention funnel ends up above 100%.
    """
    speeds_dir, graphml, crash_csv = pipeline
    tables = processed_tables(speeds_dir, graphml, crash_csv)
    sets, edge_index = build_windows(tables, history=HISTORY, horizon=HORIZON)
    ids = np.sort(tables["segments"].segment_id.unique())

    funnel = coverage_funnel(pd.read_csv(crash_csv), sets["train"], ids, edge_index)

    assert funnel.crashes.is_monotonic_decreasing
    assert (funnel.step_pct <= 100.0).all()
    assert funnel.crashes.iloc[0] == 3
    # A and C are "matched", B is not; C sits on segment 40, which has no speeds,
    # so it must be dropped at the grid stage rather than snapped to a neighbour.
    assert funnel.crashes.tolist() == [3, 3, 2, 1, 1, 1]
    assert funnel.retained_pct.iloc[-1] == pytest.approx(100 / 3, abs=0.1)


def test_grid_coverage_is_a_fraction(pipeline):
    speeds_dir, graphml, crash_csv = pipeline
    tables = processed_tables(speeds_dir, graphml, crash_csv)
    sets, _ = build_windows(tables, history=HISTORY, horizon=HORIZON)
    # Segments 10, 20 and 30 are observed at every bin; segment 40 never became a road.
    assert grid_coverage(sets["train"]) == pytest.approx(1.0)
