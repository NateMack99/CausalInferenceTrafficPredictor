"""Offline checks for the join and the crash-effect estimate.

The join is checked against a hand-built stand-in for what the data and graph
PRs write; the effect is checked against the shock the mock injects, which the
models never see because crash windows are masked out of training.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds
import pytest
import scipy.sparse as sp
import torch

from traffic_gml.data import FREQ
from traffic_gml.evaluate import (
    coverage_funnel,
    effects,
    grid_coverage,
    monthly_coverage,
    neighbour_coverage,
)
from traffic_gml.events import (
    build_windows,
    crash_index,
    event_datasets,
    pre_event_stats,
    processed_tables,
    smooth_over_neighbourhood,
)
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
    tables, sets, _, table, controls, event, control = events
    train = sets["train"]
    times, horizon = train.times, train.horizon

    # A training window at origin `o` is fitted on `[o+1, o+horizon]`, so the
    # fitted cells are those targets. Marking origins instead would let a
    # control whose origin sits in the gap band -- but whose targets were
    # fitted -- pass as held out.
    trained = np.zeros(train.values.shape, dtype=bool)
    origins, fitted = train.origins.numpy(), train.usable.numpy()
    for step in range(1, horizon + 1):
        trained[origins + step] |= fitted

    # `table.crash_time` is the floored bin stamp, so asserting against it
    # compares the arithmetic with itself. Join the reported time back instead.
    reported = tables["crashes"].set_index("crash_record_id")["crash_time"]
    bin_width = pd.Timedelta(FREQ)
    for row in table.head(40).itertuples():
        # The last input bin must *end* at or before the reported crash.
        assert times[row.origin] + bin_width <= pd.Timestamp(reported[row.crash_record_id])
        assert not trained[row.origin + 1 : row.origin + 1 + horizon, row.road].any()
    for row in controls.head(40).itertuples():
        # A control the model was fitted on would report in-sample residuals.
        assert not trained[row.origin + 1 : row.origin + 1 + horizon, row.road].any()

    # One window per crash; repeating it per control would fake the intervals.
    assert len(event) == table.crash_record_id.nunique()
    # Both arms hold one window per (origin, road): a repeated window is the
    # same speeds scored twice, which narrows the interval for free.
    assert len(event) == len(table.drop_duplicates(subset=["origin", "road"]))
    assert len(control) == len(controls.drop_duplicates(subset=["origin", "road"]))
    # Each window scores exactly the one road the crash happened on.
    assert event.usable.sum(dim=1).unique().tolist() == [1]
    assert control.usable.sum(dim=1).unique().tolist() == [1]


def test_masking_inputs_never_touches_the_measured_road(events):
    """Input masking must not reach the arm it measures.

    An event's own crash dirties cells from `origin + 1`, so its lag is
    strictly earlier than anything it marks, and both arms separately require
    their measured road clean across the whole span. If either stopped holding,
    the new `clean[lag]` term in `x_mask` would start eating the measurement
    instead of protecting it.
    """
    _, _, _, _, _, event, control = events
    for data in (event, control):
        assert len(data)
        for i in range(len(data)):
            item = data[i]
            road = int(data.usable[i].nonzero().item())
            t = int(item["origin"])
            lag = slice(t - data.history + 1, t + 1)
            assert torch.equal(item["x_mask"][:, road], torch.isfinite(data.values[lag, road]))
            assert torch.equal(
                item["y_mask"][:, road],
                torch.isfinite(data.values[t + 1 : t + 1 + data.horizon, road]),
            )


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

    # The analysis sample is what the estimator actually runs on, so the funnel
    # takes it rather than re-deriving a stricter population of its own.
    bins, roads, on_grid = crash_index(tables["crashes"], sets["train"].times, ids)
    events = pd.DataFrame({"origin": bins[on_grid] - 1, "road": roads[on_grid]})
    funnel = coverage_funnel(pd.read_csv(crash_csv), sets["train"], ids, edge_index, events)

    assert funnel.crashes.is_monotonic_decreasing
    assert (funnel.step_pct <= 100.0).all()
    assert funnel.crashes.iloc[0] == 3
    # A and C are "matched", B is not; C sits on segment 40, which has no speeds,
    # so it must be dropped at the grid stage rather than snapped to a neighbour.
    assert funnel.crashes.tolist() == [3, 3, 2, 1, 1, 1, 1]
    assert funnel.retained_pct.iloc[-1] == pytest.approx(100 / 3, abs=0.1)


def test_grid_coverage_is_a_fraction(pipeline):
    speeds_dir, graphml, crash_csv = pipeline
    tables = processed_tables(speeds_dir, graphml, crash_csv)
    sets, _ = build_windows(tables, history=HISTORY, horizon=HORIZON)
    # Segments 10, 20 and 30 are observed at every bin; segment 40 never became a road.
    assert grid_coverage(sets["train"]) == pytest.approx(1.0)


def test_two_crashes_in_one_bin_on_one_road_are_one_event(events):
    """One disturbance, counted once -- otherwise that road is double-weighted."""
    tables, sets, _, table, _, event, _ = events
    ids = np.sort(tables["segments"].segment_id.unique())

    first = table.iloc[0]
    twin = tables["crashes"].loc[tables["crashes"].crash_record_id == first.crash_record_id].copy()
    assert len(twin) == 1
    twin["crash_record_id"] = twin["crash_record_id"] + "-twin"
    doubled = pd.concat([tables["crashes"], twin], ignore_index=True)

    again, _, event_again, _ = event_datasets(sets["train"], doubled, ids, seed=0)
    assert len(event_again) == len(event)
    assert not again.duplicated(subset=["origin", "road"]).any()
    assert "-twin" not in set(again.crash_record_id)


def test_neighbour_coverage_bars_get_stricter_not_looser(events):
    """Each stricter bar must be a subset of the one above it."""
    _, sets, edge_index, table, _, _, _ = events
    nearby = neighbour_coverage(sets["train"], edge_index, table, hops=2)

    assert nearby.hop.tolist() == [1, 2]
    for row in nearby.itertuples():
        assert row.events == len(table)
        assert (
            row.events
            >= row.with_neighbour_in_graph
            >= row.any_post_bin
            >= row.half_post_bins
            >= row.all_post_bins
            >= row.full_trajectory
            >= 0
        )


def test_monthly_coverage_accounts_for_every_bin_and_event(events):
    """PR #4 wants coverage, matches and control supply by month before a longer
    run. The table has to account for the whole grid and the whole event
    sample, or a month that cannot supply controls disappears instead of
    showing up as the constraint it is."""
    tables, sets, _, table, controls, _, _ = events
    train = sets["train"]
    monthly = monthly_coverage(train, tables["crashes"], table, controls)

    assert len(monthly) > 1, "the mock spans more than one month"
    assert monthly["bins"].sum() == len(train.times)
    assert monthly["analysis_events"].sum() == len(table)
    # Only crashes inside the grid's own span are counted.
    span = pd.to_datetime(tables["crashes"]["crash_time"])
    inside = span.between(train.times[0], train.times[-1])
    assert monthly["matched_crashes"].sum() == int(inside.sum())

    assert (monthly["observed_pct"] >= 0).all() and (monthly["observed_pct"] <= 100).all()
    assert (monthly["roads_observed"] <= train.num_roads).all()
    # Exact, not approximate: a tolerance here would hide a lost or double-counted
    # control, and control supply is the number this table exists to report.
    supplied = (monthly["controls_per_event"].fillna(0) * monthly["analysis_events"]).round()
    assert int(supplied.sum()) == len(controls)

    # A crash id that does not line up must raise, not be reported as zero supply.
    with pytest.raises(ValueError, match="do not line up"):
        monthly_coverage(train, tables["crashes"], table, controls.assign(crash_record_id="nope"))


def test_pre_event_stats_are_the_trailing_mean_and_its_change():
    """Hand-built, so a window off by one bin cannot hide."""
    speeds = np.array([[10.0], [10.0], [20.0], [20.0], [np.nan], [np.nan]], dtype="float32")
    level, trend = pre_event_stats(speeds, history=4)

    assert level[3, 0] == pytest.approx(15.0)  # mean of bins 0..3
    assert trend[3, 0] == pytest.approx(10.0)  # last two minus first two
    assert level[1, 0] == pytest.approx(10.0)  # short window at the start
    # Unobserved bins contribute nothing rather than zero.
    assert level[5, 0] == pytest.approx(20.0)
    assert np.isnan(pre_event_stats(np.full((4, 1), np.nan, "float32"), 4)[0][3, 0])


def test_matched_controls_sit_closer_in_pre_event_level_and_trend(events):
    """The point of matching: the arms should start from the same place.

    Random draws left the arms differing in pre-event trend by more than the
    effect being measured, so this asserts the matched draw is closer on the
    statistics it matches on.
    """
    tables, sets, _, _, _, _, _ = events
    train = sets["train"]
    ids = np.sort(tables["segments"].segment_id.unique())
    level, trend = pre_event_stats(train.values.numpy(), train.history)

    def mean_gap(match: bool) -> float:
        table, controls, _, _ = event_datasets(
            train, tables["crashes"], ids, controls=3, seed=0, match_pre_event=match
        )
        pairs = controls.merge(
            table[["crash_record_id", "origin"]], on="crash_record_id", suffixes=("_c", "_e")
        )
        gaps = [
            abs(level[c, r] - level[e, r]) + abs(trend[c, r] - trend[e, r])
            for c, e, r in zip(pairs.origin_c, pairs.origin_e, pairs.road, strict=True)
        ]
        return float(np.nanmean(gaps))

    assert mean_gap(match=True) < mean_gap(match=False)

    # Matching is deterministic, so the *controls* do not move with the seed,
    # while a random draw's do. Comparing the event tables proved nothing: those
    # never depend on the seed either way.
    _, matched_a, _, _ = event_datasets(train, tables["crashes"], ids, controls=3, seed=1)
    _, matched_b, _, _ = event_datasets(train, tables["crashes"], ids, controls=3, seed=2)
    pd.testing.assert_frame_equal(matched_a, matched_b)
    _, drawn_a, _, _ = event_datasets(
        train, tables["crashes"], ids, controls=3, seed=1, match_pre_event=False
    )
    _, drawn_b, _, _ = event_datasets(
        train, tables["crashes"], ids, controls=3, seed=2, match_pre_event=False
    )
    assert not drawn_a.equals(drawn_b), "a random draw must move with the seed"

    with pytest.raises(ValueError, match="at least 1"):
        event_datasets(train, tables["crashes"], ids, controls=0)
    with pytest.raises(ValueError, match="no effect without"):
        event_datasets(train, tables["crashes"], ids, match_hops=2, match_pre_event=False)


def test_neighbourhood_smoothing_averages_only_observed_roads():
    """Hand-built. A road with half its neighbours missing must get the mean of
    the half that reported, not a zero-filled average."""
    stat = np.array([[10.0, 20.0, np.nan]])
    # Road 0 neighbours {0,1}, road 1 all three, road 2 itself.
    reach = sp.csr_matrix(np.array([[1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=bool))
    out = smooth_over_neighbourhood(stat, reach)

    assert out[0, 0] == pytest.approx(15.0)
    assert out[0, 1] == pytest.approx(15.0), "the NaN neighbour is left out, not counted as 0"
    assert np.isnan(out[0, 2]), "no observed road in the neighbourhood means no statistic"


def test_matching_hops_needs_a_graph(events):
    tables, sets, edge_index, _, _, _, _ = events
    ids = np.sort(tables["segments"].segment_id.unique())
    with pytest.raises(ValueError, match="edge_index"):
        event_datasets(sets["train"], tables["crashes"], ids, match_hops=2)
    # With one, the wider match changes which controls are drawn.
    narrow, _, _, _ = event_datasets(
        sets["train"], tables["crashes"], ids, controls=3, edge_index=edge_index
    )
    wide, _, w_controls, _ = event_datasets(
        sets["train"], tables["crashes"], ids, controls=3, match_hops=2, edge_index=edge_index
    )
    assert len(narrow) == len(wide), "matching changes the draw, never the eligible sample"
    assert len(w_controls)


def test_midpoint_refuses_to_replace_existing_processed_tables(pipeline, tmp_path):
    """The join writes to `--processed`, whose default holds the frozen tables
    every recorded result rests on. Replacing them silently would change what
    later `--data` runs read while the provenance still named the old window."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "midpoint", Path(__file__).resolve().parents[1] / "scripts" / "midpoint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    speeds_dir, graphml, crash_csv = pipeline
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "speeds.parquet").write_bytes(b"")
    argv = [
        "--speeds",
        str(speeds_dir),
        "--graph",
        str(graphml),
        "--crashes",
        str(crash_csv),
        "--processed",
        str(processed),
        "--out",
        str(tmp_path / "out"),
    ]
    with pytest.raises(SystemExit, match="already holds"):
        module.main(argv)

    # An empty directory is fine, and so is saying you mean it.
    assert module.parse_args([*argv, "--overwrite-processed"]).overwrite_processed
