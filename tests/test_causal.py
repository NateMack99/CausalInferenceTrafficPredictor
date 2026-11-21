"""Offline checks for the matched event study.

The mock fixture injects a known shock -- full strength on the crash road,
`0.45` of it on graph neighbours, decaying `0.7` per bin, and nothing two hops
out -- so the estimator can be held to a number rather than to a shape. That
is what the first test does, and it is the reason this module can be trusted
before any real-data run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from traffic_gml.causal import (
    arm_table,
    covariate_balance,
    event_study,
    fixed_cohort,
    pre_trends,
    reported_cells,
    summarise,
)
from traffic_gml.data import FREQ
from traffic_gml.events import build_windows, event_datasets
from traffic_gml.mock import make_mock

HISTORY, HORIZON = 12, 6
SHOCK, NEIGHBOUR, DECAY = 8.0, 0.45, 0.7


@pytest.fixture(scope="module")
def study():
    tables = make_mock(rows=4, cols=4, days=50, crashes=160, shock=SHOCK, seed=0)
    sets, edge_index = build_windows(tables, history=HISTORY, horizon=HORIZON)
    ids = np.sort(tables["segments"].segment_id.unique())
    train = sets["train"]
    table, controls, _, _ = event_datasets(train, tables["crashes"], ids, controls=5, seed=0)
    events = table.rename(columns={"crash_record_id": "event_id"})
    control = controls.rename(columns={"crash_record_id": "event_id"})
    reported = reported_cells(tables["crashes"], train.times, ids)
    did = event_study(train, events, control, edge_index, hops=2, reported=reported)
    return tables, sets, edge_index, ids, events, control, did


def test_did_recovers_the_injected_hop_and_time_profile(study):
    """The headline check: a known shock, recovered by distance and by time.

    No forecaster is involved, so this is the estimator's own arithmetic being
    held to the mock's ground truth. Tolerances are loose because the mock adds
    autocorrelated noise, clips speeds, and lets crashes overlap -- but the
    ordering and rough magnitudes are not negotiable.
    """
    *_, did = study
    table = summarise(did).set_index(["hop", "minutes"])

    immediate = {hop: table.loc[(hop, 10), "effect"] for hop in (0, 1, 2)}
    assert immediate[0] == pytest.approx(-SHOCK, abs=1.0)
    assert immediate[1] == pytest.approx(-SHOCK * NEIGHBOUR, abs=1.0)
    # The mock injects nothing two hops out, so this ring is the null check.
    assert abs(immediate[2]) < 1.0
    assert immediate[0] < immediate[1] < immediate[2]

    # Time decay on the crash road, against shock * 0.7**step.
    for step in range(HORIZON):
        expected = -SHOCK * DECAY**step
        assert table.loc[(0, (step + 1) * 10), "effect"] == pytest.approx(expected, abs=1.2)


def test_pre_trend_placebo_is_flat(study):
    """A crash cannot move speeds before it happens. If this came out sloped,
    the headline estimate would partly be arms that were already diverging."""
    tables, sets, edge_index, *_, did = study
    _, _, _, _, events, control, _ = study
    trends = pre_trends(sets["train"], events, control, edge_index, hops=1, leads=3)

    assert len(trends)
    for row in trends.itertuples():
        assert abs(row.effect) < 1.0, f"pre-trend at hop {row.hop} is {row.effect}"
        assert row.low <= 0 <= row.high, f"pre-trend at hop {row.hop} excludes zero"


def test_intervals_are_event_clustered_not_cell_counted():
    """Hand-built, because whether clustering *widens* an interval depends on the
    within-event correlation of the data, and the mock's per-road noise is
    independent so it barely matters there. Here each event's cells are
    identical, the worst case, and the interval must then be set by the number
    of events and not by the number of road-time cells.
    """
    # Three events, four perfectly correlated hop-1 cells each.
    did = pd.DataFrame(
        {
            "event_id": np.repeat(["A", "B", "C"], 4),
            "road": np.tile([1, 2, 3, 4], 3),
            "hop": 1,
            "minutes": 10,
            "did": np.repeat([-3.0, -1.0, -2.0], 4),
        }
    )
    row = summarise(did).iloc[0]

    assert row["events"] == 3 and row["n"] == 12
    clustered = (row["high"] - row["low"]) / 2
    naive = 1.96 * float(did["did"].sem())
    # Twelve copies of three numbers must not buy a narrower interval than three.
    assert clustered > naive
    assert clustered == pytest.approx(1.96 * pd.Series([-3.0, -1.0, -2.0]).sem(), abs=1e-3)
    assert row["effect"] == pytest.approx(-2.0)


def test_summarise_weights_every_event_equally(study):
    """An event with many observed ring roads must not outweigh a sparse one:
    the estimate is a mean over events, not over road-time cells."""
    *_, did = study
    ring = did[(did.hop == 1) & (did.minutes == 10)]
    assert len(ring) > ring.event_id.nunique(), "need uneven cells per event to tell these apart"

    row = summarise(did).set_index(["hop", "minutes"]).loc[(1, 10)]
    by_event = ring.groupby("event_id")["did"].mean().mean()
    assert row["effect"] == pytest.approx(by_event, abs=1e-3)
    assert row["effect"] != pytest.approx(ring["did"].mean(), abs=1e-9)


def test_a_contaminated_or_empty_baseline_is_skipped():
    """The first difference is taken from the window's own pre-event level, so a
    road whose baseline is already crash-depressed, or never observed, must be
    dropped rather than differenced from a wrong level."""
    ids = np.arange(1, 4)
    times = pd.date_range("2025-01-06", periods=200, freq=FREQ)
    speed = np.full((200, 3), 30.0, dtype="float32")
    speed[:, 2] = np.nan  # road 3 is never observed
    tables = {
        "speeds": pd.DataFrame(
            {
                "time": np.repeat(times, 3),
                "segment_id": np.tile(ids, 200),
                "speed": speed.ravel(),
            }
        ).dropna(subset=["speed"]),
        "segments": pd.DataFrame({"segment_id": ids}),
        "edges": pd.DataFrame({"src": [1, 2], "dst": [2, 3]}),
        "crashes": pd.DataFrame(
            {"crash_record_id": ["X"], "crash_time": [times[60]], "segment_id": [1]}
        ),
    }
    sets, edge_index = build_windows(tables, history=4, horizon=2, exposure_hops=0)
    data = sets["train"]

    # An origin whose lag sits inside the crash wake on road 1 (node 0).
    dirty = (~data.clean[:, 0]).nonzero().flatten().tolist()
    assert dirty, "the hand-built crash should dirty road 1"
    frame = pd.DataFrame({"event_id": ["E"], "road": [0], "origin": [dirty[0] + 1]})
    rows = arm_table(data, frame, edge_index, hops=2)
    assert 0 not in set(rows.road), "a crash-depressed baseline must not be differenced from"
    # Road 3 (node 2) has no observation at all, so it cannot supply a baseline.
    assert 2 not in set(rows.road)


def test_only_the_control_arm_demands_a_clean_post_span(study):
    """Deliberate asymmetry. The index crash dirties its own post span, so the
    event arm cannot require it clean; a control's neighbour is vetted on
    nothing, so the control arm must."""
    tables, sets, edge_index, ids, events, control, _ = study
    data = sets["train"]
    clean = data.clean.numpy()
    horizon = data.horizon

    crash_arm = arm_table(data, events, edge_index, hops=2)
    control_arm = arm_table(data, control, edge_index, hops=2, require_clean_post=True)

    def all_post_clean(frame: pd.DataFrame) -> bool:
        return all(
            clean[o + 1 : o + 1 + horizon, r].all()
            for o, r in zip(frame.origin, frame.road, strict=True)
        )

    assert all_post_clean(control_arm)
    assert not all_post_clean(crash_arm), "the event arm must keep its own crash's wake"


def test_reported_cells_marks_report_sites_only(study):
    """`clean` spreads across hops and cannot say which crash dirtied a cell;
    this marks report sites, which is what identifies a *second* crash."""
    tables, sets, *_ = study
    data = sets["train"]
    ids = np.sort(tables["segments"].segment_id.unique())
    cells = reported_cells(tables["crashes"], data.times, ids)

    assert cells.shape == tuple(data.values.shape)
    # Far fewer cells than `clean` marks, since there is no hop or time spread.
    assert cells.sum() < int((~data.clean.numpy()).sum())
    assert cells.sum() <= len(tables["crashes"])


def test_fixed_cohort_keeps_only_events_seen_everywhere(study):
    """An available-case estimate can show distance decay just because the far
    rings were measured on better-observed events."""
    *_, did = study
    cohort = fixed_cohort(did)
    wanted = did.groupby(["hop", "minutes"]).ngroups

    assert set(cohort.event_id) <= set(did.event_id)
    assert len(cohort) <= len(did)
    for _, group in cohort.groupby("event_id"):
        assert len(group[["hop", "minutes"]].drop_duplicates()) == wanted


def test_covariate_balance_compares_the_arms(study):
    """Road features balance by construction -- a control is the same road -- so
    those rows check the matching; the pre-event rows are the informative ones."""
    tables, sets, edge_index, ids, events, control, _ = study
    balance = covariate_balance(
        sets["train"], events, control, tables["segments"], ids, columns=("length_m", "degree")
    )
    features = set(balance.feature)
    assert {"pre_level", "pre_trend", "length_m", "degree"} <= features
    for row in balance.itertuples():
        if row.feature in ("length_m", "degree"):
            assert abs(row.std_diff) < 0.25, f"{row.feature} should match by construction"
        assert row.n_crash and row.n_control
