"""Offline checks for the semi-synthetic benchmark.

Every number PR #9 reports is scored against `Disruptions.truth`, so these
tests are mostly about the labels agreeing with the array they describe. The
first one is the leakage rule: an injected effect must never reach an input.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from traffic_gml.data import BIN_MINUTES
from traffic_gml.evaluate import control_corrected, effect_recovery, effects
from traffic_gml.events import build_windows, fitted_cells
from traffic_gml.mock import make_mock
from traffic_gml.models import build_model
from traffic_gml.synthetic import (
    _pre_mean,
    effect_profile,
    inject,
    matched_controls,
    treated_windows,
    untreated_windows,
)

HISTORY, HORIZON = 6, 3


@pytest.fixture(scope="module")
def built():
    tables = make_mock(rows=4, cols=4, days=12, crashes=20, seed=0)
    return build_windows(tables, history=HISTORY, horizon=HORIZON)


@pytest.fixture(scope="module")
def wide():
    """A series long enough that the held-out split spans over a week.

    `matched_controls` draws a control one or more whole weeks from its event,
    so a shorter held-out split has no candidate that is not a fitted training
    cell. 60 days puts val+test at about 18 days, which is enough.
    """
    tables = make_mock(rows=3, cols=3, days=60, crashes=40, seed=0)
    return build_windows(tables, history=HISTORY, horizon=HORIZON)


def cell_of(row) -> tuple[int, int]:
    """The absolute `(bin, road)` a truth row describes."""
    return int(row.origin + row.minutes // BIN_MINUTES), int(row.road)


def test_injection_never_reaches_the_input_history(built):
    """The rule the whole benchmark depends on. Injection starts at `origin + 1`,
    so a forecaster reading the treated grid must see byte-identical inputs to
    the untreated one; otherwise it is being shown the answer."""
    sets, edge_index = built
    disruptions = inject(sets["test"], edge_index, events=10, seed=1)
    treated = treated_windows(sets["test"], disruptions)
    untreated = untreated_windows(sets["test"], disruptions)

    for i in range(len(treated)):
        assert torch.equal(treated[i]["x"], untreated[i]["x"])
        assert torch.equal(treated[i]["x_mask"], untreated[i]["x_mask"])
    # And the targets must differ, or the injection did nothing at all.
    assert any(not torch.equal(treated[i]["y"], untreated[i]["y"]) for i in range(len(treated)))


def test_differencing_the_two_arms_is_degenerate_by_construction(built):
    """A guard against PR #9 reporting a perfect score.

    Both arms share `x`, so any forecast cancels when the residuals are
    differenced and what is left is exactly the injected truth. An estimator
    scored this way reads 0.000 mph effect MAE for every model, including an
    untrained one, and has measured nothing. This test exists so that result
    is recognised as an identity rather than celebrated.
    """
    sets, edge_index = built
    disruptions = inject(sets["test"], edge_index, events=10, seed=13)
    model = build_model("gcn", sets["train"], hidden=8, dropout=0.0)

    treated = effects(model, treated_windows(sets["test"], disruptions), edge_index)
    untreated = effects(model, untreated_windows(sets["test"], disruptions), edge_index)
    merged = treated.merge(untreated, on=["origin", "minutes", "road"], suffixes=("_t", "_u"))
    joined = merged.merge(
        disruptions.truth[["origin", "minutes", "road", "true_effect"]],
        on=["origin", "minutes", "road"],
    )

    assert len(joined) == len(disruptions.truth)
    assert np.allclose(joined.effect_t - joined.effect_u, joined.true_effect, atol=1e-3)


def test_truth_labels_exactly_the_cells_that_moved(built):
    """Each label matches its cell, and no cell changed without a label. A truth
    table that disagrees with its grid would score every estimator wrong."""
    sets, edge_index = built
    disruptions = inject(sets["test"], edge_index, events=10, seed=2)

    for row in disruptions.truth.itertuples():
        bin_, road = cell_of(row)
        assert disruptions.values[bin_, road] == pytest.approx(row.treated, abs=1e-5)
        assert disruptions.untreated[bin_, road] == pytest.approx(row.untreated, abs=1e-5)
        assert row.true_effect == pytest.approx(row.treated - row.untreated, abs=1e-5)

    # NaN != NaN, so compare only cells that were observed to begin with.
    differs = np.isfinite(disruptions.untreated) & (disruptions.values != disruptions.untreated)
    changed = {(int(b), int(r)) for b, r in np.argwhere(differs)}
    labelled = {cell_of(row) for row in disruptions.truth.itertuples() if row.true_effect != 0}
    assert changed == labelled


def test_events_never_share_a_cell(built):
    """Overlapping disruptions would superpose and make the truth ambiguous."""
    sets, edge_index = built
    disruptions = inject(sets["test"], edge_index, events=40, seed=3)
    cells = [cell_of(row) for row in disruptions.truth.itertuples()]
    assert len(cells) == len(set(cells))
    assert disruptions.truth.event_id.nunique() == len(disruptions.events)


def test_effects_shrink_with_hops_and_follow_the_time_profile(built):
    """Spatial spread and the decay shape both have to show up in the labels,
    or a recovered hop profile cannot be compared against anything."""
    sets, edge_index = built
    disruptions = inject(
        sets["test"], edge_index, events=30, hops=2, hop_decay=0.45, duration=1.0, seed=4
    )
    by_hop = disruptions.truth.groupby("hop").requested_effect.mean()
    assert by_hop[0] < by_hop[1] < by_hop[2] < 0

    direct = disruptions.truth.loc[disruptions.truth.hop == 0]
    over_time = direct.groupby("minutes").requested_effect.mean()
    # `decay` fades, so each step is nearer zero than the last.
    assert list(over_time) == sorted(over_time)


def test_plateau_holds_full_strength_then_stops(built):
    """The other half of the shape family: nothing is injected past `duration`."""
    sets, edge_index = built
    disruptions = inject(sets["test"], edge_index, events=10, shape="plateau", duration=2, seed=5)
    # The off-period is labelled zero, not dropped. Dropping it meant a model
    # that kept predicting a slowdown after the plateau ended was never scored
    # on the half of the profile this shape exists to test.
    direct = disruptions.truth.loc[disruptions.truth.hop == 0]
    assert set(direct["minutes"]) == {BIN_MINUTES, 2 * BIN_MINUTES, 3 * BIN_MINUTES}
    assert (direct.loc[direct.minutes == 3 * BIN_MINUTES, "true_effect"] == 0.0).all()
    assert (direct.loc[direct.minutes == BIN_MINUTES, "true_effect"] < 0).all()

    assert list(effect_profile("plateau", 3, 2)) == [1.0, 1.0, 0.0]
    assert effect_profile("decay", 3, 1.0)[1] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        effect_profile("ramp", 3, 1.0)


def test_injection_repeats_under_a_seed(built):
    """Recorded seeds have to reproduce the benchmark exactly."""
    sets, edge_index = built
    again = inject(sets["test"], edge_index, events=10, seed=7)
    same = inject(sets["test"], edge_index, events=10, seed=7)
    other = inject(sets["test"], edge_index, events=10, seed=8)

    pd.testing.assert_frame_equal(again.truth, same.truth)
    assert again.settings == same.settings
    assert not again.events.equals(other.events)


def test_congested_placement_lands_on_slower_roads(built):
    """The selection stress test has to actually select. If `congested` drew the
    same roads as `uniform`, comparing the two would prove nothing about
    whether the matched estimator survives selection on pre-event speed."""
    sets, edge_index = built
    data = sets["test"]
    grid = data.values.numpy()

    def mean_pre_event(disruptions) -> float:
        return float(
            np.mean(
                [
                    _pre_mean(grid[origin - HISTORY + 1 : origin + 1, road])
                    for origin, road in zip(
                        disruptions.events.origin, disruptions.events.road, strict=True
                    )
                ]
            )
        )

    slow = inject(data, edge_index, events=20, placement="congested", seed=9)
    even = inject(data, edge_index, events=20, placement="uniform", seed=9)
    assert mean_pre_event(slow) < mean_pre_event(even)

    with pytest.raises(ValueError):
        inject(data, edge_index, events=5, placement="rush-hour", seed=9)


def test_the_floor_is_recorded_as_the_realised_effect(built):
    """A slowdown clipped at 0 mph must be labelled with what happened to the
    data, not with what was requested."""
    sets, edge_index = built
    disruptions = inject(sets["test"], edge_index, events=5, magnitude=(500.0, 500.0), seed=11)
    assert (disruptions.truth.treated == 0.0).all()
    assert (disruptions.truth.true_effect > disruptions.truth.requested_effect).all()
    assert np.allclose(
        disruptions.truth.true_effect,
        disruptions.truth.treated - disruptions.truth.untreated,
    )


def test_matched_controls_are_held_to_the_real_crash_rule(wide):
    """A synthetic control must clear the same bar as a real crash's control, or
    the benchmark scores an easier estimator than the study actually has."""
    sets, edge_index = wide
    test = sets["test"]
    fitted = fitted_cells(sets["train"])
    disruptions = inject(test, edge_index, events=10, seed=21)
    pairs, windows = matched_controls(test, disruptions, fitted, controls=3, seed=21)

    assert len(pairs)
    slot = test.slot.numpy()
    clean = test.clean.numpy()
    origin_of = dict(zip(disruptions.events.event_id, disruptions.events.origin, strict=True))
    for row in pairs.itertuples():
        origin, t_, road = int(origin_of[row.event_id]), int(row.origin), int(row.road)
        assert slot[t_] == slot[origin], "a control must sit at the same slot of the week"
        assert abs(t_ - origin) >= HISTORY + HORIZON, "a control must not overlap its event"
        assert not fitted[t_ + 1 : t_ + 1 + HORIZON, road].any(), "control target was fitted on"
        assert clean[t_ - HISTORY + 1 : t_ + HORIZON + 1, road].all(), "control span not crash-free"

    # Neighbour roads get controls too, or hop effects have nothing to correct
    # against and only the crash road could be scored.
    assert set(pairs.road) > set(disruptions.events.road)
    assert pairs.groupby(["event_id", "road"]).size().max() <= 3


def test_controls_read_the_untreated_grid_and_do_not_share_event_inputs(wide):
    """Unlike `untreated_windows`, a control must not share the event's inputs.

    That is what stops the correction collapsing into the identity: if the
    prediction cancelled here too, the corrected estimate would be the truth.
    """
    sets, edge_index = wide
    test = sets["test"]
    disruptions = inject(test, edge_index, events=10, seed=22)
    _, windows = matched_controls(
        test, disruptions, fitted_cells(sets["train"]), controls=3, seed=22
    )
    assert np.shares_memory(windows.values.numpy(), disruptions.untreated)
    assert not np.array_equal(windows.values.numpy(), disruptions.values)

    treated = treated_windows(test, disruptions)
    event_inputs = {treated[i]["x"].numpy().tobytes() for i in range(len(treated))}
    control_inputs = {windows[i]["x"].numpy().tobytes() for i in range(len(windows))}
    assert not (event_inputs & control_inputs)


def test_matched_controls_repeat_under_a_seed(wide):
    sets, edge_index = wide
    test, fitted = sets["test"], fitted_cells(sets["train"])
    disruptions = inject(test, edge_index, events=10, seed=23)
    again, _ = matched_controls(test, disruptions, fitted, controls=3, seed=23)
    same, _ = matched_controls(test, disruptions, fitted, controls=3, seed=23)
    other, _ = matched_controls(test, disruptions, fitted, controls=3, seed=24)
    pd.testing.assert_frame_equal(again, same)
    assert not again.equals(other)


def test_control_corrected_subtracts_the_mean_control_residual():
    """Hand-built arithmetic, so a sign slip or a bad join cannot hide."""
    event = pd.DataFrame(
        {
            "event_id": ["A", "A", "B"],
            "road": [1, 2, 1],
            "minutes": [10, 10, 10],
            "effect": [-5.0, -2.0, -4.0],
        }
    )
    control = pd.DataFrame(
        {
            "origin": [50, 60, 70],
            "road": [1, 1, 2],
            "minutes": [10, 10, 10],
            "effect": [-1.0, -3.0, 0.5],
        }
    )
    pairs = pd.DataFrame({"event_id": ["A", "A", "A"], "road": [1, 1, 2], "origin": [50, 60, 70]})
    out = control_corrected(event, control, pairs).set_index(["event_id", "road"])
    # Road 1 of A: bias is the mean of -1 and -3, so -5 - (-2) = -3.
    assert out.loc[("A", 1), "estimated"] == pytest.approx(-3.0)
    assert out.loc[("A", 2), "estimated"] == pytest.approx(-2.5)
    # B has no matched control, so it is dropped rather than left uncorrected.
    assert "B" not in out.index.get_level_values("event_id")


def test_effect_recovery_interval_is_event_clustered():
    """Two events, five identical cells each. The interval has to be set by the
    two events, not the ten cells, or a recovery claim gets a precision the
    benchmark cannot support."""
    estimate = pd.DataFrame(
        {
            "event_id": np.repeat(["A", "B"], 5),
            "road": np.tile([1, 2, 3, 4, 5], 2),
            "minutes": 10,
            "estimated": np.repeat([-3.0, -1.0], 5),
        }
    )
    truth = estimate.assign(hop=0, true_effect=-2.0)[
        ["event_id", "road", "minutes", "hop", "true_effect"]
    ]
    row = effect_recovery(estimate, truth).query("group == 'pooled'").iloc[0]

    assert row["events"] == 2 and row["n"] == 10
    # Per-event errors are -1 and +1, so the bias is zero and the spread is 1.
    assert row["bias_mph"] == pytest.approx(0.0)
    clustered = 1.96 * float(pd.Series([-1.0, 1.0]).sem())
    assert row["bias_high"] == pytest.approx(clustered, abs=1e-3)
    assert row["bias_low"] == pytest.approx(-clustered, abs=1e-3)
    # Ten copies of two numbers must not buy a narrower interval than two.
    naive = 1.96 * float((estimate["estimated"] + 2.0).sem())
    assert clustered > naive

    # A single event cannot support an interval, and must say so rather than
    # reporting a zero-width one.
    alone = effect_recovery(estimate[estimate.event_id == "A"], truth[truth.event_id == "A"]).iloc[
        0
    ]
    assert alone["bias_low"] is None and alone["bias_high"] is None


def test_recovery_columns_compose_under_unequal_event_sizes():
    """`estimated_mph - true_mph` must equal `bias_mph` exactly.

    Both existing recovery tests give every event the same number of cells,
    which is precisely when cell-weighting and event-weighting agree, so they
    could not see the columns drifting apart. Here one event carries four cells
    and the other one.
    """
    estimate = pd.DataFrame(
        {
            "event_id": ["A", "A", "A", "A", "B"],
            "road": [1, 2, 3, 4, 1],
            "minutes": 10,
            "estimated": [-4.0, -4.0, -4.0, -4.0, 0.0],
        }
    )
    truth = estimate.assign(hop=0, true_effect=-2.0)[
        ["event_id", "road", "minutes", "hop", "true_effect"]
    ]
    row = effect_recovery(estimate, truth).query("group == 'pooled'").iloc[0]

    assert row["estimated_mph"] - row["true_mph"] == pytest.approx(row["bias_mph"], abs=1e-9)
    # Event-weighted: A averages -4 and B is 0, so the mean is -2, not the
    # cell-weighted -3.2 that four A cells against one B cell would give.
    assert row["estimated_mph"] == pytest.approx(-2.0)
    assert row["bias_mph"] == pytest.approx(0.0)


def test_effect_recovery_reports_pooled_hop_and_horizon():
    """The hop and horizon rows are the point: a pooled number alone cannot say
    whether the spatial or time profile was recovered."""
    estimate = pd.DataFrame(
        {
            "event_id": ["A", "A", "A", "A"],
            "road": [1, 1, 2, 2],
            "minutes": [10, 20, 10, 20],
            "estimated": [-4.0, -2.0, -1.0, -1.0],
        }
    )
    truth = pd.DataFrame(
        {
            "event_id": ["A", "A", "A", "A"],
            "road": [1, 1, 2, 2],
            "minutes": [10, 20, 10, 20],
            "hop": [0, 0, 1, 1],
            "true_effect": [-5.0, -2.5, -1.5, -0.5],
        }
    )
    table = effect_recovery(estimate, truth).set_index(["group", "value"])

    assert table.loc[("pooled", "all"), "n"] == 4
    # Errors are +1.0, +0.5, +0.5, -0.5: mean bias +0.375, MAE 0.625.
    assert table.loc[("pooled", "all"), "bias_mph"] == pytest.approx(0.375)
    assert table.loc[("pooled", "all"), "effect_mae"] == pytest.approx(0.625)
    assert table.loc[("hop", 0), "effect_mae"] == pytest.approx(0.75)
    assert table.loc[("hop", 1), "effect_mae"] == pytest.approx(0.5)
    assert table.loc[("minutes", 10), "n"] == 2
    assert table.loc[("hop", 0), "true_mph"] == pytest.approx(-3.75)

    with pytest.raises(ValueError, match="no estimated cell"):
        effect_recovery(estimate.assign(event_id="Z"), truth)


@pytest.fixture(scope="module")
def wide_data(tmp_path_factory):
    """Mock tables on disk, long enough for `matched_controls` to find controls."""
    path = tmp_path_factory.mktemp("wide_data")
    for name, table in make_mock(rows=3, cols=3, days=60, crashes=40, seed=0).items():
        table.to_parquet(path / f"{name}.parquet", index=False)
    return path


def test_benchmark_script_scores_both_arms_on_the_same_cells(wide_data, tmp_path):
    """The library functions above are checked by hand; this runs the wiring.

    Every finding worth having in `scripts/benchmark.py` was in `main()` --
    the per-model seeding, the two estimators drifting onto different samples,
    and a mis-divided control count written into the run record -- so the
    script needs one test that actually executes it.
    """
    spec = importlib.util.spec_from_file_location(
        "benchmark", Path(__file__).resolve().parents[1] / "scripts" / "benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    out = tmp_path / "bench"
    recovery = module.main(
        [
            "--data",
            str(wide_data),
            "--out",
            str(out),
            "--models",
            "gcn",
            "--events",
            "12",
            "--epochs",
            "2",
            "--patience",
            "1",
            "--seed",
            "7",
        ]
    )

    for name in ("effect_recovery.csv", "forecast.csv", "truth.csv", "controls.csv"):
        assert (out / name).exists(), f"{name} was not written"

    # Reproducibility: seed, arguments and data version all recorded.
    summary = json.loads((out / "summary.json").read_text())
    assert summary["args"]["seed"] == "7"
    assert summary["benchmark"]["seed"] == 7
    assert summary["code"] and summary["data"]["rows"]["speeds"]

    # The two estimators must describe the same cells, or their gap is a
    # different sample rather than the bias correction.
    pooled = recovery[recovery.group == "pooled"].set_index("estimator")
    assert pooled.loc["corrected", "n"] == pooled.loc["raw", "n"]
    assert set(recovery.estimator) == {"corrected", "raw"}
    assert {"pooled", "hop", "minutes"} <= set(recovery.group)

    # Cohort loss from missing controls is recorded rather than silent.
    assert summary["cells_after_correction"] <= summary["cells_before_correction"]
    # Per event *road*, so it cannot exceed what was requested per road.
    assert 0 < summary["controls_per_event_road"] <= int(summary["args"]["controls"])
    assert summary["event_roads"] >= summary["benchmark"]["events"]


def test_a_zero_decay_injection_is_the_synthetic_null(built):
    """Benchmark 1. `inject` labels a zero-effect ring instead of skipping it,
    so the outer rings arrive as truth rows and need no separate generator."""
    sets, edge_index = built
    disruptions = inject(sets["test"], edge_index, events=10, hops=2, hop_decay=0.0, seed=13)
    truth = disruptions.truth

    assert set(truth.hop) == {0, 1, 2}
    assert (truth.loc[truth.hop == 0, "true_effect"] < 0).all()
    assert (truth.loc[truth.hop > 0, "true_effect"] == 0.0).all()

    moved = ~np.isclose(disruptions.values, disruptions.untreated, equal_nan=True)
    assert set(zip(*np.nonzero(moved), strict=True)) == set(
        truth[truth.hop == 0].apply(cell_of, axis=1)
    )
