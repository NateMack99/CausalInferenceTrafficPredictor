"""Offline checks for the semi-synthetic benchmark.

Every number PR #9 reports is scored against `Disruptions.truth`, so these
tests are mostly about the labels agreeing with the array they describe. The
first one is the leakage rule: an injected effect must never reach an input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from traffic_gml.data import BIN_MINUTES
from traffic_gml.evaluate import effects
from traffic_gml.events import build_windows
from traffic_gml.mock import make_mock
from traffic_gml.models import build_model
from traffic_gml.synthetic import (
    _pre_mean,
    effect_profile,
    inject,
    treated_windows,
    untreated_windows,
)

HISTORY, HORIZON = 6, 3


@pytest.fixture(scope="module")
def built():
    tables = make_mock(rows=4, cols=4, days=12, crashes=20, seed=0)
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
    minutes = set(disruptions.truth.loc[disruptions.truth.hop == 0, "minutes"])
    assert minutes == {BIN_MINUTES, 2 * BIN_MINUTES}  # the third bin gets nothing

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
