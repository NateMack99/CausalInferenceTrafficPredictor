"""Offline checks for the exposure-conditioned outcome models (workstream C)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from traffic_gml.data import FREQ
from traffic_gml.evaluate import exposure_effects
from traffic_gml.events import EXPOSURES, build_windows
from traffic_gml.mock import make_mock
from traffic_gml.models import build_model

HISTORY, HORIZON = 4, 2


def path_tables(crash_segments, crash_bins):
    """A 7-segment path graph with crashes placed on named segments."""
    ids = np.arange(1, 8)
    times = pd.date_range("2025-01-06", periods=400, freq=FREQ, tz="UTC")
    return {
        "speeds": pd.DataFrame(
            {
                "time": np.repeat(times, 7),
                "segment_id": np.tile(ids, 400),
                "speed": np.full(400 * 7, 30.0, dtype="float32"),
            }
        ),
        "segments": pd.DataFrame({"segment_id": ids}),
        "edges": pd.DataFrame({"src": ids[:-1], "dst": ids[1:]}),
        "crashes": pd.DataFrame(
            {
                "crash_record_id": [f"C{i}" for i in range(len(crash_segments))],
                "crash_time": [times[b] for b in crash_bins],
                "segment_id": crash_segments,
            }
        ),
    }


def test_exposure_labels_take_the_nearest_crash():
    """Segment 1 is two hops from a crash on 3 and on its own road for a crash
    on 1, so the overlap must resolve to `own` rather than `hop2` or both."""
    sets, _ = build_windows(
        path_tables([3, 1], [200, 200]), history=HISTORY, horizon=HORIZON, label_exposure=True
    )
    labels = sets["train"].exposure[200].numpy()
    # Nodes are 0-based over sorted segment ids, so segment 3 is node 2.
    assert labels[0] == EXPOSURES.index("own")
    assert labels[2] == EXPOSURES.index("own")
    assert labels[1] == EXPOSURES.index("hop1")
    assert labels[3] == EXPOSURES.index("hop1")
    assert labels[4] == EXPOSURES.index("hop2")
    assert labels[6] == EXPOSURES.index("none")


def test_factual_cohort_scores_the_exposed_cells():
    """The forecasting sets drop exposed roads, so the exposed heads would
    otherwise never see a gradient."""
    sets, _ = build_windows(
        path_tables([4], [200]), history=HISTORY, horizon=HORIZON, label_exposure=True
    )
    forecasting = sets["train"]
    factual = forecasting.factual()
    window = int((forecasting.origins == 199).nonzero().flatten()[0])

    exposed = factual[window]["exposure"] > 0
    assert exposed.any()
    assert not (forecasting[window]["y_mask"] & exposed).any()
    assert (factual[window]["y_mask"] & exposed).any()


def test_outcomes_ignore_exposure_and_the_effect_is_a_head_contrast():
    """Exposure enters at the head only, so relabelling a cell cannot change
    the trunk's prediction for any state. That is what keeps the counterfactual
    free of post-treatment information."""
    tables = make_mock(rows=3, cols=3, days=10, crashes=20, seed=0)
    sets, edge_index = build_windows(tables, history=HISTORY, horizon=HORIZON, label_exposure=True)
    data = sets["test"].factual()
    model = build_model("treatment_gcn", data, hidden=8, dropout=0.0).eval()
    batch = next(iter(DataLoader(data, batch_size=4)))

    none = EXPOSURES.index("none")
    with torch.no_grad():
        outcomes = model.outcomes(batch["x"], batch["x_mask"], edge_index)
        factual = model(batch["x"], batch["x_mask"], edge_index, batch["exposure"])
        relabelled = model(
            batch["x"], batch["x_mask"], edge_index, torch.full_like(batch["exposure"], none)
        )
    assert outcomes.shape == (4, len(EXPOSURES), HORIZON, data.num_roads)
    assert torch.equal(factual, outcomes.gather(1, batch["exposure"].unsqueeze(1)).squeeze(1))
    # Relabelling reads another head; it does not change what the trunk computed.
    assert torch.equal(relabelled, outcomes[:, none])

    frame = exposure_effects(model, data, edge_index)
    assert np.allclose(frame.effect, frame.factual - frame.untreated, atol=1e-4)
    assert (frame.loc[frame.exposure == none, "effect"] == 0.0).all()
