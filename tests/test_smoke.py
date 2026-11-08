"""Offline checks for the properties the CLI output cannot show: the frozen
shapes in `docs/architecture.md`, the leakage and masking rules in
`AGENTS.md`, and one reproducible end-to-end run.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from traffic_gml import train
from traffic_gml.data import FREQ, align_grid
from traffic_gml.events import FEATURES, build_windows, clock_features
from traffic_gml.mock import make_mock
from traffic_gml.models import BASELINES, CONVS, build_model, predict

HISTORY, HORIZON = 6, 3
STATIC = ["length", "degree"]


@pytest.fixture(scope="module")
def tables():
    return make_mock(rows=3, cols=3, days=8, crashes=20, seed=0)


@pytest.fixture(scope="module")
def built(tables):
    return build_windows(tables, history=HISTORY, horizon=HORIZON, static_features=STATIC)


@pytest.fixture(scope="module")
def batch(built):
    return next(iter(DataLoader(built[0]["test"], batch_size=4)))


@pytest.fixture(scope="module")
def data_dir(tables, tmp_path_factory):
    path = tmp_path_factory.mktemp("mock")
    for name, table in tables.items():
        table.to_parquet(path / f"{name}.parquet", index=False)
    return path


def test_batch_matches_the_frozen_model_contract(built, batch):
    data, edge_index = built[0]["test"], built[1]
    roads, features = data.num_roads, len(FEATURES) + len(STATIC)
    assert data.num_features == features
    assert batch["x"].shape == (4, HISTORY, roads, features)
    assert batch["x_mask"].shape == (4, HISTORY, roads)
    assert batch["y"].shape == (4, HORIZON, roads)
    assert batch["y_mask"].shape == (4, HORIZON, roads)
    assert edge_index.shape[0] == 2 and edge_index.dtype == torch.long
    assert batch["x_mask"].dtype == torch.bool and batch["y_mask"].dtype == torch.bool


def test_features_use_only_the_window_and_flag_gaps_instead_of_zeroing_them(tables, built):
    data = built[0]["train"]
    segment_ids = np.sort(tables["segments"]["segment_id"].unique())
    values, times = align_grid(tables["speeds"], segment_ids)
    item = data[0]
    origin = int(item["origin"])

    window = values[origin - HISTORY + 1 : origin + 1]
    seen = np.isfinite(window)
    np.testing.assert_allclose(item["x_mask"].numpy(), seen)
    expected = np.where(seen, (window - data.mean) / data.std, 0.0)
    np.testing.assert_allclose(item["x"][..., 0].numpy(), expected, atol=1e-5)

    clock = clock_features(times)[origin - HISTORY + 1 : origin + 1]
    np.testing.assert_allclose(item["x"][:, 0, 1:5].numpy(), clock, atol=1e-6)

    # Absolute, not self-referential: a UTC bin must clock as local wall time,
    # or every rush-hour feature silently sits hours out of phase.
    noon = clock_features(pd.date_range("2025-06-02 13:00", periods=1, freq=FREQ, tz="UTC"))
    assert noon[0, 0] == pytest.approx(np.sin(2 * np.pi * 8 * 60 / 1440), abs=1e-6)
    assert torch.isfinite(item["x"]).all() and torch.isfinite(item["y"]).all()


def test_crash_contaminated_targets_are_never_unmasked(built):
    """One contaminated cell anywhere in a road's lag or target span must take all
    of that road's targets out of the window. The radius itself is checked below."""
    for data in built[0].values():
        dirty = ~data.clean
        assert dirty.any(), "the mock should contain crashes"
        for i in range(0, len(data), max(len(data) // 20, 1)):
            item = data[i]
            t = int(item["origin"])
            span = dirty[t - HISTORY + 1 : t + 1 + HORIZON].any(dim=0)
            assert not item["y_mask"][:, span].any()


def test_crash_masking_spreads_exactly_exposure_hops():
    """A hand-built path graph. The old check only covered 0 hops, so a wrong
    radius, a wrong direction, or an off-by-one in the hop loop all passed."""
    ids = np.arange(1, 8)
    times = pd.date_range("2025-01-06", periods=400, freq=FREQ, tz="UTC")
    speeds = pd.DataFrame(
        {
            "time": np.repeat(times, 7),
            "segment_id": np.tile(ids, 400),
            "speed": np.full(400 * 7, 30.0, dtype="float32"),
        }
    )
    tables = {
        "speeds": speeds,
        "segments": pd.DataFrame({"segment_id": ids}),
        "edges": pd.DataFrame({"src": ids[:-1], "dst": ids[1:]}),
        "crashes": pd.DataFrame(
            {"crash_record_id": ["C"], "crash_time": [times[200]], "segment_id": [4]}
        ),
    }
    # Segment 4 is node 3; edges are directed but contamination spreads both ways.
    for hops, expected in [(0, {3}), (1, {2, 3, 4}), (2, {1, 2, 3, 4, 5})]:
        sets, _ = build_windows(tables, history=3, horizon=2, exposure_hops=hops)
        hit = (~sets["train"].clean[200]).nonzero().flatten().tolist()
        assert set(hit) == expected, f"{hops} hops masked {hit}, expected {sorted(expected)}"


def test_splits_are_chronological_and_gapped(built):
    sets = built[0]
    first, middle, last = (sets[name].origins.numpy() for name in ("train", "val", "test"))
    assert first.max() < middle.min() <= middle.max() < last.min()
    assert middle.min() - first.max() > HISTORY + HORIZON
    assert last.min() - middle.max() > HISTORY + HORIZON


@pytest.mark.parametrize("name", [*BASELINES, *CONVS])
def test_every_model_predicts_the_target_shape(name, built, batch):
    sets, edge_index = built
    model = build_model(name, sets["train"], hidden=8)
    train.fit(model, sets, edge_index, epochs=1, verbose=False)
    out = predict(model, batch, edge_index)
    assert out.shape == batch["y"].shape and torch.isfinite(out).all()


def test_mlp_and_gcn_differ_only_by_the_graph(built, batch):
    sets, edge_index = built

    def params(name):
        return sum(p.numel() for p in build_model(name, sets["train"], hidden=8).parameters())

    assert params("mlp") == params("gcn")

    torch.manual_seed(0)
    graph = build_model("gcn", sets["train"], hidden=8, dropout=0.0).eval()
    plain = build_model("mlp", sets["train"], hidden=8, dropout=0.0).eval()
    empty = torch.zeros((2, 0), dtype=torch.long)
    with torch.no_grad():
        args = (batch["x"], batch["x_mask"])
        assert not torch.allclose(graph(*args, edge_index), graph(*args, empty))
        torch.testing.assert_close(plain(*args, edge_index), plain(*args, empty))


def test_run_records_seed_arguments_data_version_and_checkpoint(data_dir, tmp_path):
    argv = ["--data", str(data_dir), "--out", str(tmp_path), "--model", "sage", "--seed", "7"]
    argv += ["--history", str(HISTORY), "--horizon", str(HORIZON)]
    argv += ["--hidden", "16", "--epochs", "2"]
    metrics = train.main(argv)
    assert set(metrics) == {"train", "val", "test"}
    assert metrics["test"]["mae"] > 0
    assert set(metrics["test"]["mae_by_minutes"]) == {"10", "20", "30"}

    run = tmp_path / "sage-seed7"
    recorded = json.loads((run / "metrics.json").read_text())
    assert json.loads((run / "args.json").read_text())["seed"] == 7
    assert recorded["data"]["rows"]["speeds"] > 0
    assert (run / "model.pt").exists()
