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
from traffic_gml.events import (
    ABLATIONS,
    FEATURES,
    GRAPH_FEATURES,
    ablate_edges,
    build_windows,
    clock_features,
)
from traffic_gml.graph import neighbourhood
from traffic_gml.mock import make_mock
from traffic_gml.models import build_model

HISTORY, HORIZON = 6, 3
STATIC = ["length_m", "degree"]


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


def test_crash_contaminated_inputs_are_never_unmasked(built):
    """The sibling rule, on the input side. Masking only `y` let a clean target
    read a crashed neighbour inside the receptive field, which puts a real
    slowdown into the no-crash counterfactual and reaches the graph models but
    never the edgeless `mlp`. A contaminated input is masked and zeroed."""
    seen = 0
    for data in built[0].values():
        dirty = ~data.clean
        for i in range(0, len(data), max(len(data) // 20, 1)):
            item = data[i]
            t = int(item["origin"])
            span = dirty[t - HISTORY + 1 : t + 1]
            seen += int(span.sum())
            assert not item["x_mask"][span].any()
            assert (item["x"][..., 0][span] == 0.0).all()
    assert seen, "no sampled window held a contaminated input cell, so this proves nothing"


def test_dropping_zero_speed_changes_only_the_zero_cells():
    """0.0 mph is either a real standstill or a second sentinel; the loader drops
    negatives but keeps zeros. At the midpoint 164 such rows carried 31% of the
    pooled effect, so the flag has to move those cells and nothing else."""
    ids = np.arange(1, 4)
    times = pd.date_range("2025-01-06", periods=120, freq=FREQ)
    speed = np.full((120, 3), 30.0, dtype="float32")
    speed[10, 0] = 0.0
    speed[11, 1] = 0.0
    tables = {
        "speeds": pd.DataFrame(
            {
                "time": np.repeat(times, 3),
                "segment_id": np.tile(ids, 120),
                "speed": speed.ravel(),
            }
        ),
        "segments": pd.DataFrame({"segment_id": ids}),
        "edges": pd.DataFrame({"src": ids[:-1], "dst": ids[1:]}),
        "crashes": pd.DataFrame(
            {"crash_record_id": ["C"], "crash_time": [times[100]], "segment_id": [3]}
        ),
    }
    kept, _ = build_windows(tables, history=3, horizon=2)
    dropped, _ = build_windows(tables, history=3, horizon=2, drop_zero_speed=True)

    assert int((kept["train"].values == 0.0).sum()) == 2
    assert not bool((dropped["train"].values == 0.0).any())
    assert torch.isnan(dropped["train"].values[10, 0])
    assert torch.isnan(dropped["train"].values[11, 1])
    # Every other cell is untouched, so the flag is not quietly dropping observations.
    assert int(torch.isfinite(dropped["train"].values).sum()) == int(
        torch.isfinite(kept["train"].values).sum() - 2
    )


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


def test_mlp_and_gcn_differ_only_by_the_graph(built, batch):
    sets, edge_index = built

    def params(name):
        return sum(p.numel() for p in build_model(name, sets["train"], hidden=8).parameters())

    assert params("mlp") == params("gcn")

    torch.manual_seed(0)
    graph = build_model("gcn", sets["train"], hidden=8, dropout=0.0).eval()
    empty = torch.zeros((2, 0), dtype=torch.long)
    with torch.no_grad():
        args = (batch["x"], batch["x_mask"])
        assert not torch.allclose(graph(*args, edge_index), graph(*args, empty))


def test_stgcn_and_tgru_differ_only_by_the_graph(built, batch):
    """The spatiotemporal pair has to be matched the way mlp/gcn is, or its gap
    measures the temporal encoder and capacity instead of connectivity."""
    sets, edge_index = built

    def params(name):
        return sum(p.numel() for p in build_model(name, sets["train"], hidden=8).parameters())

    assert params("tgru") == params("stgcn")

    empty = torch.zeros((2, 0), dtype=torch.long)
    args = (batch["x"], batch["x_mask"])
    torch.manual_seed(0)
    graph = build_model("stgcn", sets["train"], hidden=8, dropout=0.0).eval()
    torch.manual_seed(0)
    plain = build_model("tgru", sets["train"], hidden=8, dropout=0.0).eval()
    with torch.no_grad():
        assert graph(*args, edge_index).shape == batch["y"].shape
        assert not torch.allclose(graph(*args, edge_index), graph(*args, empty))
        # The no-edge twin must be inert to the graph, or the pair is not controlled.
        assert torch.allclose(plain(*args, edge_index), plain(*args, empty))


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


def test_graph_ablation_holds_capacity_fixed_and_only_moves_the_topology(built, batch):
    """PR #11's controlled test. A graph model can beat a non-graph one just by
    averaging over *any* neighbours, so the informative ablation keeps the edge
    count and degree sequence and scrambles only which road is which. If that
    changed the amount of message passing, a loss under it would be ambiguous.
    """
    sets, edge_index = built
    roads = sets["train"].num_roads

    assert ablate_edges(edge_index, roads, "real") is edge_index
    none = ablate_edges(edge_index, roads, "none")
    assert none.size(1) == 0

    permuted = ablate_edges(edge_index, roads, "permuted", seed=0)
    assert permuted.size(1) == edge_index.size(1), "edge count must not change"
    assert permuted.dtype == edge_index.dtype
    before = np.bincount(edge_index[0].numpy(), minlength=roads)
    after = np.bincount(permuted[0].numpy(), minlength=roads)
    assert sorted(before.tolist()) == sorted(after.tolist()), "degree sequence must survive"
    # The assignment really is scrambled, and reproducible.
    assert not torch.equal(permuted, edge_index)
    assert torch.equal(permuted, ablate_edges(edge_index, roads, "permuted", seed=0))
    assert not torch.equal(permuted, ablate_edges(edge_index, roads, "permuted", seed=1))

    # And it reaches the model: a graph forecaster must give a different answer.
    torch.manual_seed(0)
    model = build_model("gcn", sets["train"], hidden=8, dropout=0.0).eval()
    with torch.no_grad():
        args = (batch["x"], batch["x_mask"])
        assert not torch.allclose(model(*args, edge_index), model(*args, permuted))
        # `none` is the roadmap's floor arm, so it has to actually run: GCNConv
        # survives an empty edge_index only because it adds self-loops.
        assert torch.isfinite(model(*args, none)).all()

    with pytest.raises(ValueError, match="unknown graph ablation"):
        ablate_edges(edge_index, roads, "rewired")
    assert set(ABLATIONS) == {"real", "permuted", "none"}
    # `degree` and `betweenness` are graph-derived, so an ablation leaves them
    # describing a graph the model no longer has. Both entry points warn.
    assert set(GRAPH_FEATURES) == {"degree", "betweenness"}


def test_neighbour_channel_summarises_only_clean_observed_neighbours(tables):
    """The fair non-graph baseline needs neighbour speeds, but not crashed ones.

    A crashed neighbour inside the summary would put a real slowdown into the
    no-crash counterfactual, which is the leak `x_mask` exists to stop.
    """
    plain, _ = build_windows(tables, history=HISTORY, horizon=HORIZON, static_features=STATIC)
    with_nearby, edge_index = build_windows(
        tables, history=HISTORY, horizon=HORIZON, static_features=STATIC, neighbour_hops=1
    )
    base, data = plain["test"], with_nearby["test"]
    assert base.num_features == len(FEATURES) + len(STATIC)
    assert data.num_features == base.num_features + 1

    item = data[0]
    assert item["x"].shape[-1] == data.num_features
    assert torch.equal(item["x"][..., :-1], base[0]["x"])

    reach = neighbourhood(edge_index, data.num_roads, 1).toarray()
    lag = slice(int(data.origins[0]) - HISTORY + 1, int(data.origins[0]) + 1)
    speeds, clean = data.values[lag].numpy(), data.clean[lag].numpy()
    usable = np.isfinite(speeds) & clean
    for road in range(data.num_roads):
        nearby = reach[road]
        seen = usable[:, nearby]
        expected = np.where(
            seen.any(axis=1),
            np.where(seen, speeds[:, nearby], 0.0).sum(axis=1) / np.maximum(seen.sum(axis=1), 1),
            np.nan,
        )
        channel = item["x"][:, road, -1].numpy()
        reported = ~np.isnan(expected)
        assert np.allclose(channel[reported], data.encode(expected[reported]), atol=1e-5)
        assert (channel[~reported] == 0.0).all()
