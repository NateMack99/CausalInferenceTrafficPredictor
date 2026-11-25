"""The settings that produce the paper.

Every cluster stage passes its settings explicitly, so an argparse default
cannot change a Run A result. These tests pin both halves: the flags
`experiments.sh` sends, and the defaults a bare local run inherits. Either one
drifting from the frozen baseline command in `docs/experiments.md` fails here.
"""

import importlib.util
import re
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = (ROOT / "scripts/hpc/experiments.sh").read_text()

WINDOW = {"history": 12, "horizon": 6, "stride": 1, "exposure_hops": 2}
TRAINING = {"epochs": 20, "patience": 4, "hidden": 64, "batch_size": 64}
FEATURES = ["length_m", "degree", "betweenness"]


def entry_point(relative: str):
    spec = importlib.util.spec_from_file_location(f"_cfg_{Path(relative).stem}", ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flags(name: str) -> list[str]:
    body = re.search(rf"^{name}=\((.*?)\)$", EXPERIMENTS, re.M | re.S).group(1)
    return shlex.split(body.replace('--data "$data"', ""))


def test_cluster_forecasting_stages_use_the_paper_settings(tmp_path):
    args = entry_point("scripts/experiment.py").parse_args(
        [*flags("window"), *flags("features"), *flags("training"), "--data", str(tmp_path)]
    )
    # `controls` and `seed` are read from the defaults here, so this also fails
    # if a default drifts away from the frozen baseline.
    assert {k: getattr(args, k) for k in {**WINDOW, **TRAINING}} == {**WINDOW, **TRAINING}
    assert args.static_features == FEATURES
    assert (args.controls, args.seed, args.effect_model) == (5, 42, "gcn")


def test_cluster_benchmark_stages_use_the_contract_intervention(tmp_path):
    args = entry_point("scripts/benchmark.py").parse_args(
        [*flags("window"), *flags("benchmark"), "--data", str(tmp_path)]
    )
    assert (args.events, args.controls, args.hops, args.hop_decay) == (200, 5, 2, 0.45)
    assert list(args.magnitude) == [4.0, 12.0]
    assert (args.shape, args.duration, args.placement) == ("decay", 2.0, "uniform")


def test_cluster_event_study_stages_use_the_contract_rings(tmp_path):
    args = entry_point("scripts/event_study.py").parse_args(
        [*flags("window"), *flags("event"), "--data", str(tmp_path)]
    )
    assert (args.hops, args.controls, args.leads) == (2, 5, 3)
    assert args.hops <= args.exposure_hops


def test_a_bare_local_run_matches_the_frozen_baseline(tmp_path):
    """The defaults used to say 30 epochs, patience 5 and one control, so a
    local run was never the paper's configuration."""
    args = entry_point("scripts/experiment.py").parse_args(["--data", str(tmp_path)])
    assert (args.epochs, args.patience, args.controls) == (20, 4, 5)
    # Graph-derived features stay opt-in: defaulting them on would silently make
    # a forgotten --static-features in an ablation arm unfair.
    assert args.static_features == []
