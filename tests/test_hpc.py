"""Offline run preparation: audits must finish without fitting a model."""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from traffic_gml.events import write_processed
from traffic_gml.mock import make_mock


def test_data_only_audit_does_not_train(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "experiment", Path(__file__).resolve().parents[1] / "scripts/experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def no_training(*args, **kwargs):
        raise AssertionError("data-only mode trained a model")

    monkeypatch.setattr(module, "fit", no_training)
    data = write_processed(make_mock(rows=3, cols=3, days=90, crashes=60), tmp_path / "data")
    out = tmp_path / "audit"
    audit = module.main(["--data", str(data), "--out", str(out), "--data-only", "--controls", "5"])
    assert audit["crashes"] > 0 and audit["control_pairs"] > 0
    assert json.loads((out / "audit.json").read_text())["windows"] == audit["windows"]
    for name in ("monthly_coverage.csv", "neighbour_coverage.csv", "events.csv", "controls.csv"):
        assert (out / name).exists()
    assert not (out / "results.csv").exists()
    assert not (out / "summary.json").exists()


def test_submit_queues_one_resumable_dependency_graph(tmp_path):
    """Validate submissions offline; never contact the cluster from pytest."""
    binary = tmp_path / "bin"
    binary.mkdir()
    for tool in ("squeue", "sacct"):
        (binary / tool).write_text("#!/bin/sh\nexit 0\n")
        (binary / tool).chmod(0o755)
    (binary / "sinfo").write_text("#!/bin/sh\necho standard\n")
    (binary / "sinfo").chmod(0o755)
    (binary / "sbatch").write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "path = Path(os.environ['SUBMISSION_LOG'])\n"
        "calls = json.loads(path.read_text()) if path.exists() else []\n"
        "calls.append(sys.argv[1:])\n"
        "path.write_text(json.dumps(calls))\n"
        "print(90000 + len(calls))\n"
    )
    (binary / "sbatch").chmod(0o755)

    root = Path(__file__).resolve().parents[1]
    # A data directory outside the project: prepare must be told where to read.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "chicago-traffic-segments.json").write_text("{}")
    log = tmp_path / "submissions.json"
    name = f"pytest-{tmp_path.name}"
    command = ["bash", "scripts/hpc/submit.sh", "--name", name]
    env = dict(
        os.environ,
        PATH=f"{binary}:{os.environ['PATH']}",
        SUBMISSION_LOG=str(log),
        MIN_FREE_GB="0",
        GPU_GRES="gpu:test:1",
        DATA_DIR=str(elsewhere),
    )
    outputs = root / "outputs/hpc" / name
    try:
        first = subprocess.run(
            command, cwd=root, env=env, capture_output=True, text=True, check=True
        )
        # Preflight asks sbatch --test-only whether each shape is acceptable;
        # those are validations, not submissions.
        calls = [a for a in json.loads(log.read_text()) if "--test-only" not in a]
        stages = [
            next(a for a in args if a.startswith("--job-name=")).removeprefix("--job-name=traffic-")
            for args in calls
        ]
        assert stages[:3] == ["prepare", "audit", "main"]
        prepare = next(a for a, s in zip(calls, stages, strict=True) if s == "prepare")
        assert prepare[-1] == str(elsewhere), "prepare must receive the dataset directory"
        assert {"treatment", "graph-ablation", "topology-spread", "aggregate", "finish"} <= set(
            stages
        )
        memory = {
            s: next(a for a in args if a.startswith("--mem="))
            for args, s in zip(calls, stages, strict=True)
        }
        assert memory["prepare"] == "--mem=128G"
        for stage in ("audit", "report", "aggregate", "finish"):
            assert memory[stage] == "--mem=8G", stage

        for args, stage in zip(calls, stages, strict=True):
            if stage == "prepare":
                assert not any(a.startswith("--dependency=") for a in args)
                continue
            dependency = next(a for a in args if a.startswith("--dependency="))
            assert "--kill-on-invalid-dep=yes" in args
            expected = "afterany:" if stage in {"aggregate", "finish"} else "afterok:"
            assert dependency.startswith(f"--dependency={expected}"), stage
            if "--gres=gpu:test:1" in args:
                for arg in args:
                    if arg.startswith("--array="):
                        assert arg.endswith("%2"), stage

        # A second plain submission must not queue the whole matrix twice.
        again = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
        assert again.returncode != 0
        assert "--resume" in again.stderr
        submitted = [a for a in json.loads(log.read_text()) if "--test-only" not in a]
        assert len(submitted) == len(calls)

        resumed = subprocess.run(
            [*command, "--resume"], cwd=root, env=env, capture_output=True, text=True, check=True
        )
        submitted = [a for a in json.loads(log.read_text()) if "--test-only" not in a]
        assert len(submitted) == 2 * len(calls)
        assert "manifest" in resumed.stdout
        assert first.stdout
    finally:
        shutil.rmtree(outputs, ignore_errors=True)


def test_aggregate_reports_missing_stages_instead_of_inventing_them(tmp_path):
    run, work = tmp_path / "run", tmp_path / "work"
    arm = run / "graph-ablation/gcn-permuted-seed43"
    arm.mkdir(parents=True)
    (arm / "summary.json").write_text(
        json.dumps({"args": {"graph_ablation": "permuted", "seed": "43"}})
    )
    (arm / "forecast.csv").write_text("model,mae,rmse\ngcn,3.1,4.2\n")
    (arm / "effect_recovery.csv").write_text(
        "model,estimator,group,value,effect_mae,bias_mph\ngcn,corrected,pooled,all,2.5,0.3\n"
    )
    (run / ".done").mkdir()
    (run / ".done/graph-ablation-0").touch()

    out = tmp_path / "paper"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts/aggregate.py"),
            "--run",
            str(run),
            "--work",
            str(work),
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    index = pd.read_csv(out / "result_index.csv")
    assert set(index.status) == {"ready", "missing"}
    assert (index.loc[index.status == "missing", "rows"] == 0).all()

    ablation = pd.read_csv(out / "graph_ablation.csv")
    assert ablation.to_dict("records") == [
        {
            "model": "gcn",
            "topology": "permuted",
            "seed": 43,
            "forecast_mae": 3.1,
            "effect_mae": 2.5,
            "effect_bias": 0.3,
        }
    ]
    assert not (out / "treatment.csv").exists()
    assert json.loads((out / "manifest.json").read_text())["completed_stages"] == [
        "graph-ablation-0"
    ]


def dataset(path, kind, files=2, rows=10):
    path.mkdir(parents=True, exist_ok=True)
    for i in range(files):
        (path / f"date=2025-01-0{i + 1}").mkdir(exist_ok=True)
        (path / f"date=2025-01-0{i + 1}/part.parquet").touch()
    (path / "_conversion.json").write_text(
        json.dumps({"kind": kind, "rows": rows, "parquet_files": files})
    )
    return path


def check(*args):
    return subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts/check_dataset.py"),
            *args,
        ],
        capture_output=True,
        text=True,
    )


def test_check_dataset_accepts_a_complete_dataset_and_rejects_a_partial_one(tmp_path):
    """The reuse decision. Directory existence is not enough: a half-copied
    dataset has a manifest and too few parquet files, and must be refused."""
    speeds = dataset(tmp_path / "chicago-speeds-history", "speeds", files=3, rows=101_138_608)
    good = check("--speeds", str(speeds))
    assert good.returncode == 0
    assert json.loads(good.stdout)["speeds"]["rows"] == 101_138_608

    (speeds / "date=2025-01-03/part.parquet").unlink()
    partial = check("--speeds", str(speeds))
    assert partial.returncode != 0
    assert "2 parquet files on disk" in partial.stderr

    assert check("--crashes", str(tmp_path / "absent")).returncode != 0
    assert check("--crashes", str(dataset(tmp_path / "wrong", "speeds"))).returncode != 0


def test_check_dataset_records_what_a_run_resolved(tmp_path):
    crashes = dataset(tmp_path / "chicago-crashes", "crashes", rows=207_612)
    record = tmp_path / "work/data-provenance.json"
    assert check("--crashes", str(crashes), "--record", str(record)).returncode == 0
    saved = json.loads(record.read_text())
    assert saved["crashes"]["path"] == str(crashes.resolve())
    assert saved["crashes"]["rows"] == 207_612


def test_prepare_is_safe_to_rerun(tmp_path):
    """`prepare` records no completion marker, so every --resume runs it again.
    The join refuses to replace existing tables, which made the second run fail
    and cancel every stage depending on it."""
    work, data, binary = tmp_path / "work", tmp_path / "data", tmp_path / "bin"
    for path in (work / "graph", data, binary, work / "processed"):
        path.mkdir(parents=True)
    for name in ("chicago_road_network.graphml", "speed_road_network.graphml"):
        (data / name).touch()
    (work / "graph/g_segments.graphml").touch()
    (work / "graph/g_crash_segments.csv").touch()
    # Stands in for the real guard in experiment.py, which refuses a used
    # --processed directory unless --overwrite-processed is passed.
    (binary / "uv").write_text(
        f'#!/bin/bash\ncase "$*" in\n  *experiment.py*)\n'
        f'    if compgen -G "{work}/processed/*.parquet" >/dev/null \\\n'
        f'       && [[ "$*" != *--overwrite-processed* ]]; then exit 1; fi\n'
        f'    mkdir -p "{work}/audit"; echo \'{{}}\' > "{work}/audit/audit.json"\n'
        f'    touch "{work}/processed/speeds.parquet" ;;\nesac\nexit 0\n'
    )
    (binary / "uv").chmod(0o755)

    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}", SLURM_JOB_ID="1")
    command = ["bash", "scripts/hpc/prepare.sh", str(work), str(data)]
    for attempt in ("first run", "resume"):
        done = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
        assert done.returncode == 0, f"{attempt} failed: {done.stderr}"


def test_aggregate_collects_the_scenario_arms_and_scores_the_zero_null(tmp_path):
    """B1-B3, the matching rules and the seed sweep all ran in Run A but were
    never collected, and the zero null is what tells a shrunken estimator from
    a working one."""
    run, work = tmp_path / "run", tmp_path / "work"
    arm = run / "synthetic-scenarios/null-seed42"
    arm.mkdir(parents=True)
    (arm / "summary.json").write_text(json.dumps({"args": {"seed": "42"}}))
    (arm / "effect_recovery.csv").write_text(
        "model,estimator,group,value,n,events,true_mph,estimated_mph,bias_mph,effect_mae\n"
        "gcn,corrected,pooled,all,4,2,-2.0,-1.5,0.5,1.0\n"
        "gcn,corrected,hop,0,2,2,-4.0,-3.0,1.0,1.5\n"
        "gcn,corrected,hop,1,2,2,0.0,-0.4,-0.4,0.4\n"
    )
    # hop 0 carries the effect, hop 1 is B1's true zero.
    (arm / "truth.csv").write_text(
        "event_id,road,hop,minutes,true_effect\n"
        "1,0,0,10,-3.0\n1,1,0,10,-5.0\n1,2,1,10,0.0\n1,3,1,10,0.0\n"
    )
    event = run / "event"
    event.mkdir(parents=True)
    (event / "effect_by_hop.csv").write_text("hop,minutes,effect\n1,30,-0.111\n")
    rule = run / "matching/random-seed42"
    rule.mkdir(parents=True)
    (rule / "effect_by_hop.csv").write_text("hop,minutes,effect\n1,30,-0.2\n")
    seeded = run / "forecast-seeds/gcn-seed43"
    seeded.mkdir(parents=True)
    (seeded / "metrics.json").write_text(
        json.dumps({"splits": {"test": {"mae": 3.1, "rmse": 4.2}, "val": {"mae": 3.0}}})
    )

    out = tmp_path / "paper"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts/aggregate.py"),
            "--run",
            str(run),
            "--work",
            str(work),
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    scenarios = pd.read_csv(out / "scenarios.csv")
    assert set(scenarios.scenario) == {"B1 direct-only"}
    pooled = scenarios.loc[scenarios.group == "pooled"].iloc[0]
    assert pooled.zero_mae == 2.0  # mean |-3, -5, 0, 0|
    rings = scenarios.loc[scenarios.group == "hop"].set_index("value")
    assert rings.loc["0", "zero_mae"] == 4.0
    # The true zero ring: nothing to recover, so any estimate is pure invention
    # and the null is unbeatable.
    assert rings.loc["1", "zero_mae"] == 0.0

    assert set(pd.read_csv(out / "matching.csv").rule) == {
        "matched pre-event, 0 hops (reported)",
        "random-seed42",
    }
    assert pd.read_csv(out / "forecast_seeds.csv").to_dict("records") == [
        {"model": "gcn", "seed": 43, "test_mae": 3.1, "test_rmse": 4.2, "val_mae": 3.0}
    ]


def test_topology_spread_covers_ten_relabellings_at_three_seeds():
    """One relabelling cannot characterise wrong topologies. The arm arithmetic
    has to span both axes without repeating a pair."""
    script = Path(__file__).resolve().parents[1] / "scripts/hpc/experiments.sh"
    body = re.search(r"    topology-spread\)(.*?)\n        ;;", script.read_text(), re.S).group(1)
    arms = {
        (42 + task // 10, 100 + task % 10)
        for task in range(int(re.search(r"task < (\d+)", body).group(1)))
    }
    assert len(arms) == 30
    assert {seed for seed, _ in arms} == {42, 43, 44}
    assert {graph for _, graph in arms} == set(range(100, 110))
    # graph seed 42 is the graph-ablation arm; these must all be new draws.
    assert 42 not in {graph for _, graph in arms}
