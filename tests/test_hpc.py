"""Offline run preparation: audits must finish without fitting a model."""

import importlib.util
import json
import os
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
    log = tmp_path / "submissions.json"
    name = f"pytest-{tmp_path.name}"
    command = ["bash", "scripts/hpc/submit.sh", "--name", name]
    env = dict(
        os.environ,
        PATH=f"{binary}:{os.environ['PATH']}",
        SUBMISSION_LOG=str(log),
        MIN_FREE_GB="0",
        GPU_GRES="gpu:test:1",
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
        assert stages[:4] == ["verify", "prepare", "audit", "main"]
        assert {"treatment", "graph-ablation", "aggregate", "finish"} <= set(stages)

        for args, stage in zip(calls, stages, strict=True):
            if stage == "verify":
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
