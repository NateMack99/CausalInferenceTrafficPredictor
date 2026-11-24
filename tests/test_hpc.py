"""Offline run preparation: audits must finish without fitting a model."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

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


def test_slurm_matrix_has_failure_dependencies_and_bounded_gpu_arrays(tmp_path):
    """Validate submissions offline; never contact the cluster from pytest."""
    binary = tmp_path / "bin"
    binary.mkdir()
    stub = binary / "sbatch"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "path = Path(os.environ['SUBMISSION_LOG'])\n"
        "calls = json.loads(path.read_text()) if path.exists() else []\n"
        "calls.append(sys.argv[1:])\n"
        "path.write_text(json.dumps(calls))\n"
        "print(90000 + len(calls))\n"
    )
    stub.chmod(0o755)
    log = tmp_path / "submissions.json"
    manifest = tmp_path / "jobs.tsv"
    script = Path(__file__).resolve().parents[1] / "scripts/hpc/submit.sh"
    command = [
        "bash",
        str(script),
        str(tmp_path / "work"),
        str(tmp_path / "run"),
        str(manifest),
        str(tmp_path / "logs"),
        "80001",
        "80002",
    ]
    env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}", SUBMISSION_LOG=str(log))
    subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    calls = json.loads(log.read_text())
    assert len(calls) == 13
    for args in calls:
        assert "--kill-on-invalid-dep=yes" in args
        dependency = next(a for a in args if a.startswith("--dependency="))
        if "--job-name=traffic-finish" in args:
            assert dependency.startswith("--dependency=afterany:")
        else:
            assert dependency.startswith("--dependency=afterok:")
        if "--partition=gpu" in args:
            assert "--gres=gpu:a100:1" in args
            for arg in args:
                if arg.startswith("--array="):
                    assert arg.endswith("%2")
    # Duplicate invocation must not quietly queue the entire GPU workload twice.
    again = subprocess.run(command, env=env, capture_output=True, text=True)
    assert again.returncode != 0
    assert len(json.loads(log.read_text())) == 13
