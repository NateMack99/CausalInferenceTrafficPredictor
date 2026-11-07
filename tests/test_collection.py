"""Offline checks for the relocated collection and graph-preview tools."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import pyarrow.dataset as ds
import pytest
from shapely.geometry import LineString

from traffic_gml.data import csv_to_parquet
from traffic_gml.graph import DATA, build_graph, load_graph
from traffic_gml.plot import plot_graph

ROOT = Path(__file__).resolve().parents[1]


def test_conversion_types_partitions_and_refresh(tmp_path):
    source = tmp_path / "speeds.csv"
    pd.DataFrame(
        {
            "TIME": ["01/01/2025 12:00:00 AM", "2025-01-02T00:00:00.123", "01/01/2025 12:10:00 AM"],
            "SEGMENT_ID": [1, 2, 1],
            "SPEED": [-1, 0, 25],
        }
    ).to_csv(source, index=False)
    result = csv_to_parquet(source, kind="speeds", chunksize=1)
    path = Path(result["output_dir"])
    dataset = ds.dataset(path, format="parquet", partitioning="hive")
    assert dataset.count_rows() == 3
    assert dataset.to_table().column("speed").null_count == 1
    assert dataset.count_rows(filter=ds.field("date") == "2025-01-02") == 1
    source.write_text("TIME,SEGMENT_ID,SPEED\ninvalid,1,20\n")
    with pytest.raises(ValueError, match="timestamps"):
        csv_to_parquet(source, kind="speeds", overwrite=True)
    assert ds.dataset(path, format="parquet", partitioning="hive").count_rows() == 3


def test_graph_windows_and_plot(tmp_path):
    assert DATA == ROOT / "data"
    graph = nx.MultiDiGraph(crs="EPSG:4326", simplified=True)
    coordinates = {
        1: (-87.63, 41.88),
        2: (-87.62, 41.88),
        3: (-87.63, 41.8802),
        4: (-87.62, 41.8802),
        5: (-87.62, 41.89),
    }
    for node, (x, y) in coordinates.items():
        graph.add_node(node, x=x, y=y)
    for u, v in [(1, 2), (2, 5), (3, 4)]:
        graph.add_edge(
            u, v, key=0, length=1000, geometry=LineString([coordinates[u], coordinates[v]])
        )
    ox.save_graphml(graph, filepath=tmp_path / "chicago_road_network.graphml")
    starter = graph.edge_subgraph([(1, 2, 0), (2, 5, 0)]).copy()
    for edge, sid, coverage in [((1, 2, 0), 10, 0.9), ((2, 5, 0), 12, 0.5)]:
        starter.edges[edge].update(segment_id=sid, coverage_fraction=coverage, max_speed_mph=40.0)
    ox.save_graphml(starter, filepath=tmp_path / "speed_road_network.graphml")
    speeds = pd.DataFrame(
        {
            "TIME": ["01/01/2025 08:00:00 AM", "01/01/2025 08:30:00 AM", "01/01/2025 09:00:00 AM"],
            "SEGMENT_ID": [10, 10, 10],
            "SPEED": [0, 20, 40],
        }
    )
    crashes = pd.DataFrame(
        {
            "CRASH_DATE": ["01/01/2025 12:00:00 PM"] * 2 + ["01/01/2025 01:00:00 PM"],
            "LATITUDE": [41.88, 41.8802, 41.88],
            "LONGITUDE": [-87.625] * 3,
        }
    )
    for name, kind, frame in [
        ("chicago-speeds", "speeds", speeds),
        ("chicago-crashes", "crashes", crashes),
    ]:
        source = tmp_path / f"{name}.csv"
        frame.to_csv(source, index=False)
        csv_to_parquet(source, kind=kind)
    windows = (("2025-01-01 12:00", "2025-01-01 13:00"), ("2025-01-01 08:00", "2025-01-01 09:00"))
    result = build_graph(*windows, data_dir=tmp_path, output_dir=tmp_path / "output")
    assert result.edges[1, 2, 0]["speed_mph"] == 10
    assert result.edges[1, 2, 0]["crash_count"] == 1
    assert np.isnan(result.edges[2, 5, 0]["speed_mph"])
    assert json.loads(result.graph["crash_status_counts_json"]) == {
        "matched": 1,
        "outside_subgraph": 1,
    }
    assert load_graph(result.graph["output_path"]).edges[1, 2, 0]["speed_count"] == 2
    figure, _ = plot_graph(result)
    plt.close(figure)
    assert Path(result.graph["output_path"]).with_suffix(".png").exists()
    subset = build_graph(
        *windows, only_80_percent=True, data_dir=tmp_path, output_dir=tmp_path / "output"
    )
    assert set(subset.edges(keys=True)) == {(1, 2, 0)}


@pytest.mark.parametrize("download", [False, True])
@pytest.mark.parametrize("existing_venv", [False, True])
def test_setup_uses_locked_uv_environment_and_saved_token(tmp_path, download, existing_venv):
    scripts, data, binary = tmp_path / "scripts", tmp_path / "data", tmp_path / "bin"
    for path in (scripts, data, binary):
        path.mkdir()
    if existing_venv:
        (tmp_path / ".venv").mkdir()
    shutil.copy(ROOT / "scripts/setup.sh", scripts / "setup.sh")
    (scripts / "fetch_data.sh").write_text('printf "%s" "$PYTHON_PATH" > "$SETUP_FETCH_LOG"\n')
    token = "CHICAGO_DATA_APP_TOKEN=mock-token\n"
    (data / ".env").write_text(token)
    uv = binary / "uv"
    uv.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> "$SETUP_UV_LOG"\n')
    curl = binary / "curl"
    curl.write_text("#!/bin/sh\nprintf 200\n")
    uv.chmod(0o755)
    curl.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{binary}{os.pathsep}{os.environ['PATH']}",
        SETUP_UV_LOG=str(tmp_path / "uv.log"),
        SETUP_FETCH_LOG=str(tmp_path / "fetch.log"),
    )
    result = subprocess.run(
        ["bash", str(scripts / "setup.sh")],
        input="y\n" if download else "n\n",
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "skipping token" not in result.stderr
    expected_creation = [] if existing_venv else ["venv", "--python", "3.12", ".venv"]
    assert (tmp_path / "uv.log").read_text().splitlines() == expected_creation + [
        "sync",
        "--locked",
        "--python",
        "3.12",
    ]
    assert (data / ".env").read_text() == token
    assert (tmp_path / "fetch.log").exists() == download
    if download:
        assert (tmp_path / "fetch.log").read_text() == str(tmp_path / ".venv/bin/python")


@pytest.mark.parametrize("crashes_only", [False, True])
def test_fetch_relocated_scripts(tmp_path, crashes_only):
    scripts, data, binary = tmp_path / "scripts", tmp_path / "data", tmp_path / "bin"
    for path in (scripts, data, binary):
        path.mkdir()
    for name in ("fetch_data.sh", "convert_to_parquet.py"):
        shutil.copy(ROOT / "scripts" / name, scripts / name)
    for name in ("download_street_graph.py", "prepare_roads.py"):
        (scripts / name).write_text(
            'from pathlib import Path\nPath(__file__).with_suffix(".ran").touch()\n'
        )
    (data / ".env").write_text("CHICAGO_DATA_APP_TOKEN=mock-token\n")
    curl = binary / "curl"
    curl.write_text(
        f"#!{sys.executable}\n"
        + """import csv, sys
out = sys.argv[sys.argv.index('--output') + 1]
with open(out, 'w', newline='') as target:
    if out.endswith('.json'):
        target.write('[]')
    else:
        if 'speeds' in out:
            rows = [['TIME', 'SEGMENT_ID', 'SPEED'], ['01/01/2025 12:00:00 AM', '1', '20']]
        else:
            rows = [['CRASH_DATE'], ['01/01/2025 12:00:00 AM']]
        csv.writer(target).writerows(rows)
"""
    )
    curl.chmod(0o755)
    env = dict(
        os.environ, PATH=f"{binary}{os.pathsep}{os.environ['PATH']}", PYTHON_PATH=sys.executable
    )
    command = ["bash", str(scripts / "fetch_data.sh")]
    if crashes_only:
        command.append("--crashes-only")
    subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    assert (data / "parquet/chicago-crashes/_conversion.json").exists()
    assert (scripts / "prepare_roads.ran").exists() == (not crashes_only)
