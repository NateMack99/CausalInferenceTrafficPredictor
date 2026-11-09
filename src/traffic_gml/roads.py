"""Road reference data: the Chicago street network, the speed-observed road
subgraph built from it, and the provisioning of the inputs both need."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import pyarrow.dataset as ds
from shapely.geometry import LineString
from shapely.strtree import STRtree

from .data import csv_to_parquet

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
#: The only route to the portal; it always writes to <project>/data.
FETCH_SCRIPT = ROOT / "scripts" / "fetch_data.sh"


def _street(name: str) -> str:
    words = re.sub(r"[^a-z0-9 ]", " ", name.lower()).split()
    if words and words[0] in {"north", "south", "east", "west", "n", "s", "e", "w"}:
        words = words[1:]
    if words and words[-1] in {
        "avenue",
        "ave",
        "street",
        "st",
        "road",
        "rd",
        "boulevard",
        "blvd",
        "drive",
        "dr",
        "parkway",
        "pkwy",
        "court",
        "ct",
        "place",
        "pl",
    }:
        words = words[:-1]
    return " ".join(words)


def prepare_roads(data_dir: str | Path = DATA) -> nx.MultiDiGraph:
    """Save speed_segments.parquet, speed_roads.parquet, and the starter GraphML."""
    data_dir = Path(data_dir)
    source = data_dir / "parquet/chicago-speeds"
    manifest = json.loads((source / "_conversion.json").read_text())
    days = (
        pd.Timestamp(manifest["last_timestamp"]).normalize()
        - pd.Timestamp(manifest["first_timestamp"]).normalize()
    ).days + 1
    bins, maxima, counts = defaultdict(int), defaultdict(float), defaultdict(int)
    dataset = ds.dataset(source, format="parquet", partitioning="hive")
    for batch in dataset.to_batches(
        columns=["timestamp", "segment_id", "speed"],
        filter=ds.field("speed_available"),
        batch_size=500_000,
    ):
        frame = batch.to_pandas()
        frame["day"] = frame.timestamp.dt.normalize()
        frame["slot"] = frame.timestamp.dt.hour * 6 + frame.timestamp.dt.minute // 10
        for key, slots in frame.groupby(["segment_id", "day"]).slot.unique().items():
            bins[key] |= sum(1 << int(slot) for slot in slots)
        for sid, group in frame.groupby("segment_id"):
            maxima[int(sid)] = max(maxima[int(sid)], float(group.speed.max()))
            counts[int(sid)] += len(group)
    covered = defaultdict(int)
    for (sid, _), mask in bins.items():
        covered[int(sid)] += mask.bit_count()
    reference = pd.DataFrame(
        [
            {
                "segment_id": sid,
                "valid_speed_count": counts[sid],
                "max_speed_mph": maxima[sid],
                "coverage_fraction": covered[sid] / (days * 144),
            }
            for sid in sorted(counts)
        ]
    )
    if reference.empty:
        raise ValueError("No valid speeds in the Parquet dataset")
    metadata = pd.DataFrame(json.loads((data_dir / "chicago-traffic-segments.json").read_text()))
    metadata["segment_id"] = metadata.segmentid.astype(int)
    reference = reference.merge(
        metadata[["segment_id", "street", "_direction", "_fromst", "_tost"]],
        on="segment_id",
        how="left",
        validate="one_to_one",
    ).rename(columns={"_direction": "direction", "_fromst": "from_street", "_tost": "to_street"})
    graph = ox.load_graphml(data_dir / "chicago_road_network.graphml")
    projected = ox.project_graph(graph)
    metadata = metadata.loc[metadata.segment_id.isin(reference.segment_id)].copy()
    metadata["geometry"] = [
        LineString(
            [
                (float(row.start_lon), float(row._lif_lat)),
                (float(row._lit_lon), float(row._lit_lat)),
            ]
        )
        for _, row in metadata.iterrows()
    ]
    segments = gpd.GeoDataFrame(metadata, geometry="geometry", crs="EPSG:4326").to_crs(
        projected.graph["crs"]
    )
    lines = list(segments.geometry)
    tree = STRtree(lines)
    matches = []
    for (u, v, key), edge in ox.graph_to_gdfs(projected, nodes=False).iterrows():
        name = edge.get("name", "")
        names = {
            _street(n) for n in (name if isinstance(name, list) else [name]) if isinstance(n, str)
        } - {""}
        vector = np.array(
            [
                projected.nodes[v]["x"] - projected.nodes[u]["x"],
                projected.nodes[v]["y"] - projected.nodes[u]["y"],
            ]
        )
        points = [edge.geometry.interpolate(f, normalized=True) for f in np.linspace(0.1, 0.9, 5)]
        candidates = []
        for i in tree.query(edge.geometry.buffer(40)):
            line = lines[i]
            direction = np.array(line.coords[-1]) - np.array(line.coords[0])
            scale = np.linalg.norm(vector) * np.linalg.norm(direction)
            if not scale or np.dot(vector, direction) / scale < np.cos(np.pi / 4):
                continue
            if _street(segments.iloc[i].street) not in names:
                continue
            distances = np.array([line.distance(point) for point in points])
            support = float((distances <= 40).mean())
            if support >= 0.6:
                candidates.append(
                    (-support, float(distances.mean()), int(segments.iloc[i].segment_id))
                )
        if candidates:
            _, offset, sid = min(candidates)
            matches.append(
                {
                    "edge_u": u,
                    "edge_v": v,
                    "edge_key": key,
                    "segment_id": sid,
                    "match_distance_m": offset,
                }
            )
    roads = pd.DataFrame(matches).merge(reference, on="segment_id", validate="many_to_one")
    starter = graph.edge_subgraph(
        [(r.edge_u, r.edge_v, r.edge_key) for r in roads.itertuples()]
    ).copy()
    for row in roads.itertuples():
        starter.edges[row.edge_u, row.edge_v, row.edge_key].update(
            segment_id=int(row.segment_id),
            coverage_fraction=float(row.coverage_fraction),
            max_speed_mph=float(row.max_speed_mph),
        )
    starter.graph["speed_reference_period"] = (
        f"{manifest['first_timestamp']}/{manifest['last_timestamp']}"
    )
    reference.to_parquet(data_dir / "speed_segments.parquet", index=False)
    reference.loc[reference.coverage_fraction >= 0.8].to_csv(
        data_dir / "dense_80pct_segments.csv", index=False
    )
    roads.to_parquet(data_dir / "speed_roads.parquet", index=False)
    ox.save_graphml(starter, filepath=data_dir / "speed_road_network.graphml")
    print(
        f"Saved {len(reference)} observed segments and {len(roads)} matched road edges "
        f"in {data_dir}"
    )
    return starter


def download_street_graph(data_dir: str | Path = DATA) -> nx.MultiDiGraph:
    """Download Chicago's directed driving network into the local data folder."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    graph = ox.graph_from_place("Chicago, Illinois, USA", network_type="drive")
    ox.save_graphml(graph, filepath=data_dir / "chicago_road_network.graphml")
    return graph


# --------------------------------------------------------------------------
# Inputs: where they live, what counts as present, and how to make them
# --------------------------------------------------------------------------
def _paths(data_dir: Path) -> dict[str, Path]:
    return {
        "speeds": data_dir / "parquet" / "chicago-speeds",
        "crashes": data_dir / "parquet" / "chicago-crashes",
        "segments_json": data_dir / "chicago-traffic-segments.json",
        "street_graph": data_dir / "chicago_road_network.graphml",
        "speed_segments": data_dir / "speed_segments.parquet",
        "speed_roads": data_dir / "speed_roads.parquet",
        "speed_network": data_dir / "speed_road_network.graphml",
    }


def _converted(path: Path) -> bool:
    return (path / "_conversion.json").is_file()


def _has_coordinates(path: Path) -> bool:
    """Crash Parquet from an older fetch may lack latitude/longitude."""
    if not _converted(path):
        return False
    try:
        names = ds.dataset(path, format="parquet", partitioning="hive").schema.names
    except Exception:  # unreadable or half-written dataset: treat as missing
        return False
    return {"latitude", "longitude"} <= set(names)


def _csv_has_coordinates(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        columns = pd.read_csv(path, nrows=0, dtype=str).columns.str.strip().str.lower()
    except Exception:
        return False
    return {"latitude", "longitude"} <= set(columns)


def _present(key: str, path: Path) -> bool:
    if key == "speeds":
        return _converted(path)
    if key == "crashes":
        return _has_coordinates(path)
    return path.exists()


def _find_bash(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if os.name != "nt":
        found = shutil.which("bash")
        if not found:
            raise RuntimeError("bash was not found on PATH; it is needed to run fetch_data.sh")
        return found
    roots = [os.environ.get(v) for v in ("ProgramFiles", "ProgramFiles(x86)")]
    candidates = [Path(r) / "Git" / "bin" / "bash.exe" for r in roots if r]
    if os.environ.get("LOCALAPPDATA"):
        candidates.append(Path(os.environ["LOCALAPPDATA"]) / "Programs/Git/bin/bash.exe")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("bash")
    if found and "system32" not in found.lower():
        return found
    raise RuntimeError(
        "Git Bash was not found. Install Git for Windows, or pass --bash <path to bash.exe>. "
        "(WSL's bash in System32 cannot use the Windows virtual environment.)"
    )


def _check_line_endings(files: list[Path]) -> None:
    for path in files:
        if b"\r\n" in path.read_bytes():
            raise ValueError(
                f"{path} has Windows (CRLF) line endings, which this bash cannot run. "
                f"Convert it, e.g.: dos2unix '{path}'  (or, in PowerShell: "
                f'(Get-Content \'{path}\' -Raw) -replace "`r`n","`n" | '
                f"Set-Content -NoNewline '{path}')"
            )


def run_fetch_script(
    data_dir: Path,
    *args: str,
    bash: str | None = None,
    app_token: str | None = None,
) -> None:
    """Run scripts/fetch_data.sh (optionally `--crashes-only`) with this interpreter."""
    if not FETCH_SCRIPT.is_file():
        raise FileNotFoundError(f"{FETCH_SCRIPT} not found")
    if data_dir.resolve() != (ROOT / "data").resolve():
        raise ValueError(
            f"fetch_data.sh always writes to {ROOT / 'data'}; use --data-dir to match it, "
            "or create the inputs yourself"
        )
    windows = os.name == "nt"
    if not windows:
        files = [FETCH_SCRIPT]
        env_file = data_dir / ".env"
        if env_file.is_file():
            files.append(env_file)
        _check_line_endings(files)
    # `-o igncr` makes Git Bash ignore the CR in CRLF line endings, including in
    # sourced files such as data/.env. Other platforms' bash lacks the option.
    command = [
        _find_bash(bash),
        *(["-o", "igncr"] if windows else []),
        FETCH_SCRIPT.as_posix(),
        *args,
    ]
    print("Running: " + " ".join(command[1:]))
    # PYTHON_PATH is how fetch_data.sh is told which interpreter (and venv) to use.
    env = {**os.environ, "PYTHON_PATH": Path(sys.executable).as_posix()}
    if app_token:
        env["CHICAGO_DATA_APP_TOKEN"] = app_token
    result = subprocess.run(command, cwd=ROOT, env=env)
    if result.returncode:
        raise RuntimeError(f"fetch_data.sh exited with code {result.returncode}")


def ensure_data(
    data_dir: str | Path = DATA,
    *,
    refresh: bool = False,
    prepare: bool = False,
    network: bool = True,
    bash: str | None = None,
    app_token: str | None = None,
) -> list[str]:
    """Create missing inputs, in order; return what was (re)built.

    Anything that needs a download goes through fetch_data.sh (or
    download_street_graph for the street network). An existing CSV is converted
    in place with csv_to_parquet instead of being downloaded again.
    `refresh` reruns the whole of fetch_data.sh. `prepare` forces prepare_roads.
    `network=False` refuses to download and says what is missing.
    """
    data_dir = Path(data_dir)
    p = _paths(data_dir)
    speeds_csv, crashes_csv = data_dir / "chicago-speeds.csv", data_dir / "chicago-crashes.csv"
    built: list[str] = []

    # fetch_data.sh is the only way to get the speeds and segment JSON, and it
    # always fetches both, so a missing one means running all of it.
    full_fetch = (
        refresh
        or not p["segments_json"].is_file()
        or not (_converted(p["speeds"]) or speeds_csv.is_file())
    )
    crashes_ok = _has_coordinates(p["crashes"])
    local_crash_convert = not crashes_ok and _csv_has_coordinates(crashes_csv)
    crash_fetch = not full_fetch and not crashes_ok and not local_crash_convert
    street_fetch = not full_fetch and not p["street_graph"].is_file()

    if not network:
        wanted = [
            label
            for label, needed in (
                ("speeds / segment metadata (fetch_data.sh)", full_fetch),
                ("crashes with coordinates (fetch_data.sh --crashes-only)", crash_fetch),
                ("street graph (download_street_graph)", street_fetch),
            )
            if needed
        ]
        if wanted:
            raise FileNotFoundError(
                "--no-fetch was given, but these need a download:\n  " + "\n  ".join(wanted)
            )

    inputs_changed = False  # of prepare_roads: speeds, segment JSON, street graph
    if full_fetch:
        run_fetch_script(data_dir, bash=bash, app_token=app_token)
        built.append("everything (fetch_data.sh)")
    else:
        if not _converted(p["speeds"]):
            print(f"Converting {speeds_csv.name} to Parquet ...")
            csv_to_parquet(speeds_csv, p["speeds"], kind="speeds", overwrite=True)
            built.append("speeds Parquet")
            inputs_changed = True
        if local_crash_convert:
            print(f"Converting {crashes_csv.name} to Parquet ...")
            csv_to_parquet(crashes_csv, p["crashes"], kind="crashes", overwrite=True)
            built.append("crashes Parquet")
        elif crash_fetch:
            run_fetch_script(data_dir, "--crashes-only", bash=bash, app_token=app_token)
            built.append("crashes (fetch_data.sh --crashes-only)")
        if street_fetch:
            print("Downloading the Chicago street network (slow) ...")
            download_street_graph(data_dir)
            built.append("street graph")
            inputs_changed = True
        prepared = (p["speed_segments"], p["speed_roads"], p["speed_network"])
        if prepare or inputs_changed or not all(path.exists() for path in prepared):
            print("Preparing speed-road reference ...")
            prepare_roads(data_dir)
            built.append("speed-road reference")

    if prepare and full_fetch:  # fetch_data.sh already ran it; honour an explicit request
        prepare_roads(data_dir)
    missing = [str(path) for key, path in p.items() if not _present(key, path)]
    if missing:
        raise RuntimeError("Inputs are still missing after setup:\n  " + "\n  ".join(missing))
    print("Generated: " + ", ".join(built) if built else "All inputs already present.")
    return built


def check_windows(
    crash_window: tuple[str, str],
    speed_window: tuple[str, str],
    data_dir: str | Path = DATA,
) -> None:
    """Warn when a requested window is not covered by the converted data."""
    p = _paths(Path(data_dir))
    for label, dataset, window in (
        ("crash", p["crashes"], crash_window),
        ("speed", p["speeds"], speed_window),
    ):
        manifest = json.loads((dataset / "_conversion.json").read_text())
        first = pd.Timestamp(manifest["first_timestamp"])
        last = pd.Timestamp(manifest["last_timestamp"])
        start, end = map(pd.Timestamp, window)
        if start < first.normalize() or end > last.normalize() + pd.Timedelta(days=1):
            print(
                f"WARNING: {label} window {start} to {end} is outside the downloaded data "
                f"({first} to {last}). To cover it, change the dates in scripts/fetch_data.sh "
                "and rerun with --refresh-data; otherwise the graph will be empty there."
            )
