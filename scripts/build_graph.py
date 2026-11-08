"""Generate the project road NetworkX graph, creating any missing inputs first.

This script never talks to the Chicago portal itself. Missing inputs are made by
the project's own helpers, in dependency order, skipping whatever already exists:

    scripts/fetch_data.sh             download + convert (speeds, crashes, segment
                                      JSON, street graph, speed-road reference)
    traffic_gml.data.csv_to_parquet   convert a CSV already on disk (what
                                      convert_to_parquet.py wraps)
    traffic_gml.roads.download_street_graph
    traffic_gml.roads.prepare_roads

Then the road graph is built, followed by the segment graph, centrality, crash
matching and map (traffic_gml.segments.analyze).

fetch_data.sh uses the public Chicago Data Portal endpoints without a token and
optionally reads CHICAGO_DATA_APP_TOKEN from data/.env. It always writes to
<project>/data.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from traffic_gml.data import csv_to_parquet
from traffic_gml.graph import DATA, OUTPUT, build_graph
from traffic_gml.roads import download_street_graph, prepare_roads
from traffic_gml.segments import analyze

SCRIPTS = Path(__file__).resolve().parent
FETCH_SCRIPT = SCRIPTS / "fetch_data.sh"


# --------------------------------------------------------------------------
# Where things live, and what counts as present
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


# --------------------------------------------------------------------------
# Running fetch_data.sh
# --------------------------------------------------------------------------
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


def _fetch_command(bash: str, script: Path, args: tuple[str, ...], windows: bool) -> list[str]:
    """`-o igncr` makes Git Bash ignore the CR in CRLF line endings, including in
    sourced files such as data/.env. Other platforms' bash lacks the option."""
    return [bash, *(["-o", "igncr"] if windows else []), script.as_posix(), *args]


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
    project = FETCH_SCRIPT.parent.parent
    if data_dir.resolve() != (project / "data").resolve():
        raise ValueError(
            f"fetch_data.sh always writes to {project / 'data'}; use --data-dir to match it, "
            "or create the inputs yourself"
        )
    windows = os.name == "nt"
    if not windows:
        files = [FETCH_SCRIPT]
        env_file = data_dir / ".env"
        if env_file.is_file():
            files.append(env_file)
        _check_line_endings(files)
    command = _fetch_command(_find_bash(bash), FETCH_SCRIPT, args, windows)
    print("Running: " + " ".join(command[1:]))
    # PYTHON_PATH is how fetch_data.sh is told which interpreter (and venv) to use.
    env = {**os.environ, "PYTHON_PATH": Path(sys.executable).as_posix()}
    if app_token:
        env["CHICAGO_DATA_APP_TOKEN"] = app_token
    result = subprocess.run(command, cwd=project, env=env)
    if result.returncode:
        raise RuntimeError(f"fetch_data.sh exited with code {result.returncode}")


# --------------------------------------------------------------------------
# Make whatever is missing
# --------------------------------------------------------------------------
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


def _check_windows(data_dir: Path, windows: dict[str, tuple[str, tuple[str, str]]]) -> None:
    """Warn when a requested window is not covered by the converted data."""
    for label, (folder, window) in windows.items():
        manifest = json.loads((data_dir / "parquet" / folder / "_conversion.json").read_text())
        first = pd.Timestamp(manifest["first_timestamp"])
        last = pd.Timestamp(manifest["last_timestamp"])
        start, end = map(pd.Timestamp, window)
        if start < first.normalize() or end > last.normalize() + pd.Timedelta(days=1):
            print(
                f"WARNING: {label} window {start} to {end} is outside the downloaded data "
                f"({first} to {last}). To cover it, change the dates in scripts/fetch_data.sh "
                "and rerun with --refresh-data; otherwise the graph will be empty there."
            )


# --------------------------------------------------------------------------
# Graph generation
# --------------------------------------------------------------------------
def generate_road_graph(
    crash_window: tuple[str, str],
    speed_window: tuple[str, str],
    *,
    data_dir: str | Path = DATA,
    output_dir: str | Path = OUTPUT,
    only_80_percent: bool = False,
    prepare: bool = False,
    fetch: bool = True,
    refresh: bool = False,
    bash: str | None = None,
    segment_analysis: bool = True,
    betweenness_k: int | None = None,
    max_crash_points: int = 5_000,
    app_token: str | None = None,
):
    """Build a crash-and-speed-enriched graph, making missing inputs first."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    ensure_data(
        data_dir,
        refresh=refresh,
        prepare=prepare,
        network=fetch,
        bash=bash,
        app_token=app_token,
    )
    _check_windows(
        data_dir,
        {"crash": ("chicago-crashes", crash_window), "speed": ("chicago-speeds", speed_window)},
    )

    graph = build_graph(
        crash_window,
        speed_window,
        only_80_percent,
        data_dir=data_dir,
        output_dir=output_dir,
    )
    print(f"NetworkX graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")
    print(graph.graph["output_path"])
    if segment_analysis:
        analyze(
            graph,
            data_dir=data_dir,
            output_dir=output_dir,
            k=betweenness_k,
            max_crash_points=max_crash_points,
        )
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crash-window", nargs=2, required=True, metavar=("START", "END"))
    parser.add_argument("--speed-window", nargs=2, required=True, metavar=("START", "END"))
    parser.add_argument("--only-80-percent", action="store_true")
    parser.add_argument("--prepare-roads", action="store_true", help="force prepare_roads")
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="never download; convert/prepare local files, or fail with missing inputs",
    )
    parser.add_argument("--refresh-data", action="store_true", help="rerun all of fetch_data.sh")
    parser.add_argument("--bash", help="path to bash (default: Git Bash on Windows)")
    parser.add_argument("--app-token", help="optional Chicago Data Portal app token")
    parser.add_argument(
        "--no-segment-analysis",
        action="store_true",
        help="skip the segment graph, centrality and map",
    )
    parser.add_argument(
        "--betweenness-k",
        type=int,
        default=None,
        help="sample this many sources (default: exact up to 5,000 segments)",
    )
    parser.add_argument(
        "--max-crash-points", type=int, default=5_000, help="crash markers drawn per map layer"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    generate_road_graph(
        tuple(args.crash_window),
        tuple(args.speed_window),
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        only_80_percent=args.only_80_percent,
        prepare=args.prepare_roads,
        fetch=not args.no_fetch,
        refresh=args.refresh_data,
        bash=args.bash,
        segment_analysis=not args.no_segment_analysis,
        betweenness_k=args.betweenness_k,
        max_crash_points=args.max_crash_points,
        app_token=args.app_token,
    )


if __name__ == "__main__":
    main()
