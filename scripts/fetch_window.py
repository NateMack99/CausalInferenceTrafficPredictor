"""Fetch a study window into one data directory, reusing the project fetchers.

The shell fetchers own the month-by-month requests, resume behaviour and count
checks; this only points them at ``--data-dir`` and converts the speed CSV.

    uv run python scripts/fetch_window.py --data-dir work/data \
        --window 2024-07-01 2026-04-01 --kinds speeds
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from traffic_gml.data import csv_to_parquet

SCRIPTS = Path(__file__).resolve().parent
# Directory names follow the CSV stems that scripts/convert_to_parquet.py uses.
DATASETS = {"speeds": "chicago-speeds-history", "crashes": "chicago-crashes"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--window", nargs=2, required=True, metavar=("START", "END"))
    parser.add_argument(
        "--kinds", nargs="+", choices=["speeds", "crashes"], default=["speeds", "crashes"]
    )
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    start, end = args.window
    data_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "DATA_DIR": str(data_dir), "PYTHON_PATH": sys.executable}
    for kind in args.kinds:
        manifest = data_dir / "parquet" / DATASETS[kind] / "_conversion.json"
        if manifest.exists():
            raise SystemExit(
                f"{manifest} exists; use a fresh data directory or --kinds for missing data"
            )
        if kind == "speeds":
            subprocess.run(
                ["bash", str(SCRIPTS / "fetch_speeds_history.sh"), start, end],
                env=env,
                check=True,
            )
            csv = data_dir / f"{DATASETS[kind]}.csv"
            print(json.dumps(csv_to_parquet(csv, kind="speeds", overwrite=True), indent=2))
        else:
            subprocess.run(
                ["bash", str(SCRIPTS / "fetch_data.sh"), "--crashes-only"],
                env={**env, "CRASH_START": start, "CRASH_END": end},
                check=True,
            )
        if not manifest.exists():
            raise SystemExit(f"{kind} fetch finished without writing {manifest}")


if __name__ == "__main__":
    main()
