"""Fetch a specified study window with monthly requests and count validation."""

import argparse
import json
import os
from pathlib import Path

from traffic_gml.data import portal_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--window", nargs=2, required=True, metavar=("START", "END"))
    parser.add_argument(
        "--kinds", nargs="+", choices=["speeds", "crashes"], default=["speeds", "crashes"]
    )
    args = parser.parse_args()
    for kind in args.kinds:
        manifest = args.data_dir / "parquet" / f"chicago-{kind}" / "_conversion.json"
        if manifest.exists():
            raise SystemExit(
                f"{manifest} exists; use a fresh data directory or --kinds for missing data"
            )
        result = portal_download(
            args.data_dir, kind, tuple(args.window), os.environ.get("CHICAGO_DATA_APP_TOKEN")
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
