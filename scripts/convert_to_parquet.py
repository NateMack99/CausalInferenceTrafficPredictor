"""Convert an existing Chicago CSV into date-partitioned Parquet."""

import argparse
import json
from pathlib import Path

from traffic_gml.data import csv_to_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--kind", required=True, choices=["speeds", "crashes"])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = csv_to_parquet(
        args.csv_path, args.output_dir, kind=args.kind, overwrite=args.overwrite
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
