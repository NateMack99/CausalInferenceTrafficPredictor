"""Rebuild the local speed-road reference from downloaded data."""

import argparse
from pathlib import Path

from traffic_gml.roads import DATA, prepare_roads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()
    prepare_roads(args.data_dir)


if __name__ == "__main__":
    main()
