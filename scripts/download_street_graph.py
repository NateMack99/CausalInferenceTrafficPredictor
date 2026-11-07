"""Download the Chicago driving network as GraphML."""

import argparse
from pathlib import Path

from traffic_gml.roads import DATA, download_street_graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()
    download_street_graph(args.data_dir)


if __name__ == "__main__":
    main()
