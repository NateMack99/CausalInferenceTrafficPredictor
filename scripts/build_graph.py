"""Build a road graph for independent crash and speed time windows."""

import argparse
from pathlib import Path

from traffic_gml.graph import OUTPUT, build_graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crash-window", nargs=2, required=True, metavar=("START", "END"))
    parser.add_argument("--speed-window", nargs=2, required=True, metavar=("START", "END"))
    parser.add_argument("--only-80-percent", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    graph = build_graph(
        tuple(args.crash_window),
        tuple(args.speed_window),
        args.only_80_percent,
        output_dir=args.output_dir,
    )
    print(graph.graph["output_path"])


if __name__ == "__main__":
    main()
