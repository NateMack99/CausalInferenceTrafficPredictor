"""Plot an existing graph, or build and plot an example window graph."""

import argparse
from pathlib import Path

from traffic_gml.graph import build_graph, load_graph
from traffic_gml.plot import plot_graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--crash-window", nargs=2, default=["2025-01-02 11:30", "2025-01-03 12:00"])
    parser.add_argument("--speed-window", nargs=2, default=["2025-01-02 12:00", "2025-01-03 13:30"])
    parser.add_argument("--only-80-percent", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    graph = (
        load_graph(args.graph)
        if args.graph
        else build_graph(tuple(args.crash_window), tuple(args.speed_window), args.only_80_percent)
    )
    plot_graph(graph, show=args.show)
    print(Path(graph.graph["output_path"]).with_suffix(".png"))


if __name__ == "__main__":
    main()
