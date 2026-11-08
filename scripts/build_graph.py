"""Build the project road graph, creating any missing inputs first.

This script never talks to the Chicago portal itself. `roads.ensure_data` makes
the missing inputs, in dependency order, skipping whatever already exists:
scripts/fetch_data.sh for downloads (speeds, crashes, segment JSON),
`data.csv_to_parquet` for a CSV already on disk, then `roads.download_street_graph`
and `roads.prepare_roads`. The road graph follows, then the segment graph,
centrality, crash matching and map (`segments.analyze`).

fetch_data.sh uses the public Chicago Data Portal endpoints without a token and
optionally reads CHICAGO_DATA_APP_TOKEN from data/.env. It always writes to
<project>/data.
"""

import argparse
from pathlib import Path

from traffic_gml.graph import DATA, OUTPUT, build_graph
from traffic_gml.plot import plot_graph
from traffic_gml.roads import check_windows, ensure_data
from traffic_gml.segments import analyze


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
    parser.add_argument(
        "--plot", action="store_true", help="also save the congestion PNG beside the graph"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()

    crash_window, speed_window = tuple(args.crash_window), tuple(args.speed_window)
    ensure_data(
        args.data_dir,
        refresh=args.refresh_data,
        prepare=args.prepare_roads,
        network=not args.no_fetch,
        bash=args.bash,
        app_token=args.app_token,
    )
    check_windows(crash_window, speed_window, args.data_dir)
    graph = build_graph(
        crash_window,
        speed_window,
        args.only_80_percent,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    print(f"NetworkX graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")
    print(graph.graph["output_path"])
    if args.plot:
        path = Path(graph.graph["output_path"]).with_suffix(".png")
        plot_graph(graph, path)
        print(path)
    if not args.no_segment_analysis:
        analyze(
            graph,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            k=args.betweenness_k,
            max_crash_points=args.max_crash_points,
        )


if __name__ == "__main__":
    main()
