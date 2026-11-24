"""Download the Chicago driving network as GraphML, or rebuild it from a local extract."""

import argparse
from pathlib import Path

import osmnx as ox

from traffic_gml.osm_extract import graph_from_extract
from traffic_gml.roads import DATA, download_street_graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--overpass-url", help="Overpass mirror to use instead of the default")
    offline = parser.add_argument_group("offline rebuild when Overpass is unavailable")
    offline.add_argument("--osm-xml", type=Path, help="local .osm or .osm.gz extract")
    offline.add_argument("--boundary", type=Path, help="GeoJSON boundary geometry")
    offline.add_argument("--source-url", help="where the extract came from; stored in the graph")
    offline.add_argument("--buffer-m", type=float, default=500)
    args = parser.parse_args()
    if args.osm_xml:
        if not args.boundary or not args.source_url:
            parser.error("--osm-xml also needs --boundary and --source-url")
        graph_from_extract(
            args.osm_xml,
            args.boundary,
            args.data_dir / "chicago_road_network.graphml",
            source_url=args.source_url,
            buffer_m=args.buffer_m,
        )
        return
    if args.overpass_url:
        ox.settings.overpass_url = args.overpass_url
    download_street_graph(args.data_dir)


if __name__ == "__main__":
    main()
