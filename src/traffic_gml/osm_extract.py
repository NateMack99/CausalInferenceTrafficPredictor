"""Build the OSMnx driving network from a local OSM XML extract."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx
import osmnx as ox
from shapely.geometry import box, shape


def _elements(path: Path) -> Iterator[ET.Element]:
    """Stream complete top-level elements without retaining the whole extract."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        context = ET.iterparse(stream, events=("start", "end"))
        _, root = next(context)
        for event, element in context:
            if event == "end" and element.tag in {"node", "way", "relation", "bounds"}:
                yield element
                root.clear()


def _drive_filters() -> list[tuple[str, re.Pattern[str] | None]]:
    # Read the installed, locked OSMnx filter instead of maintaining another list.
    query = ox._overpass._get_network_filter("drive")
    parts = re.findall(r'\["([^"\]]+)"(?:!~"([^"\]]+)")?\]', query)
    reconstructed = "".join(f'["{k}"!~"{v}"]' if v else f'["{k}"]' for k, v in parts)
    if reconstructed != query:
        raise ValueError(f"Unsupported OSMnx drive filter: {query}")
    return [(key, re.compile(value) if value else None) for key, value in parts]


def _drivable(tags: dict[str, str], filters: list[tuple[str, re.Pattern[str] | None]]) -> bool:
    return all(
        key in tags if pattern is None else pattern.search(tags.get(key, "")) is None
        for key, pattern in filters
    )


def graph_from_extract(
    source: str | Path,
    boundary: str | Path,
    destination: str | Path,
    *,
    source_url: str,
    buffer_m: float = 500,
) -> nx.MultiDiGraph:
    """Preserve drive filtering, direction and OSMnx simplification.

    ``boundary`` is the saved GeoJSON geometry for Chicago. Require XML bounds
    covering the entire buffered boundary; incomplete ways fail before publication.
    """
    source, boundary, destination = Path(source), Path(boundary), Path(destination)
    if buffer_m <= 0:
        raise ValueError("buffer_m must be positive")
    polygon = shape(json.loads(boundary.read_text()))
    projected, crs = ox.projection.project_geometry(polygon)
    buffered, _ = ox.projection.project_geometry(
        projected.buffer(buffer_m), crs=crs, to_latlong=True
    )
    filters = _drive_filters()
    ways, needed = [], set()
    covered = False
    for element in _elements(source):
        if element.tag == "bounds":
            extent = box(
                *(float(element.attrib[k]) for k in ("minlon", "minlat", "maxlon", "maxlat"))
            )
            covered = extent.covers(buffered)
            if not covered:
                raise ValueError(
                    f"OSM extract bounds do not cover the boundary plus {buffer_m:g} m"
                )
        elif element.tag == "way":
            tags = {tag.attrib["k"]: tag.attrib["v"] for tag in element.findall("tag")}
            if _drivable(tags, filters):
                ways.append(ET.tostring(element))
                needed.update(node.attrib["ref"] for node in element.findall("nd"))
    if not covered:
        raise ValueError("OSM extract has no bounds covering the requested buffered boundary")
    if not ways:
        raise ValueError("No driving ways in OSM extract")
    print(f"Selected {len(ways):,} driving ways and {len(needed):,} referenced nodes", flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        filtered = Path(temporary) / "drive.osm"
        with filtered.open("wb") as stream:
            stream.write(b'<osm version="0.6">\n')
            for element in _elements(source):
                if element.tag == "node" and element.attrib["id"] in needed:
                    stream.write(ET.tostring(element))
                    needed.remove(element.attrib["id"])
            if needed:
                raise ValueError(f"OSM extract is missing {len(needed)} referenced road nodes")
            for way in ways:
                stream.write(way)
            stream.write(b"</osm>")
        graph = ox.graph_from_xml(filtered, simplify=False, retain_all=True)
        graph = ox.truncate.truncate_graph_polygon(graph, buffered)
        graph = ox.truncate.largest_component(graph, strongly=False)
        graph = ox.simplification.simplify_graph(graph)
        result = ox.truncate.truncate_graph_polygon(graph, polygon)
        result = ox.truncate.largest_component(result, strongly=False)
        nx.set_node_attributes(
            result, ox.stats.count_streets_per_node(graph, nodes=result.nodes), "street_count"
        )
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        result.graph.update(
            source_url=source_url,
            source_sha256=digest,
            boundary_sha256=hashlib.sha256(boundary.read_bytes()).hexdigest(),
            network_type="drive",
            buffer_m=buffer_m,
            drive_filter=ox._overpass._get_network_filter("drive"),
            downloaded_utc=datetime.now(UTC).isoformat(),
        )
        staged = Path(temporary) / "network.graphml"
        ox.save_graphml(result, filepath=staged)
        staged.replace(destination)
    print(f"Saved {len(result):,} nodes and {result.number_of_edges():,} driving edges", flush=True)
    return result
