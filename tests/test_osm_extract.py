"""Offline extract recovery preserves road selection and directed topology."""

import json
import subprocess
import sys
from pathlib import Path

import osmnx as ox
import pytest
from shapely.geometry import box, mapping

from traffic_gml.osm_extract import _drivable, _drive_filters, graph_from_extract

ROOT = Path(__file__).resolve().parents[1]


def _source(path, *, maxlat="42.0", third='<node id="3" lat="41.802" lon="-87.698"/>'):
    path.write_text(
        '<osm version="0.6">'
        f'<bounds minlat="41.5" minlon="-88" maxlat="{maxlat}" maxlon="-87"/>'
        '<node id="1" lat="41.800" lon="-87.700"/>'
        '<node id="2" lat="41.801" lon="-87.699"/>'
        + third
        + '<node id="4" lat="41.803" lon="-87.697"/>'
        '<way id="10"><nd ref="1"/><nd ref="2"/><nd ref="3"/>'
        '<tag k="highway" v="residential"/><tag k="oneway" v="yes"/></way>'
        '<way id="20"><nd ref="3"/><nd ref="4"/>'
        '<tag k="highway" v="footway"/></way></osm>'
    )
    return path


def _boundary(path):
    path.write_text(json.dumps(mapping(box(-87.71, 41.79, -87.69, 41.81))))
    return path


@pytest.mark.parametrize(
    ("extra", "allowed"),
    [
        ({}, True),
        ({"highway": "footway"}, False),
        ({"highway": "service"}, False),
        ({"access": "private"}, False),
        ({"motor_vehicle": "no"}, False),
        ({"service": "parking_aisle"}, False),
        ({"area": "yes"}, False),
        ({"oneway": "yes"}, True),
    ],
)
def test_drive_filter(extra, allowed):
    assert _drivable({"highway": "residential", **extra}, _drive_filters()) == allowed


@pytest.mark.parametrize("buffer_m", [100, 500])
@pytest.mark.parametrize("fault", [None, "bounds", "missing_node"])
def test_extract_graph_direction_filter_and_publication(tmp_path, fault, buffer_m):
    boundary = _boundary(tmp_path / "boundary.json")
    source = _source(
        tmp_path / "source.osm",
        maxlat="41.79" if fault == "bounds" else "42.0",
        third="" if fault == "missing_node" else '<node id="3" lat="41.802" lon="-87.698"/>',
    )
    destination = tmp_path / "network.graphml"
    if fault:
        with pytest.raises(ValueError, match="bounds" if fault == "bounds" else "missing"):
            graph_from_extract(
                source,
                boundary,
                destination,
                source_url="https://example.test/osm",
                buffer_m=buffer_m,
            )
        assert not destination.exists()
        return
    graph = graph_from_extract(
        source, boundary, destination, source_url="https://example.test/osm", buffer_m=buffer_m
    )
    assert set(graph) == {1, 3}  # shape node simplified, footway excluded
    assert graph.has_edge(1, 3) and not graph.has_edge(3, 1)
    assert graph.edges[1, 3, 0]["length"] > 200
    restored = ox.load_graphml(destination)
    assert restored.graph["source_sha256"] == graph.graph["source_sha256"]
    assert restored.graph["network_type"] == "drive"
    assert float(restored.graph["buffer_m"]) == buffer_m


def test_cli_rebuilds_the_street_graph_offline(tmp_path):
    """scripts/hpc/extract.sh passes these flags; a live query must never be needed."""
    data = tmp_path / "data"
    data.mkdir()
    command = [
        sys.executable,
        str(ROOT / "scripts/download_street_graph.py"),
        "--data-dir",
        str(data),
        "--osm-xml",
        str(_source(tmp_path / "source.osm")),
        "--boundary",
        str(_boundary(tmp_path / "boundary.json")),
        "--source-url",
        "https://example.test/Chicago.osm.gz",
        "--buffer-m",
        "100",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    graph = ox.load_graphml(data / "chicago_road_network.graphml")
    assert graph.graph["source_url"] == "https://example.test/Chicago.osm.gz"
    assert set(graph) == {1, 3}
    incomplete = subprocess.run(command[:6], capture_output=True, text=True)
    assert incomplete.returncode != 0 and "--boundary" in incomplete.stderr
