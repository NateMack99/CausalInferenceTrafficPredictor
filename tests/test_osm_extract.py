"""Offline extract recovery preserves road selection and directed topology."""

import json

import osmnx as ox
import pytest
from shapely.geometry import box, mapping

from traffic_gml.osm_extract import _drivable, _drive_filters, graph_from_extract


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
    boundary = tmp_path / "boundary.json"
    boundary.write_text(json.dumps(mapping(box(-87.71, 41.79, -87.69, 41.81))))
    source = tmp_path / "source.osm"
    maxlat = "41.79" if fault == "bounds" else "42.0"
    third = "" if fault == "missing_node" else '<node id="3" lat="41.802" lon="-87.698"/>'
    source.write_text(
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
