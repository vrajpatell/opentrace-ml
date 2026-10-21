from pathlib import Path

import pytest


def test_original_osm_fixture_builds_without_network(monkeypatch):
    ox = pytest.importorskip("osmnx")
    import socket

    import requests

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline fixture attempted network access")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    graph = ox.graph_from_xml(
        Path(__file__).parent / "fixtures/original-road.osm", bidirectional=True, simplify=False
    )
    assert set(graph.nodes) == {1, 2, 3}
    assert set(graph.edges()) == {(1, 2), (2, 1), (2, 3), (3, 2)}
    assert all(e["highway"] == "residential" for _, _, e in graph.edges(data=True))
