"""Fetch a public OpenStreetMap driving graph for a small area."""

from opentrace_ml.datasets import fetch_osm_drive_graph

graph = fetch_osm_drive_graph("Vadodara, Gujarat, India")
print(f"nodes={len(graph.nodes):,} edges={len(graph.edges):,}")
print("Attribution: © OpenStreetMap contributors; data available under ODbL 1.0")
