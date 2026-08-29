"""OpenTrace ML public API."""

from .forecasting import OnlineTrafficForecaster
from .geo import detections_to_geojson, geolocate_detections, route_to_geojson
from .models import BoundingBox, Detection, GeoDetection, GeoPoint
from .routing import RouteSignals, score_route

__all__ = [
    "BoundingBox",
    "Detection",
    "GeoDetection",
    "GeoPoint",
    "OnlineTrafficForecaster",
    "RouteSignals",
    "detections_to_geojson",
    "geolocate_detections",
    "route_to_geojson",
    "score_route",
]

__version__ = "0.1.0"

