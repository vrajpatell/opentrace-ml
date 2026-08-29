"""OpenTrace ML public API."""

from .evaluation import (
    DetectionMetrics,
    RegressionMetrics,
    bounding_box_iou,
    detection_metrics,
    regression_metrics,
    rolling_backtest,
)
from .forecasting import OnlineTrafficForecaster
from .geo import detections_to_geojson, geolocate_detections, route_to_geojson
from .gpx import load_gpx_points
from .models import BoundingBox, Detection, GeoDetection, GeoPoint
from .protocols import CallableDetector, Detector, RawDetection
from .routing import RouteSignals, score_route

__all__ = [
    "BoundingBox",
    "CallableDetector",
    "Detection",
    "DetectionMetrics",
    "Detector",
    "GeoDetection",
    "GeoPoint",
    "OnlineTrafficForecaster",
    "RawDetection",
    "RegressionMetrics",
    "RouteSignals",
    "bounding_box_iou",
    "detection_metrics",
    "detections_to_geojson",
    "geolocate_detections",
    "load_gpx_points",
    "regression_metrics",
    "rolling_backtest",
    "route_to_geojson",
    "score_route",
]

__version__ = "0.1.0"
