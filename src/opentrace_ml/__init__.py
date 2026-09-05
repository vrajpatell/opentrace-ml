"""OpenTrace ML public API."""

from .evaluation import (
    DetectionMetrics,
    PerClassDetectionMetrics,
    RegressionMetrics,
    bounding_box_iou,
    detection_metrics,
    per_class_detection_metrics,
    regression_metrics,
    rolling_backtest,
)
from .forecasting import OnlineTrafficForecaster
from .geo import detections_to_geojson, geolocate_detections, route_to_geojson
from .gpx import load_gpx_points
from .map_matching import (
    CallableMapMatcher,
    MapMatcher,
    MapMatchObservation,
    MapMatchResult,
    MapMatchSegment,
    RawMapMatch,
)
from .models import BoundingBox, Detection, GeoDetection, GeoPoint
from .protocols import CallableDetector, Detector, RawDetection
from .routing import RouteSignals, score_route
from .trace import (
    PreparedTrace,
    TraceCleaningConfig,
    TraceCleaningReport,
    prepare_trace,
    pseudonymize_trip_id,
)

__all__ = [
    "BoundingBox",
    "CallableDetector",
    "CallableMapMatcher",
    "Detection",
    "DetectionMetrics",
    "Detector",
    "GeoDetection",
    "GeoPoint",
    "MapMatchObservation",
    "MapMatchResult",
    "MapMatchSegment",
    "MapMatcher",
    "OnlineTrafficForecaster",
    "PerClassDetectionMetrics",
    "PreparedTrace",
    "RawDetection",
    "RawMapMatch",
    "RegressionMetrics",
    "RouteSignals",
    "TraceCleaningConfig",
    "TraceCleaningReport",
    "bounding_box_iou",
    "detection_metrics",
    "detections_to_geojson",
    "geolocate_detections",
    "load_gpx_points",
    "per_class_detection_metrics",
    "prepare_trace",
    "pseudonymize_trip_id",
    "regression_metrics",
    "rolling_backtest",
    "route_to_geojson",
    "score_route",
]

__version__ = "0.1.0"
