"""OpenTrace ML public API."""

from .aggregation import AggregateRoadCandidate, aggregate_unmatched_traces
from .baselines import SeasonalNaiveForecaster
from .detectors import RTMDetDetector
from .ensemble import HorizonEnsembleForecaster, ModelSelectionPath, optimal_model_path
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
from .neural import NeuralTrafficForecaster
from .portable_ensemble import PortableEnsembleModel
from .portable_forecasting import PortableTrafficModel
from .portable_neural import PortableNeuralModel
from .protocols import CallableDetector, Detector, RawDetection, TrafficForecaster
from .routing import RouteSignals, score_route
from .trace import (
    PreparedTrace,
    PreparedTraceSegments,
    TraceCleaningConfig,
    TraceCleaningReport,
    TraceSegmentationReport,
    prepare_trace,
    pseudonymize_trip_id,
)

__all__ = [
    "AggregateRoadCandidate",
    "BoundingBox",
    "CallableDetector",
    "CallableMapMatcher",
    "Detection",
    "DetectionMetrics",
    "Detector",
    "GeoDetection",
    "GeoPoint",
    "HorizonEnsembleForecaster",
    "MapMatchObservation",
    "MapMatchResult",
    "MapMatchSegment",
    "MapMatcher",
    "ModelSelectionPath",
    "NeuralTrafficForecaster",
    "OnlineTrafficForecaster",
    "PerClassDetectionMetrics",
    "PortableEnsembleModel",
    "PortableNeuralModel",
    "PortableTrafficModel",
    "PreparedTrace",
    "PreparedTraceSegments",
    "RTMDetDetector",
    "RawDetection",
    "RawMapMatch",
    "RegressionMetrics",
    "RouteSignals",
    "SeasonalNaiveForecaster",
    "TraceCleaningConfig",
    "TraceCleaningReport",
    "TraceSegmentationReport",
    "TrafficForecaster",
    "aggregate_unmatched_traces",
    "bounding_box_iou",
    "detection_metrics",
    "detections_to_geojson",
    "geolocate_detections",
    "load_gpx_points",
    "optimal_model_path",
    "per_class_detection_metrics",
    "prepare_trace",
    "pseudonymize_trip_id",
    "regression_metrics",
    "rolling_backtest",
    "route_to_geojson",
    "score_route",
]

__version__ = "0.2.0a1"
