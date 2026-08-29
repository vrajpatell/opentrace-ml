"""Geospatial helpers for joining model output to routes."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .models import Detection, GeoDetection, GeoPoint


def interpolate_position(points: Sequence[GeoPoint], timestamp_seconds: float) -> GeoPoint:
    """Linearly interpolate a position at ``timestamp_seconds``.

    Values before or after the trace are clamped to its first or last point. This
    explicit behavior makes delayed video frames deterministic.
    """

    if not points:
        raise ValueError("At least one GPS point is required")
    if timestamp_seconds < 0:
        raise ValueError("Timestamp cannot be negative")

    ordered = list(points)
    if any(
        current.timestamp_seconds > following.timestamp_seconds
        for current, following in zip(ordered, ordered[1:])
    ):
        raise ValueError("GPS points must be ordered by timestamp")

    if timestamp_seconds <= ordered[0].timestamp_seconds:
        point = ordered[0]
        return GeoPoint(point.latitude, point.longitude, timestamp_seconds)
    if timestamp_seconds >= ordered[-1].timestamp_seconds:
        point = ordered[-1]
        return GeoPoint(point.latitude, point.longitude, timestamp_seconds)

    for left, right in zip(ordered, ordered[1:]):
        if left.timestamp_seconds <= timestamp_seconds <= right.timestamp_seconds:
            duration = right.timestamp_seconds - left.timestamp_seconds
            if duration == 0:
                return GeoPoint(right.latitude, right.longitude, timestamp_seconds)
            ratio = (timestamp_seconds - left.timestamp_seconds) / duration
            return GeoPoint(
                latitude=left.latitude + ratio * (right.latitude - left.latitude),
                longitude=left.longitude + ratio * (right.longitude - left.longitude),
                timestamp_seconds=timestamp_seconds,
            )

    raise RuntimeError("Unable to interpolate an ordered GPS trace")


def geolocate_detections(
    detections: Sequence[Detection], points: Sequence[GeoPoint]
) -> list[GeoDetection]:
    """Attach each timestamped detection to its interpolated GPS position."""

    located: list[GeoDetection] = []
    for detection in detections:
        position = interpolate_position(points, detection.timestamp_seconds)
        located.append(GeoDetection(detection, position.latitude, position.longitude))
    return located


def haversine_distance_m(left: GeoPoint, right: GeoPoint) -> float:
    """Return great-circle distance between two GPS points in metres."""

    radius_m = 6_371_008.8
    lat1, lat2 = math.radians(left.latitude), math.radians(right.latitude)
    dlat = lat2 - lat1
    dlon = math.radians(right.longitude - left.longitude)
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * radius_m * math.asin(math.sqrt(value))


def route_distance_m(points: Sequence[GeoPoint]) -> float:
    """Return total great-circle distance for an ordered route."""

    return sum(haversine_distance_m(left, right) for left, right in zip(points, points[1:]))


def route_to_geojson(points: Sequence[GeoPoint], **properties: object) -> dict[str, object]:
    """Convert a route to a GeoJSON Feature with correct lon/lat coordinate order."""

    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {
            "type": "LineString",
            "coordinates": [[point.longitude, point.latitude] for point in points],
        },
    }


def detections_to_geojson(detections: Sequence[GeoDetection]) -> dict[str, object]:
    """Convert geolocated detections to a GeoJSON FeatureCollection."""

    features: list[dict[str, object]] = []
    for item in detections:
        detection = item.detection
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "label": detection.label,
                    "confidence": detection.confidence,
                    "timestamp_seconds": detection.timestamp_seconds,
                    "frame_id": detection.frame_id,
                    "bbox": [
                        detection.bbox.xmin,
                        detection.bbox.ymin,
                        detection.bbox.xmax,
                        detection.bbox.ymax,
                    ],
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [item.longitude, item.latitude],
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}

