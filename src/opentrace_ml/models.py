"""Shared data contracts used by vision, forecasting, and map modules."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Pixel-space bounding box in ``xmin, ymin, xmax, ymax`` order."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) for value in (self.xmin, self.ymin, self.xmax, self.ymax)
        ):
            raise ValueError("Bounding box coordinates must be finite")
        if self.xmax < self.xmin or self.ymax < self.ymin:
            raise ValueError("Bounding box maximums must be greater than or equal to minimums")


@dataclass(frozen=True, slots=True)
class Detection:
    """One computer-vision detection tied to a video-relative timestamp."""

    label: str
    confidence: float
    bbox: BoundingBox
    timestamp_seconds: float
    frame_id: str | None = None

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("Detection label cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Detection confidence must be between 0 and 1")
        if self.timestamp_seconds < 0:
            raise ValueError("Detection timestamp cannot be negative")


@dataclass(frozen=True, slots=True)
class GeoPoint:
    """A GPS point whose timestamp is relative to the trip start."""

    latitude: float
    longitude: float
    timestamp_seconds: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (self.latitude, self.longitude, self.timestamp_seconds)
        ):
            raise ValueError("GPS coordinates and timestamp must be finite")
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError("Latitude must be between -90 and 90")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError("Longitude must be between -180 and 180")
        if self.timestamp_seconds < 0:
            raise ValueError("GPS timestamp cannot be negative")


@dataclass(frozen=True, slots=True)
class GeoDetection:
    """A vision detection interpolated onto a trip trace."""

    detection: Detection
    latitude: float
    longitude: float
