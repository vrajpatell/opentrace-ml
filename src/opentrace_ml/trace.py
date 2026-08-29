"""Privacy-aware preparation of GPS traces for downstream processing."""

from __future__ import annotations

import hashlib
import hmac
import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from .geo import haversine_distance_m
from .models import GeoPoint


@dataclass(frozen=True, slots=True)
class TraceCleaningConfig:
    """Deterministic limits applied at the private trace-ingestion boundary."""

    max_speed_m_s: float = 70.0
    min_points: int = 2

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_speed_m_s) or self.max_speed_m_s <= 0:
            raise ValueError("max_speed_m_s must be a positive finite value")
        if self.min_points < 2:
            raise ValueError("min_points must be at least 2")


@dataclass(frozen=True, slots=True)
class TraceCleaningReport:
    """Aggregate counts safe to log without exposing trace coordinates."""

    input_points: int
    output_points: int
    duplicate_points_removed: int
    speed_outliers_removed: int


@dataclass(frozen=True, slots=True)
class PreparedTrace:
    """A consented, pseudonymized trace ready for private downstream processing."""

    trip_id: str
    points: tuple[GeoPoint, ...]
    report: TraceCleaningReport


def pseudonymize_trip_id(trip_id: str, secret_key: str | bytes) -> str:
    """Return a deterministic HMAC pseudonym without retaining the raw trip ID."""

    normalized_id = trip_id.strip()
    if not normalized_id:
        raise ValueError("trip_id cannot be empty")
    key = secret_key.encode("utf-8") if isinstance(secret_key, str) else secret_key
    if len(key) < 16:
        raise ValueError("secret_key must contain at least 16 bytes")
    digest = hmac.new(
        key,
        b"opentrace-trip-v1\x00" + normalized_id.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"trip_{digest}"


def _clean_trace(
    points: Sequence[GeoPoint], config: TraceCleaningConfig
) -> tuple[tuple[GeoPoint, ...], TraceCleaningReport]:
    if len(points) < config.min_points:
        raise ValueError(f"Trace must contain at least {config.min_points} points")

    ordered = list(points)
    if any(
        current.timestamp_seconds > following.timestamp_seconds
        for current, following in pairwise(ordered)
    ):
        raise ValueError("GPS points must be ordered by timestamp")

    kept = [ordered[0]]
    duplicates_removed = 0
    speed_outliers_removed = 0

    for point in ordered[1:]:
        previous = kept[-1]
        duration = point.timestamp_seconds - previous.timestamp_seconds
        distance_m = haversine_distance_m(previous, point)

        if duration == 0:
            if distance_m == 0:
                duplicates_removed += 1
                continue
            raise ValueError("Different GPS positions cannot share the same timestamp")

        if distance_m / duration > config.max_speed_m_s:
            speed_outliers_removed += 1
            continue
        kept.append(point)

    if len(kept) < config.min_points:
        raise ValueError("Trace contains too few usable points after cleaning")

    start = kept[0].timestamp_seconds
    normalized = tuple(
        GeoPoint(point.latitude, point.longitude, point.timestamp_seconds - start)
        for point in kept
    )
    report = TraceCleaningReport(
        input_points=len(ordered),
        output_points=len(normalized),
        duplicate_points_removed=duplicates_removed,
        speed_outliers_removed=speed_outliers_removed,
    )
    return normalized, report


def prepare_trace(
    points: Sequence[GeoPoint],
    *,
    trip_id: str,
    secret_key: str | bytes,
    consent_granted: bool,
    config: TraceCleaningConfig | None = None,
) -> PreparedTrace:
    """Enforce consent, pseudonymize the trip ID, and clean a private trace.

    Callers must pass a trip-scoped identifier, never a hardware device ID. The
    returned coordinates remain sensitive and must not be logged or published.
    """

    if consent_granted is not True:
        raise PermissionError("Explicit consent is required before preparing a GPS trace")
    cleaned, report = _clean_trace(points, config or TraceCleaningConfig())
    return PreparedTrace(
        trip_id=pseudonymize_trip_id(trip_id, secret_key),
        points=cleaned,
        report=report,
    )
