"""Load timestamped GPS traces from GPX files."""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from xml.etree import ElementTree

from .models import GeoPoint


def _parse_timestamp(value: str) -> datetime:
    """Parse an ISO-8601 timestamp and normalize aware values to UTC."""

    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def load_gpx_points(path: str | Path) -> list[GeoPoint]:
    """Load GPX track points in document order with trip-relative timestamps.

    Timezone-aware timestamps are normalized to UTC. Timezone-naive timestamps
    are interpreted consistently with one another. Track order is preserved and
    must also be chronological.
    """

    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError as error:
        raise ValueError("GPX file is not valid XML") from error

    points: list[tuple[float, float, datetime]] = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "trkpt":
            continue
        latitude = element.get("lat")
        longitude = element.get("lon")
        timestamp = next(
            (child.text for child in element if child.tag.rsplit("}", 1)[-1] == "time"),
            None,
        )
        if latitude is None or longitude is None:
            raise ValueError("GPX track point is missing latitude or longitude")
        if not timestamp:
            raise ValueError("GPX track point is missing a timestamp")
        try:
            points.append((float(latitude), float(longitude), _parse_timestamp(timestamp)))
        except (TypeError, ValueError) as error:
            raise ValueError("GPX track point has an invalid timestamp or coordinate") from error

    if not points:
        raise ValueError("GPX file contains no track points")
    if any(current[2] > following[2] for current, following in pairwise(points)):
        raise ValueError("GPX track points must be ordered by timestamp")

    start = points[0][2]
    return [GeoPoint(lat, lon, (timestamp - start).total_seconds()) for lat, lon, timestamp in points]
