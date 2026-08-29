"""Load timestamped GPS traces from GPX files."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

from .models import GeoPoint


def load_gpx_points(path: str | Path) -> list[GeoPoint]:
    """Load ordered GPX track points with timestamps relative to the first point."""
    root = ElementTree.parse(path).getroot()
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
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            points.append((float(latitude), float(longitude), parsed))
        except (TypeError, ValueError) as error:
            raise ValueError("GPX track point has an invalid timestamp or coordinate") from error

    if not points:
        raise ValueError("GPX file contains no track points")
    start = points[0][2]
    return [GeoPoint(lat, lon, (timestamp - start).total_seconds()) for lat, lon, timestamp in points]
