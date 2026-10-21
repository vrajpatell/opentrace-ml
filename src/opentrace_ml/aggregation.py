"""Explicit privacy boundary for reviewed, thresholded candidate road geometry."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral

from .map_matching import MapMatchResult


@dataclass(frozen=True, slots=True)
class AggregateRoadCandidate:
    """A quantized aggregate cell; never contains contributor IDs or raw paths."""

    cell_lat: int
    cell_lon: int
    grid_degrees: float
    contributors: int
    minimum_contributors: int
    human_reviewed: bool = False

    def to_geojson(self) -> dict[str, object]:
        if self.human_reviewed is not True:
            raise PermissionError("Explicit human review is required before export")
        if (
            isinstance(self.minimum_contributors, bool)
            or not isinstance(self.minimum_contributors, Integral)
            or self.minimum_contributors < 3
            or isinstance(self.contributors, bool)
            or not isinstance(self.contributors, Integral)
            or self.contributors < self.minimum_contributors
        ):
            raise ValueError("Candidate does not satisfy its contributor threshold")
        _grid(self.grid_degrees)
        for value in (self.cell_lat, self.cell_lon):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError("Cell indices must be integers")
        south = max(-90.0, self.cell_lat * self.grid_degrees)
        west = max(-180.0, self.cell_lon * self.grid_degrees)
        north = min(90.0, (self.cell_lat + 1) * self.grid_degrees)
        east = min(180.0, (self.cell_lon + 1) * self.grid_degrees)
        if not south < north or not west < east:
            raise ValueError("Cell lies outside coordinate bounds")
        return {
            "type": "Feature",
            "properties": {
                "contributing_trips": self.contributors,
                "human_reviewed": True,
                "grid_degrees": self.grid_degrees,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[west, south], [east, south], [east, north], [west, north], [west, south]]
                ],
            },
        }


def _grid(value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or not 0.001 <= value <= 1:
        raise ValueError("grid_degrees must be between 0.001 and 1 degree")


def aggregate_unmatched_traces(
    results: Iterable[MapMatchResult],
    *,
    minimum_contributors: int = 3,
    grid_degrees: float = 0.001,
    consent_granted: bool,
) -> tuple[AggregateRoadCandidate, ...]:
    """Count distinct consenting trip pseudonyms per cell; suppress small groups.

    Repeat uploads/points from the same pseudonym count once. The caller must
    keep trip IDs stable across retries and verify consent for aggregation.
    Grid cells are discovery areas, not inferred road centerlines or anonymity.
    """
    if consent_granted is not True:
        raise PermissionError("Explicit aggregation consent is required")
    if (
        isinstance(minimum_contributors, bool)
        or not isinstance(minimum_contributors, Integral)
        or minimum_contributors < 3
    ):
        raise ValueError("minimum_contributors must be an integer of at least 3")
    _grid(grid_degrees)
    groups: dict[tuple[int, int], set[str]] = defaultdict(set)
    for result in results:
        if not isinstance(result, MapMatchResult) or not re.fullmatch(
            r"trip_[0-9a-f]{64}", result.trip_id
        ):
            raise ValueError("Aggregation requires validated map-match results")
        for observation in result.observations:
            if observation.is_matched:
                continue
            p = observation.source_point
            # Keep exact upper boundaries inside the final finite cell.
            lat = min(math.floor(p.latitude / grid_degrees), math.ceil(90 / grid_degrees) - 1)
            lon = min(math.floor(p.longitude / grid_degrees), math.ceil(180 / grid_degrees) - 1)
            groups[(lat, lon)].add(result.trip_id)
    return tuple(
        AggregateRoadCandidate(lat, lon, grid_degrees, len(ids), minimum_contributors)
        for (lat, lon), ids in sorted(groups.items())
        if len(ids) >= minimum_contributors
    )
