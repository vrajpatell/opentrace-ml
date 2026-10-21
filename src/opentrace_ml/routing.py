"""Transparent route-intelligence scoring built from ML and map signals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real


@dataclass(frozen=True, slots=True)
class RouteSignals:
    """Normalized signals for one route candidate."""

    route_id: str
    distance_m: float
    shortest_distance_m: float
    potholes: int = 0
    other_hazards: int = 0
    traffic_ratio: float = 0.0
    unmatched_ratio: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.route_id, str) or not self.route_id:
            raise ValueError("Route ID must be a non-empty string")
        if any(
            isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v <= 0
            for v in (self.distance_m, self.shortest_distance_m)
        ):
            raise ValueError("Route distances must be positive and finite")
        if any(
            isinstance(v, bool) or not isinstance(v, Integral) or v < 0
            for v in (self.potholes, self.other_hazards)
        ):
            raise ValueError("Hazard counts must be non-negative integers")
        if (
            isinstance(self.traffic_ratio, bool)
            or not isinstance(self.traffic_ratio, Real)
            or not 0 <= self.traffic_ratio <= 1
        ):
            raise ValueError("traffic_ratio must be between 0 and 1")
        if (
            isinstance(self.unmatched_ratio, bool)
            or not isinstance(self.unmatched_ratio, Real)
            or not 0 <= self.unmatched_ratio <= 1
        ):
            raise ValueError("unmatched_ratio must be between 0 and 1")


def score_route(signals: RouteSignals) -> dict[str, object]:
    """Return a 0–100 reliability score with auditable penalty components."""

    detour_ratio = max(0.0, signals.distance_m / signals.shortest_distance_m - 1.0)
    penalties = {
        "damage": min(35.0, signals.potholes * 7.0 + signals.other_hazards * 3.0),
        "traffic": signals.traffic_ratio * 30.0,
        "map_uncertainty": signals.unmatched_ratio * 20.0,
        "detour": min(15.0, detour_ratio * 50.0),
    }
    score = max(0.0, 100.0 - sum(penalties.values()))
    return {
        "route_id": signals.route_id,
        "reliability_score": round(score, 2),
        "penalties": {name: round(value, 2) for name, value in penalties.items()},
    }
