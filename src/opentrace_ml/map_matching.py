"""Model-independent contracts for matching private traces to a road network."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .models import GeoPoint
from .trace import PreparedTrace

_TRIP_PSEUDONYM = re.compile(r"trip_[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class RawMapMatch:
    """One adapter result before its source point is attached."""

    source_index: int
    matched_point: GeoPoint | None = None
    confidence: float = 0.0
    edge_id: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_index, int)
            or isinstance(self.source_index, bool)
            or self.source_index < 0
        ):
            raise ValueError("source_index must be a non-negative integer")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Map-match confidence must be finite and between 0 and 1")
        if self.matched_point is None:
            if self.confidence != 0.0 or self.edge_id is not None:
                raise ValueError("Unmatched observations cannot have confidence or an edge ID")
            return
        if self.confidence <= 0.0:
            raise ValueError("Matched observations must have positive confidence")
        if self.edge_id is None or not self.edge_id.strip():
            raise ValueError("Matched observations require a non-empty edge ID")


@dataclass(frozen=True, slots=True)
class MapMatchObservation:
    """A source trace point and its optional road-network match."""

    source_index: int
    source_point: GeoPoint
    matched_point: GeoPoint | None
    confidence: float
    edge_id: str | None

    def __post_init__(self) -> None:
        RawMapMatch(
            source_index=self.source_index,
            matched_point=self.matched_point,
            confidence=self.confidence,
            edge_id=self.edge_id,
        )
        if (
            self.matched_point is not None
            and self.matched_point.timestamp_seconds != self.source_point.timestamp_seconds
        ):
            raise ValueError("Matched points must preserve the source timestamp")

    @property
    def is_matched(self) -> bool:
        """Whether the source point was matched to a road-network edge."""

        return self.matched_point is not None


@dataclass(frozen=True, slots=True)
class MapMatchSegment:
    """A contiguous matched edge or unmatched run without raw coordinates."""

    start_source_index: int
    end_source_index: int
    matched: bool
    edge_id: str | None
    mean_confidence: float

    def __post_init__(self) -> None:
        if self.start_source_index < 0 or self.end_source_index < self.start_source_index:
            raise ValueError("Map-match segment index range is invalid")
        if not math.isfinite(self.mean_confidence) or not 0.0 <= self.mean_confidence <= 1.0:
            raise ValueError("Segment confidence must be finite and between 0 and 1")
        if self.matched:
            if self.edge_id is None or not self.edge_id.strip():
                raise ValueError("Matched segments require a non-empty edge ID")
            if self.mean_confidence <= 0.0:
                raise ValueError("Matched segments must have positive confidence")
        elif self.edge_id is not None or self.mean_confidence != 0.0:
            raise ValueError("Unmatched segments cannot have confidence or an edge ID")


@dataclass(frozen=True, slots=True)
class MapMatchResult:
    """Ordered map-match output for one pseudonymized private trace."""

    trip_id: str
    observations: tuple[MapMatchObservation, ...]

    def __post_init__(self) -> None:
        if _TRIP_PSEUDONYM.fullmatch(self.trip_id) is None:
            raise ValueError("Map-match results require an OpenTrace trip pseudonym")
        if not self.observations:
            raise ValueError("Map-match results cannot be empty")
        indices = tuple(observation.source_index for observation in self.observations)
        if indices != tuple(range(len(self.observations))):
            raise ValueError("Map-match observations must cover every source point in order")

    @property
    def matched_count(self) -> int:
        return sum(observation.is_matched for observation in self.observations)

    @property
    def unmatched_count(self) -> int:
        return len(self.observations) - self.matched_count

    @property
    def unmatched_ratio(self) -> float:
        return self.unmatched_count / len(self.observations)

    @property
    def segments(self) -> tuple[MapMatchSegment, ...]:
        """Group consecutive observations by matched edge or unmatched status."""

        segments: list[MapMatchSegment] = []
        start = 0
        while start < len(self.observations):
            first = self.observations[start]
            key = (first.is_matched, first.edge_id)
            end = start + 1
            while end < len(self.observations):
                current = self.observations[end]
                if (current.is_matched, current.edge_id) != key:
                    break
                end += 1
            run = self.observations[start:end]
            segments.append(
                MapMatchSegment(
                    start_source_index=start,
                    end_source_index=end - 1,
                    matched=first.is_matched,
                    edge_id=first.edge_id,
                    mean_confidence=sum(item.confidence for item in run) / len(run),
                )
            )
            start = end
        return tuple(segments)


@runtime_checkable
class MapMatcher(Protocol):
    """Minimal interface implemented by road-network matcher adapters."""

    def match(self, trace: PreparedTrace) -> MapMatchResult:
        """Match a consented, pseudonymized trace without publishing its points."""


@dataclass(slots=True)
class CallableMapMatcher:
    """Wrap a deterministic callable in the OpenTrace map-matcher contract."""

    matcher: Callable[[PreparedTrace], Iterable[RawMapMatch]]

    def match(self, trace: PreparedTrace) -> MapMatchResult:
        if not isinstance(trace, PreparedTrace):
            raise TypeError("Map matchers require a PreparedTrace")

        raw_matches = tuple(self.matcher(trace))
        if len(raw_matches) != len(trace.points):
            raise ValueError("A matcher must return one observation for every source point")

        observations: list[MapMatchObservation] = []
        for expected_index, item in enumerate(raw_matches):
            if not isinstance(item, RawMapMatch):
                raise TypeError("Map matcher adapters must return RawMapMatch objects")
            if item.source_index != expected_index:
                raise ValueError("Map matcher output must preserve source-point order")
            observations.append(
                MapMatchObservation(
                    source_index=item.source_index,
                    source_point=trace.points[item.source_index],
                    matched_point=item.matched_point,
                    confidence=item.confidence,
                    edge_id=item.edge_id,
                )
            )

        return MapMatchResult(trip_id=trace.trip_id, observations=tuple(observations))
