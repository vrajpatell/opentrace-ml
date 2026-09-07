"""Versioned, data-only traffic model snapshots and deterministic inference."""

from __future__ import annotations

import json
import math
import re
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

TRAFFIC_MODEL_FORMAT = "opentrace.traffic-linear.v1"
TRAFFIC_FEATURE_LAYOUT = "wall_clock_calendar5_lags_oldest_first"
MAX_TRAFFIC_MODEL_BYTES = 1 << 20
MAX_TRAFFIC_LAGS = 4096
_FIELDS = frozenset({
    "format", "feature_layout", "lags", "mean", "scale", "coefficients", "intercept", "history"
})
_RFC3339 = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])"
)


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} exceeds float64 range") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _timestamp(value: datetime | str) -> datetime:
    if isinstance(value, str):
        if _RFC3339.fullmatch(value) is None:
            raise ValueError("Timestamp must be RFC3339 with an explicit UTC offset")
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise ValueError("Timestamp must be a datetime or RFC3339 string")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp must have an explicit UTC offset")
    if not 1 <= value.year <= 9999:
        raise ValueError("Timestamp must have a valid calendar year")
    if value.utcoffset().total_seconds() % 60 or getattr(value, "nanosecond", 0):
        raise ValueError("Timestamps require whole-minute offsets and at most microsecond precision")
    try:
        value.astimezone(timezone.utc)
    except (OverflowError, ValueError) as error:
        raise ValueError("Timestamp UTC year is outside the supported range") from error
    return value


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate model field: {name}")
        result[name] = value
    return result


def _reject_constant(value: str):
    raise ValueError(f"Non-finite JSON number: {value}")


@dataclass(frozen=True, slots=True)
class PortableTrafficModel:
    """An immutable inference snapshot; contains no executable serialized objects.

    Calendar features use the timestamp's displayed local hour/weekday, not an
    implicit UTC conversion. Lags represent observations in oldest-first order.
    Retraining and observation updates remain in OnlineTrafficForecaster.
    """

    lags: int
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    history: tuple[float, ...]
    format: str = TRAFFIC_MODEL_FORMAT
    feature_layout: str = TRAFFIC_FEATURE_LAYOUT

    def __post_init__(self) -> None:
        if self.format != TRAFFIC_MODEL_FORMAT or self.feature_layout != TRAFFIC_FEATURE_LAYOUT:
            raise ValueError("Unsupported traffic model format or feature layout")
        if isinstance(self.lags, bool) or not isinstance(self.lags, int):
            raise ValueError("lags must be an integer")
        if not 1 <= self.lags <= MAX_TRAFFIC_LAGS:
            raise ValueError(f"lags must be between 1 and {MAX_TRAFFIC_LAGS}")
        for name in ("mean", "scale", "coefficients", "history"):
            values = getattr(self, name)
            expected = self.lags if name == "history" else 5 + self.lags
            if not isinstance(values, (list, tuple)) or len(values) != expected:
                raise ValueError(f"{name} must contain exactly {expected} numbers")
            normalized = tuple(_number(value, name) for value in values)
            if name == "scale" and any(value <= 0 for value in normalized):
                raise ValueError("scale values must be positive")
            if name == "history" and any(value < 0 for value in normalized):
                raise ValueError("history values must be non-negative")
            object.__setattr__(self, name, normalized)
        object.__setattr__(self, "intercept", _number(self.intercept, "intercept"))

    def to_dict(self) -> dict[str, object]:
        """Return an owned JSON-compatible snapshot of inference parameters."""
        return {
            "format": self.format,
            "feature_layout": self.feature_layout,
            "lags": self.lags,
            "mean": list(self.mean),
            "scale": list(self.scale),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PortableTrafficModel:
        """Validate exact fields, dimensions, versions, and finite parameters."""
        if not isinstance(value, Mapping) or set(value) != _FIELDS:
            raise ValueError("Traffic model must contain exactly the v1 fields")
        return cls(**value)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str | bytes) -> PortableTrafficModel:
        if not isinstance(payload, (str, bytes)):
            raise ValueError("Model JSON must be text or UTF-8 bytes")
        encoded = payload.encode("utf-8") if isinstance(payload, str) else payload
        if len(encoded) > MAX_TRAFFIC_MODEL_BYTES:
            raise ValueError("Traffic model JSON exceeds the 1 MiB limit")
        try:
            decoded = json.loads(
                encoded.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeError, RecursionError) as error:
            raise ValueError("Invalid traffic model JSON") from error
        return cls.from_dict(decoded)

    def save(self, path: str | Path) -> None:
        """Write a data-only JSON snapshot. The destination is replaced."""
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> PortableTrafficModel:
        with Path(path).open("rb") as stream:
            return cls.from_json(stream.read(MAX_TRAFFIC_MODEL_BYTES + 1))

    def _predict(self, timestamp: datetime, history: Iterable[float]) -> float:
        hour = timestamp.hour + timestamp.minute / 60.0
        weekday = timestamp.weekday()
        calendar = (
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            math.sin(2 * math.pi * weekday / 7),
            math.cos(2 * math.pi * weekday / 7),
            float(weekday >= 5),
        )
        # Keep the scaler operation order; folding scale into coefficients can
        # change floating-point cancellation relative to sklearn predictions.
        value = 0.0
        for index, feature in enumerate((*calendar, *history)):
            value += ((feature - self.mean[index]) / self.scale[index]) * self.coefficients[index]
        value += self.intercept
        if not math.isfinite(value):
            raise ValueError("Traffic prediction exceeds float64 range")
        return max(0.0, value)

    def predict(self, timestamp: datetime | str) -> float:
        """Predict in O(lags) without changing model parameters or lag history."""
        return self._predict(_timestamp(timestamp), self.history)

    def forecast(self, timestamps: Iterable[datetime | str]) -> list[float]:
        """Recursively predict strictly increasing instants without mutating state.

        Timestamp selection is explicit so applications own cadence and DST
        policy. Each prediction becomes the next lag observation locally.
        """
        history = deque(self.history, maxlen=self.lags)
        predictions = []
        previous = None
        for timestamp in timestamps:
            parsed = _timestamp(timestamp)
            instant = parsed.astimezone(timezone.utc)
            if previous is not None and instant <= previous:
                raise ValueError("Forecast timestamps must be strictly increasing instants")
            value = self._predict(parsed, history)
            predictions.append(value)
            history.append(value)
            previous = instant
        if not predictions:
            raise ValueError("At least one forecast timestamp is required")
        return predictions
