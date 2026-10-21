"""Bounded data-only MLP inference with an explicit fixed-offset time contract."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from .portable_forecasting import _number, _reject_constant, _timestamp, _unique_object

NEURAL_MODEL_FORMAT = "opentrace.traffic-mlp.v1"
NEURAL_FEATURE_LAYOUT = "lags_oldest_first_wall_clock_calendar5_seconds"
MAX_NEURAL_BYTES = 16 << 20
MAX_NEURAL_PARAMETERS = 1_000_000
_FIELDS = {
    "format",
    "feature_layout",
    "lags",
    "mean",
    "scale",
    "weights",
    "biases",
    "target_mean",
    "target_scale",
    "history",
    "last_timestamp",
    "frequency_seconds",
}


@dataclass(frozen=True, slots=True)
class PortableNeuralModel:
    """Immutable ReLU hidden layers, linear output, and both training scalers.

    Weight matrices use input-major rows, exactly as sklearn MLPRegressor.
    v1 supports fixed UTC offsets and an integer-second cadence, not DST zones.
    """

    lags: int
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[tuple[tuple[float, ...], ...], ...]
    biases: tuple[tuple[float, ...], ...]
    target_mean: float
    target_scale: float
    history: tuple[float, ...]
    last_timestamp: str
    frequency_seconds: int
    format: str = NEURAL_MODEL_FORMAT
    feature_layout: str = NEURAL_FEATURE_LAYOUT

    def __post_init__(self):
        if self.format != NEURAL_MODEL_FORMAT or self.feature_layout != NEURAL_FEATURE_LAYOUT:
            raise ValueError("Unsupported neural model format or feature layout")
        for name, upper in [("lags", 4096), ("frequency_seconds", 31536000)]:
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= upper:
                raise ValueError(f"{name} is outside its integer bounds")
        if not isinstance(self.last_timestamp, str):
            raise TypeError("last_timestamp must be RFC3339 text")
        _timestamp(self.last_timestamp)
        for name in ("mean", "scale", "history"):
            raw = getattr(self, name)
            if not isinstance(raw, (list, tuple)) or len(raw) != (
                self.lags if name == "history" else self.lags + 5
            ):
                raise ValueError(f"Invalid {name} shape")
            values = tuple(_number(x, name) for x in raw)
            if name == "scale" and any(x <= 0 for x in values):
                raise ValueError("Input scales must be positive")
            if name == "history" and any(x < 0 for x in values):
                raise ValueError("History must be non-negative")
            object.__setattr__(self, name, values)
        for name in ("target_mean", "target_scale"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if self.target_scale <= 0:
            raise ValueError("Target scale must be positive")
        if not isinstance(self.weights, (list, tuple)) or not 2 <= len(self.weights) <= 16:
            raise ValueError("Expected 1–15 hidden layers plus output")
        if not isinstance(self.biases, (list, tuple)) or len(self.biases) != len(self.weights):
            raise ValueError("Bias/layer count mismatch")
        width, parameters, matrices, biases = self.lags + 5, 0, [], []
        for index, (matrix, bias) in enumerate(zip(self.weights, self.biases)):
            if not isinstance(bias, (list, tuple)) or not 1 <= len(bias) <= 4096:
                raise ValueError("Invalid layer width")
            if not isinstance(matrix, (list, tuple)) or len(matrix) != width:
                raise ValueError("Invalid layer input shape")
            out = len(bias)
            if index == len(self.weights) - 1 and out != 1:
                raise ValueError("Output layer must have width one")
            parameters += width * out + out
            if parameters > MAX_NEURAL_PARAMETERS:
                raise ValueError("Neural parameter limit exceeded")
            rows = []
            for row in matrix:
                if not isinstance(row, (list, tuple)) or len(row) != out:
                    raise ValueError("Invalid weight row shape")
                rows.append(tuple(_number(x, "weight") for x in row))
            matrices.append(tuple(rows))
            biases.append(tuple(_number(x, "bias") for x in bias))
            width = out
        object.__setattr__(self, "weights", tuple(matrices))
        object.__setattr__(self, "biases", tuple(biases))

    def to_dict(self):
        return {name: getattr(self, name) for name in _FIELDS}

    def to_json(self):
        payload = json.dumps(self.to_dict(), allow_nan=False, sort_keys=True, separators=(",", ":"))
        if len(payload.encode()) > MAX_NEURAL_BYTES:
            raise ValueError("Neural model exceeds 16 MiB limit")
        return payload

    @classmethod
    def from_dict(cls, value: Mapping):
        if not isinstance(value, Mapping) or set(value) != _FIELDS:
            raise ValueError("Neural model must contain exactly the v1 fields")
        try:
            return cls(**value)
        except (TypeError, OverflowError) as error:
            raise ValueError("Invalid neural parameter type") from error

    @classmethod
    def from_json(cls, payload: str | bytes):
        if not isinstance(payload, (str, bytes)):
            raise TypeError("Neural model JSON must be text or bytes")
        encoded = payload.encode() if isinstance(payload, str) else payload
        if len(encoded) > MAX_NEURAL_BYTES:
            raise ValueError("Neural model exceeds 16 MiB limit")
        try:
            value = json.loads(
                encoded.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeError, RecursionError) as error:
            raise ValueError("Invalid neural JSON") from error
        return cls.from_dict(value)

    def save(self, path: str | Path):
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        with Path(path).open("rb") as stream:
            return cls.from_json(stream.read(MAX_NEURAL_BYTES + 1))

    def forecast(self, timestamps: Iterable) -> list[float]:
        history = list(self.history)
        last = _timestamp(self.last_timestamp)
        offset = last.utcoffset()
        result = []
        for timestamp in timestamps:
            parsed = _timestamp(timestamp)
            if parsed.utcoffset() != offset or parsed != last + timedelta(
                seconds=self.frequency_seconds
            ):
                raise ValueError(
                    "Forecast must immediately follow history at the fixed offset and cadence"
                )
            hour = parsed.hour + parsed.minute / 60 + parsed.second / 3600
            weekday = parsed.weekday()
            features = history + [
                math.sin(2 * math.pi * hour / 24),
                math.cos(2 * math.pi * hour / 24),
                math.sin(2 * math.pi * weekday / 7),
                math.cos(2 * math.pi * weekday / 7),
                float(weekday >= 5),
            ]
            values = [(x - mean) / scale for x, mean, scale in zip(features, self.mean, self.scale)]
            for layer, (weights, bias) in enumerate(zip(self.weights, self.biases)):
                values = [
                    sum(values[i] * weights[i][j] for i in range(len(values))) + b
                    for j, b in enumerate(bias)
                ]
                if not all(math.isfinite(x) for x in values):
                    raise ValueError("Neural inference exceeded float64 range")
                if layer < len(self.weights) - 1:
                    values = [max(0.0, x) for x in values]
            value = values[0] * self.target_scale + self.target_mean
            if not math.isfinite(value):
                raise ValueError("Neural inference exceeded float64 range")
            value = max(0.0, value)
            result.append(value)
            history = history[1:] + [value]
            last = parsed
        if not result:
            raise ValueError("At least one timestamp is required")
        return result

    def predict(self, timestamp) -> float:
        return self.forecast([timestamp])[0]
