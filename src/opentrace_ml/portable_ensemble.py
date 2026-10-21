"""Data-only export of selected built-in horizon ensemble components."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from .portable_forecasting import (
    PortableTrafficModel,
    _number,
    _reject_constant,
    _timestamp,
    _unique_object,
)
from .portable_neural import MAX_NEURAL_BYTES, PortableNeuralModel

ENSEMBLE_MODEL_FORMAT = "opentrace.horizon-ensemble.v1"


@dataclass(frozen=True, slots=True, init=False)
class PortableEnsembleModel:
    """An owned canonical JSON payload; custom Python callbacks are never exported."""

    _payload: str

    def __init__(self):
        raise TypeError("Use from_dict or from_json to validate a portable ensemble")

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {
            "format",
            "last_timestamp",
            "frequency_seconds",
            "selected_models",
            "candidates",
        }:
            raise ValueError("Invalid ensemble fields")
        if value["format"] != ENSEMBLE_MODEL_FORMAT:
            raise ValueError("Unsupported ensemble format")
        last = _timestamp(value["last_timestamp"])
        frequency = value["frequency_seconds"]
        if (
            isinstance(frequency, bool)
            or not isinstance(frequency, int)
            or not 1 <= frequency <= 31536000
        ):
            raise ValueError("Invalid ensemble frequency")
        path = value["selected_models"]
        candidates = value["candidates"]
        if (
            not isinstance(path, list)
            or not 1 <= len(path) <= 4096
            or any(not isinstance(x, str) or not x for x in path)
        ):
            raise ValueError("Invalid selected path")
        if (
            not isinstance(candidates, dict)
            or not 1 <= len(candidates) <= 64
            or set(path) != set(candidates)
        ):
            raise ValueError("Selected models and components must match")
        for component in candidates.values():
            if not isinstance(component, dict):
                raise TypeError("Invalid component")
            kind = component.get("format")
            if kind == "opentrace.traffic-mlp.v1":
                model = PortableNeuralModel.from_dict(component)
                stamp = _timestamp(model.last_timestamp)
                if (
                    stamp != last
                    or stamp.utcoffset() != last.utcoffset()
                    or model.frequency_seconds != frequency
                ):
                    raise ValueError("Neural component history differs from ensemble")
            elif kind == "opentrace.traffic-linear.v1":
                PortableTrafficModel.from_dict(component)
            elif kind == "opentrace.seasonal.v1":
                if (
                    set(component) != {"format", "season"}
                    or not isinstance(component["season"], (list, tuple))
                    or not 1 <= len(component["season"]) <= 4096
                ):
                    raise ValueError("Invalid seasonal component")
                if any(_number(x, "season") < 0 for x in component["season"]):
                    raise ValueError("Negative season")
            else:
                raise ValueError("Unsupported ensemble component")
        payload = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        if len(payload.encode()) > MAX_NEURAL_BYTES:
            raise ValueError("Ensemble exceeds 16 MiB")
        instance = object.__new__(cls)
        object.__setattr__(instance, "_payload", payload)
        return instance

    def to_dict(self):
        return json.loads(self._payload)

    def to_json(self):
        return self._payload

    @classmethod
    def from_json(cls, payload):
        if not isinstance(payload, (str, bytes)):
            raise TypeError("Expected JSON text or bytes")
        raw = payload.encode() if isinstance(payload, str) else payload
        if len(raw) > MAX_NEURAL_BYTES:
            raise ValueError("Ensemble exceeds 16 MiB")
        try:
            value = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeError, RecursionError) as e:
            raise ValueError("Invalid ensemble JSON") from e
        return cls.from_dict(value)

    def save(self, path):
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as stream:
            return cls.from_json(stream.read(MAX_NEURAL_BYTES + 1))

    def forecast(self, timestamps):
        data = self.to_dict()
        times = list(timestamps)
        last = _timestamp(data["last_timestamp"])
        offset = last.utcoffset()
        if not 1 <= len(times) <= len(data["selected_models"]):
            raise ValueError("Invalid forecast horizon")
        for t in times:
            stamp = _timestamp(t)
            if stamp.utcoffset() != offset or stamp != last + timedelta(
                seconds=data["frequency_seconds"]
            ):
                raise ValueError("Invalid ensemble timestamp grid")
            last = stamp
        predictions = {}
        for name in set(data["selected_models"][: len(times)]):
            component = data["candidates"][name]
            kind = component["format"]
            if kind == "opentrace.traffic-mlp.v1":
                values = PortableNeuralModel.from_dict(component).forecast(times)
            elif kind == "opentrace.traffic-linear.v1":
                values = PortableTrafficModel.from_dict(component).forecast(times)
            else:
                values = [
                    component["season"][i % len(component["season"])] for i in range(len(times))
                ]
            predictions[name] = values
        return [
            predictions[name][i] for i, name in enumerate(data["selected_models"][: len(times)])
        ]
