"""Adapter contracts for detection and traffic forecasting models."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np
import pandas as pd

from .models import BoundingBox, Detection

ImageInput: TypeAlias = str | Path | np.ndarray


@runtime_checkable
class TrafficForecaster(Protocol):
    """Train/forecast interface accepted by rolling backtests.

    Factories must supply a fresh model per fold. Forecasts contain exactly the
    requested timestamps and a finite, non-negative predicted_traffic_volume.
    """

    def fit_frame(
        self,
        frame: pd.DataFrame,
        *,
        timestamp_column: str = "date_time",
        target_column: str = "traffic_volume",
    ) -> TrafficForecaster: ...

    def forecast(
        self,
        start: datetime | pd.Timestamp | str,
        *,
        periods: int,
        frequency: str = "h",
    ) -> pd.DataFrame: ...


@dataclass(frozen=True, slots=True)
class RawDetection:
    """Model output before trip time and frame metadata are attached."""

    label: str
    confidence: float
    bbox: BoundingBox


@runtime_checkable
class Detector(Protocol):
    """Minimal interface implemented by computer-vision detector adapters."""

    def predict(
        self,
        image: ImageInput,
        *,
        timestamp_seconds: float = 0.0,
        frame_id: str | None = None,
    ) -> Sequence[Detection]:
        """Return normalized detections for one image or frame."""


@dataclass(slots=True)
class CallableDetector:
    """Wrap a simple prediction callable in the OpenTrace detector contract."""

    predictor: Callable[[ImageInput], Iterable[RawDetection]]

    def predict(
        self,
        image: ImageInput,
        *,
        timestamp_seconds: float = 0.0,
        frame_id: str | None = None,
    ) -> list[Detection]:
        return [
            Detection(
                label=item.label,
                confidence=item.confidence,
                bbox=item.bbox,
                timestamp_seconds=timestamp_seconds,
                frame_id=frame_id,
            )
            for item in self.predictor(image)
        ]
