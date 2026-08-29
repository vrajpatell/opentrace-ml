"""Adapter contracts for integrating external computer-vision models."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np

from .models import BoundingBox, Detection

ImageInput: TypeAlias = str | Path | np.ndarray


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

