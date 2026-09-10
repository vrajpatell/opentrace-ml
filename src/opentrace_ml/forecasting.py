"""Incremental traffic-volume forecasting suitable for streaming updates."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from ._temporal import positive_frequency, positive_integer, validated_traffic_frame
from .portable_forecasting import PortableTrafficModel


class OnlineTrafficForecaster:
    """Small online forecaster based on calendar signals and recent observations.

    ``update`` calls ``partial_fit`` so a running service can learn from each new
    traffic observation without retraining from scratch.
    """

    def __init__(self, lags: int = 6, random_state: int = 42) -> None:
        positive_integer(lags, "lags")
        self.lags = lags
        self._history: deque[float] = deque(maxlen=lags)
        self._scaler = StandardScaler()
        self._model = SGDRegressor(
            loss="huber",
            penalty="l2",
            alpha=0.0001,
            learning_rate="adaptive",
            eta0=0.01,
            random_state=random_state,
        )
        self._fitted = False
        self._last_timestamp: pd.Timestamp | None = None

    @staticmethod
    def _as_datetime(value: datetime | pd.Timestamp | str) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            raise ValueError("Timestamp cannot be missing")
        return timestamp

    def _require_later_timestamp(self, timestamp: pd.Timestamp) -> None:
        if self._last_timestamp is None:
            return
        if str(timestamp.tz) != str(self._last_timestamp.tz):
            raise ValueError("Timestamps must use the same timezone as previous observations")
        if timestamp <= self._last_timestamp:
            raise ValueError("Timestamp must be strictly later than the last observation")

    def _features(self, timestamp: pd.Timestamp, history: Iterable[float]) -> np.ndarray:
        values = list(history)
        if len(values) < self.lags:
            raise RuntimeError(f"Need {self.lags} observations before forecasting")

        hour = timestamp.hour + timestamp.minute / 60.0
        weekday = timestamp.dayofweek
        calendar = [
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            math.sin(2 * math.pi * weekday / 7),
            math.cos(2 * math.pi * weekday / 7),
            float(weekday >= 5),
        ]
        return np.asarray([calendar + values[-self.lags :]], dtype=float)

    def update(self, timestamp: datetime | pd.Timestamp | str, observed_volume: float) -> None:
        """Learn from one newly observed traffic volume."""

        volume = float(observed_volume)
        if not math.isfinite(volume) or volume < 0:
            raise ValueError("Traffic volume must be a finite non-negative number")

        parsed = self._as_datetime(timestamp)
        self._require_later_timestamp(parsed)
        if len(self._history) == self.lags:
            features = self._features(parsed, self._history)
            self._scaler.partial_fit(features)
            transformed = self._scaler.transform(features)
            self._model.partial_fit(transformed, np.asarray([volume]))
            self._fitted = True
        self._history.append(volume)
        self._last_timestamp = parsed

    def fit_frame(
        self,
        frame: pd.DataFrame,
        *,
        timestamp_column: str = "date_time",
        target_column: str = "traffic_volume",
    ) -> OnlineTrafficForecaster:
        """Append a validated, sorted batch without resetting learned state.

        Duplicate/missing timestamps and invalid targets are rejected before any
        update. A later batch must start after the last learned observation.
        """

        ordered = validated_traffic_frame(frame, timestamp_column, target_column)
        self._require_later_timestamp(ordered.iloc[0][timestamp_column])
        for timestamp, volume in ordered.itertuples(index=False, name=None):
            self.update(timestamp, volume)
        return self

    def predict(self, timestamp: datetime | pd.Timestamp | str) -> float:
        """Predict traffic volume for one timestamp using current history."""

        if not self._fitted:
            raise RuntimeError("The forecaster has not received enough training observations")
        parsed = self._as_datetime(timestamp)
        self._require_later_timestamp(parsed)
        features = self._features(parsed, self._history)
        prediction = float(self._model.predict(self._scaler.transform(features))[0])
        return max(0.0, prediction)

    def export_model(self) -> PortableTrafficModel:
        """Copy fitted scaler, linear parameters, and lags for portable inference.

        This is an inference snapshot, not a checkpoint for resuming training.
        Callers must use the same timestamp wall-clock convention and sampling
        cadence as training. No Python objects or pickle payloads are serialized.
        """
        if not self._fitted:
            raise RuntimeError("The forecaster has not received enough training observations")
        return PortableTrafficModel(
            lags=self.lags,
            mean=tuple(self._scaler.mean_.tolist()),
            scale=tuple(self._scaler.scale_.tolist()),
            coefficients=tuple(self._model.coef_.tolist()),
            intercept=float(self._model.intercept_[0]),
            history=tuple(self._history),
        )

    def forecast(
        self,
        start: datetime | pd.Timestamp | str,
        *,
        periods: int,
        frequency: str = "h",
    ) -> pd.DataFrame:
        """Produce a recursive multi-step forecast without mutating the model."""

        positive_integer(periods, "periods")
        offset = positive_frequency(frequency)
        if not self._fitted:
            raise RuntimeError("The forecaster has not received enough training observations")

        parsed = self._as_datetime(start)
        self._require_later_timestamp(parsed)
        timestamps = pd.date_range(start=parsed, periods=periods, freq=offset)
        if timestamps[0] != parsed:
            raise ValueError("Forecast start must be aligned to the requested frequency")
        history: deque[float] = deque(self._history, maxlen=self.lags)
        predictions: list[float] = []
        for timestamp in timestamps:
            features = self._features(timestamp, history)
            value = max(0.0, float(self._model.predict(self._scaler.transform(features))[0]))
            predictions.append(value)
            history.append(value)
        return pd.DataFrame({"timestamp": timestamps, "predicted_traffic_volume": predictions})
