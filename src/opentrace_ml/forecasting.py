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


class OnlineTrafficForecaster:
    """Small online forecaster based on calendar signals and recent observations.

    ``update`` calls ``partial_fit`` so a running service can learn from each new
    traffic observation without retraining from scratch.
    """

    def __init__(self, lags: int = 6, random_state: int = 42) -> None:
        if lags < 1:
            raise ValueError("lags must be at least 1")
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

    @staticmethod
    def _as_datetime(value: datetime | pd.Timestamp | str) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            raise ValueError("Timestamp cannot be missing")
        return timestamp

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
        if len(self._history) == self.lags:
            features = self._features(parsed, self._history)
            self._scaler.partial_fit(features)
            transformed = self._scaler.transform(features)
            self._model.partial_fit(transformed, np.asarray([volume]))
            self._fitted = True
        self._history.append(volume)

    def fit_frame(
        self,
        frame: pd.DataFrame,
        *,
        timestamp_column: str = "date_time",
        target_column: str = "traffic_volume",
    ) -> OnlineTrafficForecaster:
        """Fit incrementally from a time-ordered pandas DataFrame."""

        required = {timestamp_column, target_column}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        ordered = frame.loc[:, [timestamp_column, target_column]].copy()
        ordered[timestamp_column] = pd.to_datetime(ordered[timestamp_column])
        ordered = ordered.dropna().sort_values(timestamp_column)
        for timestamp, volume in ordered.itertuples(index=False, name=None):
            self.update(timestamp, volume)
        return self

    def predict(self, timestamp: datetime | pd.Timestamp | str) -> float:
        """Predict traffic volume for one timestamp using current history."""

        if not self._fitted:
            raise RuntimeError("The forecaster has not received enough training observations")
        features = self._features(self._as_datetime(timestamp), self._history)
        prediction = float(self._model.predict(self._scaler.transform(features))[0])
        return max(0.0, prediction)

    def forecast(
        self,
        start: datetime | pd.Timestamp | str,
        *,
        periods: int,
        frequency: str = "h",
    ) -> pd.DataFrame:
        """Produce a recursive multi-step forecast without mutating the model."""

        if periods < 1:
            raise ValueError("periods must be at least 1")
        if not self._fitted:
            raise RuntimeError("The forecaster has not received enough training observations")

        timestamps = pd.date_range(start=self._as_datetime(start), periods=periods, freq=frequency)
        history: deque[float] = deque(self._history, maxlen=self.lags)
        predictions: list[float] = []
        for timestamp in timestamps:
            features = self._features(timestamp, history)
            value = max(0.0, float(self._model.predict(self._scaler.transform(features))[0]))
            predictions.append(value)
            history.append(value)
        return pd.DataFrame({"timestamp": timestamps, "predicted_traffic_volume": predictions})
