"""CPU neural-network traffic forecasts with training-only preprocessing."""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ._temporal import (
    future_timestamps,
    positive_frequency,
    positive_integer,
    regular_traffic_frame,
)


class NeuralTrafficForecaster:
    """Batch-trained multilayer perceptron using lag and calendar features.

    Input and target scalers are fitted only on training examples. Each example
    uses observations strictly before its target time. ``fit_frame`` replaces
    learned state; use OnlineTrafficForecaster for incremental updates instead.
    Multi-step forecasts recursively consume predictions and never change the
    fitted history. This experimental point forecaster has no uncertainty model.
    """

    def __init__(
        self,
        lags: int = 24,
        hidden_layer_sizes: tuple[int, ...] = (32, 16),
        *,
        frequency: str = "h",
        alpha: float = 0.01,
        max_iter: int = 500,
        random_state: int = 42,
    ) -> None:
        positive_integer(lags, "lags")
        positive_integer(max_iter, "max_iter")
        positive_integer(random_state, "random_state", minimum=0)
        positive_frequency(frequency)
        if not hidden_layer_sizes:
            raise ValueError("At least one hidden layer is required")
        for width in hidden_layer_sizes:
            positive_integer(width, "hidden layer width")
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and non-negative")
        self.lags = lags
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.frequency = frequency
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self._model: TransformedTargetRegressor | None = None
        self._history: tuple[float, ...] = ()
        self._last_timestamp: pd.Timestamp | None = None

    @staticmethod
    def _features(timestamp: pd.Timestamp, history: list[float]) -> list[float]:
        hour = timestamp.hour + timestamp.minute / 60 + timestamp.second / 3600
        weekday = timestamp.dayofweek
        return history + [
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            math.sin(2 * math.pi * weekday / 7),
            math.cos(2 * math.pi * weekday / 7),
            float(weekday >= 5),
        ]

    def fit_frame(
        self,
        frame: pd.DataFrame,
        *,
        timestamp_column: str = "date_time",
        target_column: str = "traffic_volume",
    ) -> NeuralTrafficForecaster:
        """Fit a fresh network on at least ``lags + 2`` regular observations.

        Duplicate timestamps, gaps, missing values, and negative/non-finite
        volumes are rejected. Shuffled input is sorted without modifying it.
        Failed validation leaves any previously fitted model intact.
        """

        ordered = regular_traffic_frame(
            frame, timestamp_column, target_column, self.frequency
        )
        if len(ordered) < self.lags + 2:
            raise ValueError(f"Need at least {self.lags + 2} observations to train the network")
        values = ordered[target_column].to_numpy(dtype=float)
        timestamps = pd.DatetimeIndex(ordered[timestamp_column])
        features = np.asarray([
            self._features(timestamps[i], values[i - self.lags : i].tolist())
            for i in range(self.lags, len(values))
        ])
        # L-BFGS uses all training examples; no random validation split or hidden
        # early-stopping holdout can mix future observations into preprocessing.
        model = TransformedTargetRegressor(
            regressor=make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=self.hidden_layer_sizes,
                    activation="relu",
                    solver="lbfgs",
                    alpha=self.alpha,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                ),
            ),
            transformer=StandardScaler(),
        )
        model.fit(features, values[self.lags :])
        self._model = model
        self._history = tuple(values[-self.lags :])
        self._last_timestamp = timestamps[-1]
        return self

    def forecast(
        self,
        start: datetime | pd.Timestamp | str,
        *,
        periods: int,
        frequency: str = "h",
    ) -> pd.DataFrame:
        """Recursively forecast consecutive steps immediately after training.

        The requested frequency and timezone must match training. Negative
        estimates are clipped to zero; non-finite model outputs raise an error.
        Convergence warnings during fitting remain visible to callers.
        """

        timestamps = future_timestamps(
            self._last_timestamp, start, periods, frequency, self.frequency
        )
        assert self._model is not None
        history = deque(self._history, maxlen=self.lags)
        predictions: list[float] = []
        for timestamp in timestamps:
            features = np.asarray([self._features(timestamp, list(history))])
            value = float(self._model.predict(features)[0])
            if not math.isfinite(value):
                raise ValueError("The neural network produced a non-finite forecast")
            value = max(0.0, value)
            predictions.append(value)
            history.append(value)
        return pd.DataFrame({"timestamp": timestamps, "predicted_traffic_volume": predictions})
