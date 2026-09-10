"""Simple traffic baselines for assessing whether learned models add value."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from ._temporal import (
    future_timestamps,
    positive_frequency,
    positive_integer,
    regular_traffic_frame,
)


class SeasonalNaiveForecaster:
    """Repeat the last observed season without learning any parameters.

    ``season_length=1`` is persistence. At hourly frequency, 24 repeats the
    previous 24 observations (elapsed hours, not necessarily a local DST day).
    ``fit_frame`` replaces history and ``forecast`` does not mutate it.
    """

    def __init__(self, season_length: int = 24, *, frequency: str = "h") -> None:
        positive_integer(season_length, "season_length")
        positive_frequency(frequency)
        self.season_length = season_length
        self.frequency = frequency
        self._season: np.ndarray | None = None
        self._last_timestamp: pd.Timestamp | None = None

    def fit_frame(
        self,
        frame: pd.DataFrame,
        *,
        timestamp_column: str = "date_time",
        target_column: str = "traffic_volume",
    ) -> SeasonalNaiveForecaster:
        """Store the final season from validated, regular traffic observations."""

        ordered = regular_traffic_frame(
            frame, timestamp_column, target_column, self.frequency
        )
        if len(ordered) < self.season_length:
            raise ValueError(f"Need at least {self.season_length} observations for a season")
        self._season = ordered[target_column].to_numpy(dtype=float)[-self.season_length :].copy()
        self._last_timestamp = ordered.iloc[-1][timestamp_column]
        return self

    def forecast(
        self,
        start: datetime | pd.Timestamp | str,
        *,
        periods: int,
        frequency: str = "h",
    ) -> pd.DataFrame:
        timestamps = future_timestamps(
            self._last_timestamp, start, periods, frequency, self.frequency
        )
        assert self._season is not None
        return pd.DataFrame({
            "timestamp": timestamps,
            "predicted_traffic_volume": np.resize(self._season, periods),
        })
