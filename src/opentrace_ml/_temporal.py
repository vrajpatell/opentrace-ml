"""Shared, private validation helpers for temporal ML inputs."""

from __future__ import annotations

from datetime import datetime
from numbers import Integral

import numpy as np
import pandas as pd


def positive_integer(value: int, name: str, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


def positive_frequency(frequency: str) -> pd.DateOffset:
    try:
        offset = pd.tseries.frequencies.to_offset(frequency)
    except (TypeError, ValueError) as error:
        raise ValueError("frequency must be a valid positive pandas frequency") from error
    if offset is None or offset.n <= 0:
        raise ValueError("frequency must be a valid positive pandas frequency")
    return offset


def validated_traffic_frame(
    frame: pd.DataFrame,
    timestamp_column: str,
    target_column: str,
    *,
    allow_duplicates: bool = False,
) -> pd.DataFrame:
    """Return a sorted copy; never silently discard or impute observations."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Traffic observations must be a pandas DataFrame")
    if timestamp_column == target_column or not frame.columns.is_unique:
        raise ValueError("Timestamp and target columns must be distinct and unambiguous")
    required = {timestamp_column, target_column}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("At least one traffic observation is required")

    ordered = frame.loc[:, [timestamp_column, target_column]].copy()
    try:
        timestamps = pd.DatetimeIndex(pd.to_datetime(ordered[timestamp_column]))
    except (TypeError, ValueError) as error:
        raise ValueError("Timestamps must be valid and use a consistent timezone") from error
    if timestamps.hasnans:
        raise ValueError("Timestamps cannot be missing")
    if not allow_duplicates and timestamps.has_duplicates:
        raise ValueError("Duplicate timestamps must be resolved explicitly before evaluation")
    try:
        values = np.asarray(ordered[target_column], dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Traffic values must be finite non-negative numbers") from error
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Traffic values must be finite non-negative numbers")
    ordered[timestamp_column] = timestamps
    ordered[target_column] = values
    return ordered.sort_values(timestamp_column, kind="stable").reset_index(drop=True)


def regular_traffic_frame(
    frame: pd.DataFrame, timestamp_column: str, target_column: str, frequency: str
) -> pd.DataFrame:
    """Validate a complete series at the caller's declared cadence."""

    offset = positive_frequency(frequency)
    ordered = validated_traffic_frame(frame, timestamp_column, target_column)
    timestamps = pd.DatetimeIndex(ordered[timestamp_column])
    expected = pd.date_range(timestamps[0], periods=len(timestamps), freq=offset)
    if not timestamps.equals(expected):
        raise ValueError("Timestamps must form an uninterrupted grid at the requested frequency")
    return ordered


def future_timestamps(
    last: pd.Timestamp | None,
    start: datetime | pd.Timestamp | str,
    periods: int,
    frequency: str,
    training_frequency: str,
) -> pd.DatetimeIndex:
    """Require an immediately following forecast at the training cadence."""

    positive_integer(periods, "periods")
    offset = positive_frequency(frequency)
    if last is None:
        raise RuntimeError("Fit the forecaster before requesting a forecast")
    if offset != positive_frequency(training_frequency):
        raise ValueError("Forecast frequency must match the training frequency")
    parsed = pd.Timestamp(start)
    if pd.isna(parsed) or str(parsed.tz) != str(last.tz):
        raise ValueError("Forecast start must be valid and use the training timezone")
    expected_start = pd.date_range(last, periods=2, freq=offset)[1]
    if parsed != expected_start:
        raise ValueError("Forecast start must immediately follow the last training timestamp")
    return pd.date_range(parsed, periods=periods, freq=offset)
