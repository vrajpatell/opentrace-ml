"""Backtest the incremental forecaster on public UCI traffic data."""

from __future__ import annotations

import json

from opentrace_ml import OnlineTrafficForecaster, regression_metrics, rolling_backtest
from opentrace_ml.datasets import fetch_uci_traffic_volume


def main() -> None:
    frame = fetch_uci_traffic_volume().sort_values("date_time").tail(720)
    result = rolling_backtest(
        frame,
        forecaster_factory=lambda: OnlineTrafficForecaster(lags=24),
        initial_window=480,
        horizon=24,
        step=24,
    )
    metrics = regression_metrics(result["actual"], result["predicted"])
    print(json.dumps(metrics.as_dict(), indent=2))


if __name__ == "__main__":
    main()

