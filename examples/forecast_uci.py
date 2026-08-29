"""Train a small incremental baseline using public UCI traffic data."""

from opentrace_ml.datasets import fetch_uci_traffic_volume
from opentrace_ml.forecasting import OnlineTrafficForecaster


frame = fetch_uci_traffic_volume().sort_values("date_time").tail(5_000)
model = OnlineTrafficForecaster(lags=24).fit_frame(frame.iloc[:-24])
start = frame.iloc[-24]["date_time"]
print(model.forecast(start, periods=24))

