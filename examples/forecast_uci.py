"""Train a small incremental baseline using public UCI traffic data."""

from opentrace_ml.datasets import fetch_uci_traffic_volume, select_hourly_traffic_window
from opentrace_ml.forecasting import OnlineTrafficForecaster

frame = select_hourly_traffic_window(fetch_uci_traffic_volume(), samples=720)
model = OnlineTrafficForecaster(lags=24).fit_frame(frame.iloc[:-24])
start = frame.iloc[-24]["date_time"]
print(model.forecast(start, periods=24))
