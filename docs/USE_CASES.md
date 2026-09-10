# Current-stage use cases

OpenTrace ML 0.1–0.2 is useful as a small pipeline component rather than a
complete navigation product.

## 1. Road-survey annotation export

Input one RDD2022-style annotation and an ordered GPS trace, interpolate the
detection location, and export a GeoJSON layer. This can support a GIS demo,
municipal road-inspection prototype, or annotation quality check.

```bash
python examples/road_damage_route_demo.py tests/fixtures/rdd_sample.xml
```

Open `road_damage_demo.geojson` in geojson.io, QGIS, or a MapLibre application.

## 2. Traffic forecast baseline

Download the public UCI traffic-volume dataset, perform rolling out-of-sample
evaluation, and report MAE, RMSE, and MAPE.

```bash
pip install -e '.[data]'
python examples/traffic_backtest.py
```

This is appropriate for validating the forecasting API before adding a larger
spatiotemporal model.

For a small ML experiment, compare persistence, yesterday's hourly pattern,
incremental linear learning, and a CPU neural network on the same folds:

```bash
python examples/benchmark_traffic_models.py --synthetic-demo
python examples/benchmark_traffic_models.py --uci
```

Use the JSON report in a notebook or a dashboard showing error by forecast lead.
The synthetic run verifies the workflow; evaluate public or consented data from
your target setting before claiming accuracy. See the
[neural forecasting guide](NEURAL_FORECASTING.md).

## 3. Route comparison service

Use `RouteSignals` and `score_route` inside a small API or batch job to compare
candidate routes using predicted traffic and detected hazards. The scoring is
transparent and intended as a baseline, not a production routing policy.

## 4. Model adapter integration

Wrap output from an external detector with `CallableDetector`. The downstream
geolocation and GeoJSON code then remains independent of the chosen CV model.

## 5. Native Go traffic forecasting

For native Go services and CLIs, use the [portable traffic model](STAGE_5.md) to
train in Python and perform local forecasts in Go. This supports an offline
dashboard backend, a scheduled traffic report, or an experimental route-scoring
component without shipping a Python runtime alongside the Go application.

## Not ready yet

- production navigation decisions;
- real-time phone tracking;
- automatic edits to OpenStreetMap;
- safety-critical road-condition alerts;
- claims of model accuracy without evaluation on the target geography.
