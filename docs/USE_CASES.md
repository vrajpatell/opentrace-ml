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

## 3. Route comparison service

Use `RouteSignals` and `score_route` inside a small API or batch job to compare
candidate routes using predicted traffic and detected hazards. The scoring is
transparent and intended as a baseline, not a production routing policy.

## 4. Model adapter integration

Wrap output from an external detector with `CallableDetector`. The downstream
geolocation and GeoJSON code then remains independent of the chosen CV model.

## Not ready yet

- production navigation decisions;
- real-time phone tracking;
- automatic edits to OpenStreetMap;
- safety-critical road-condition alerts;
- claims of model accuracy without evaluation on the target geography.

