# OpenTrace ML roadmap

## 0.1 — Common contracts and public-data adapters

- Parse RDD2022 Pascal VOC road-damage annotations.
- Geolocate timestamped detections on GPS traces.
- Incrementally forecast traffic volume.
- Convert routes and detections to GeoJSON.
- Score routes from damage, traffic, map confidence, and distance signals.

## 0.2 — Reproducible baselines

- [x] Add a detector adapter protocol and callable reference adapter.
- [x] Add detection metrics and rolling traffic backtesting.
- [x] Add runnable current-stage mini-projects.
- [ ] Add an Apache-2.0-compatible reference detection model.
- Add train/evaluate commands for an RDD2022 subset.
- Add small cached OSM extracts for integration tests without hitting public services.

## 0.3 — Route intelligence

- Connect forecasts to road-network edges.
- Map-match GPS traces with Valhalla.
- Cluster repeated unmatched traces as candidate missing roads.
- Require human review before exporting road-layer changes.

## 0.4 — Streaming and web integration

- Publish detection, forecast, and route events through a stable JSON schema.
- Add a FastAPI reference service.
- Add a MapLibre example application.
- Measure end-to-end latency and synchronization error.
