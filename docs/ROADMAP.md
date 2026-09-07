# OpenTrace ML roadmap

## 0.2.2 — Native Go execution core

- [x] Add shared versioned JSON contracts and cross-language fixtures.
- [x] Add native trace, GPS, GPX, GeoJSON, route, map-match, and metric APIs.
- [x] Guarantee linear ordered-batch geolocation and logarithmic indexed lookup.
- [x] Add race, fuzz-seed, conformance, and allocation benchmarks to CI.
- [ ] Publish the Go module after API review and a tagged pre-release.
- [ ] Add benchmark regression reporting with `benchstat`.
- [ ] Define an open serialized format for Go traffic-forecast inference.

## 0.1 — Common contracts and public-data adapters

- Parse RDD2022 Pascal VOC road-damage annotations.
- Geolocate timestamped detections on GPS traces.
- Incrementally forecast traffic volume.
- Convert routes and detections to GeoJSON.
- Score routes from damage, traffic, map confidence, and distance signals.

## 0.2 — Reproducible baselines

- [x] Add a detector adapter protocol and callable reference adapter.
- [x] Add detection metrics and rolling traffic backtesting.
- [x] Add per-class detection metrics at a configurable IoU threshold.
- [x] Add runnable current-stage mini-projects.
- [ ] Add an Apache-2.0-compatible reference detection model.
- Add train/evaluate commands for an RDD2022 subset.
- Add small cached OSM extracts for integration tests without hitting public services.
- [x] Add GPX parsing and timestamp normalization.

## 0.2.1 — Privacy-safe trace preparation

- [x] Require explicit consent at the downstream trace-preparation boundary.
- [x] Replace raw trip identifiers with secret-keyed HMAC pseudonyms.
- [x] Remove duplicate samples and implausible speed jumps deterministically.
- [x] Emit aggregate cleaning counts without logging coordinates.
- [ ] Segment traces around long recording gaps.
- [x] Define key-rotation and retention guidance for deployments.

## 0.3 — Route intelligence

- [x] Define a model-independent map-matcher protocol and offline callable adapter.
- Connect forecasts to road-network edges.
- Map-match GPS traces with Valhalla.
- Cluster repeated unmatched traces as candidate missing roads.
- Require a minimum contributing-trip threshold and human review before
  exporting aggregated road-layer changes.

## 0.4 — Streaming and web integration

- [x] Publish detection, GPS, and route-signal contracts through a stable JSON schema.
- Publish forecast and route-event contracts through the shared schema.
- Add a FastAPI reference service.
- Add a MapLibre example application.
- Measure end-to-end latency and synchronization error.
