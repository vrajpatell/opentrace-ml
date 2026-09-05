# Stage two: reproducible baselines

Stage two turns the initial data contracts into measurable ML components.

## Delivered foundation

- `Detector` protocol for model-independent CV integrations.
- `CallableDetector` reference adapter.
- bounding-box IoU and single-threshold detection metrics.
- per-class, frame- and IoU-aware road-damage evaluation reports.
- MAE, RMSE, and zero-safe MAPE metrics.
- leakage-resistant rolling traffic backtesting.
- runnable road-damage GeoJSON and traffic-backtest examples.

## Next implementation slices

1. Add an MMDetection/RTMDet adapter without making it a core dependency.
2. Add train/evaluate commands for a small documented RDD2022 subset.
3. Add mean average precision over documented IoU thresholds.
4. Cache a tiny OSM extract for offline integration testing.
5. [x] Add GPX parsing and timestamp normalization.

Trace ingestion now continues in [stage three](STAGE_3.md), where consent,
pseudonymization, and deterministic cleaning form the privacy boundary before
map matching.

Each slice should have a small public-data fixture, deterministic tests, CLI
documentation, and licence notes before it is merged.
