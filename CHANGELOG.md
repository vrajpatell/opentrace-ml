# Changelog

## 0.2.0a1 — 2026-10-21

- Reject invalid route distances/IDs/counts, non-finite detection timestamps,
  invalid geographic positions, and unrepresentable evaluation results.
- Repair online learning with squared loss, normalized per-observation updates,
  and coefficient rebasing when the training-only feature scaler changes.
  Existing linear inference snapshots remain compatible; retraining produces
  intentionally different parameters. New `step_size` defaults to 0.2.
- Add DP horizon selection and parallel independent candidate calibration/refit.
- Add immutable data-only neural and built-in ensemble exports and native Go
  inference. v1 requires explicit fixed UTC offsets and integer-second cadence.
- Add opt-in gap segmentation (`TraceCleaningConfig.max_gap_seconds`). When
  configured, `prepare_trace` returns `PreparedTraceSegments`, not a flattened
  trace; iterate its segments. Default behavior remains unchanged.
- Add distinct-trip aggregation with minimum three contributors, quantized cells,
  and explicit human-review gating before export. This is not formal anonymity.
- Add optional MMDetection RTMDet integration and an original offline OSM fixture.
- Add multi-season aggregate evaluation reports, neural/process benchmarks,
  shared invalid-input/model fixtures, loader fuzzing, and package build checks.

Compatibility tightening: invalid inputs previously accepted now fail. GeoJSON
route helpers require at least two positions. Batch neural/ensemble models remain
experimental point forecasters; training warnings are reported, not suppressed.
No datasets, third-party weights, service deployment, or package-registry upload
is included in this release preparation.
