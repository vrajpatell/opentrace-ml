# Architecture

OpenTrace ML uses shared event contracts to keep vision, forecasting, and
routing components independent.

## Data flow

1. A detector emits a label, confidence, bounding box, frame identifier, and
   video-relative timestamp.
2. The private trace boundary verifies consent, pseudonymizes the trip ID, and
   removes duplicate or implausible samples.
3. A map-matcher adapter emits one ordered matched or unmatched observation for
   every prepared trace point.
4. GPS interpolation attaches the detection to a latitude and longitude.
5. The traffic forecaster learns from each new timestamped volume observation.
6. Route scoring combines hazard counts, predicted traffic, unmatched map
   coverage, and distance into an auditable reliability score.
7. Routes and detections are exported as GeoJSON for MapLibre, GIS software, or
   a downstream API.

## Boundaries

- The core package does not download or bundle datasets automatically.
- Model frameworks are adapters, not required core dependencies.
- Map rendering belongs in downstream applications.
- OSM routing engines such as Valhalla remain external services.
- Consent and privacy checks must happen before private GPS traces enter a
  downstream pipeline.
- Raw device IDs must never enter OpenTrace. Deployments provide a trip-scoped
  identifier and retain the HMAC secret outside code and logs.
- `PreparedTrace.points` remains sensitive. Community layers must expose only
  thresholded, aggregated centerlines after human review, never raw traces.
- RDD2022-derived detections remain application-layer output and are not an
  authorized source for OpenStreetMap edits.

## Stable contracts

The initial contracts are `BoundingBox`, `Detection`, `GeoPoint`,
`GeoDetection`, `PreparedTrace`, and `MapMatchResult`. Backends may change as
long as they produce these objects or equivalent serialized records.

## Next integration boundary

OSRM, Valhalla, FMM, or other adapters implement the model-independent
`MapMatcher` protocol. The next integration milestone is a legally documented,
offline OSM fixture so adapter tests never contact public OSM services.
