# Architecture

OpenTrace ML uses shared event contracts to keep vision, forecasting, and
routing components independent.

## Data flow

1. A detector emits a label, confidence, bounding box, frame identifier, and
   video-relative timestamp.
2. The private trace boundary verifies consent, pseudonymizes the trip ID, and
   removes duplicate or implausible samples.
3. GPS interpolation attaches the detection to a latitude and longitude.
4. The traffic forecaster learns from each new timestamped volume observation.
5. Route scoring combines hazard counts, predicted traffic, unmatched map
   coverage, and distance into an auditable reliability score.
6. Routes and detections are exported as GeoJSON for MapLibre, GIS software, or
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

## Stable contracts

The initial contracts are `BoundingBox`, `Detection`, `GeoPoint`, and
`GeoDetection`. Backends may change as long as they produce these objects or
equivalent serialized records.

## Next integration boundary

The next route-intelligence milestone will define a model-independent map-matcher
protocol. OSRM, Valhalla, or other adapters will emit shared match results so
unmatched-segment analysis does not depend on one routing engine.
