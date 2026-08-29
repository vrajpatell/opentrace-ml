# Architecture

OpenTrace ML uses shared event contracts to keep vision, forecasting, and
routing components independent.

## Data flow

1. A detector emits a label, confidence, bounding box, frame identifier, and
   video-relative timestamp.
2. GPS interpolation attaches the detection to a latitude and longitude.
3. The traffic forecaster learns from each new timestamped volume observation.
4. Route scoring combines hazard counts, predicted traffic, unmatched map
   coverage, and distance into an auditable reliability score.
5. Routes and detections are exported as GeoJSON for MapLibre, GIS software, or
   a downstream API.

## Boundaries

- The core package does not download or bundle datasets automatically.
- Model frameworks are adapters, not required core dependencies.
- Map rendering belongs in downstream applications.
- OSM routing engines such as Valhalla remain external services.
- Consent and privacy checks must happen before private GPS traces enter a
  downstream pipeline.

## Stable contracts

The initial contracts are `BoundingBox`, `Detection`, `GeoPoint`, and
`GeoDetection`. Backends may change as long as they produce these objects or
equivalent serialized records.

## Next integration boundary

The next milestone will define a detector protocol with a reference RDD2022
training/evaluation command. The protocol will emit existing `Detection`
objects, so the geospatial modules will not depend on a particular CV library.

