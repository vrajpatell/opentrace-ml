# Stage four: map-matching contracts

Stage four begins route intelligence with an offline, model-independent boundary.
It defines what OSRM, Valhalla, FMM, or another adapter must return without
adding any routing engine or network client to the core package.

## Delivered foundation

- a runtime-checkable `MapMatcher` protocol that accepts only `PreparedTrace`;
- `RawMapMatch` for adapter output;
- ordered `MapMatchObservation` and `MapMatchResult` contracts;
- explicit unmatched observations instead of silently dropped GPS points;
- contiguous `MapMatchSegment` summaries and unmatched-coverage metrics;
- a deterministic callable adapter, offline example, and validation tests.

## Contract rules

Every source point must produce exactly one observation in the same order. A
matched observation has a snapped point, positive confidence, and a non-empty
edge ID. An unmatched observation has none of those values. Snapped points must
preserve the source timestamp.

`MapMatchResult` accepts only OpenTrace HMAC trip pseudonyms. This prevents raw
trip identifiers from moving accidentally into the route-intelligence layer.
The observations still contain sensitive source and snapped coordinates and
must remain inside the private processing boundary.

## Adapter boundary

The core library intentionally contains no HTTP client and no routing-engine
dependency. An external adapter implements `MapMatcher` and converts its engine's
response into `RawMapMatch` objects. Unit tests and examples use
`CallableMapMatcher`, so they are deterministic and never contact public OSM
services.

## Data and OSM boundary

RDD2022 detections are application-layer outputs. OpenTrace does not treat them
as data that can be uploaded to OpenStreetMap. A future OSM contribution workflow
must use an independently authorized source, verify licence compatibility, and
require human review. Until those conditions are explicitly implemented,
candidate roads remain a separate OpenTrace layer.

## Next implementation slices

1. Integrate the legally documented offline OSM fixture from issue #3.
2. Add an optional Valhalla or OSRM adapter outside the core dependency set.
3. Split prepared traces around long gaps before matching (issue #8).
4. Cluster unmatched segments behind the contributor threshold in issue #10.
5. Add explicit human-review states before any candidate geometry is exported.
