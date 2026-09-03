# Stage three: privacy-safe trace preparation

Stage three creates the boundary between raw GPS exports and downstream map
matching. It is intentionally local and deterministic: no network service or
third-party dataset is required.

## Delivered foundation

- explicit consent gate in `prepare_trace`;
- HMAC-SHA256 trip pseudonyms scoped by a deployment secret;
- chronological-order validation and timestamp normalization;
- removal of exact duplicate samples and configurable speed outliers;
- aggregate cleaning reports that contain no coordinates;
- a CLI example that never prints the cleaned trace.

## Privacy contract

- Use a trip-scoped export identifier, never a device serial, advertising ID,
  phone number, account ID, or other durable personal identifier.
- Keep `OPENTRACE_PSEUDONYM_KEY` outside the repository and application logs.
- Treat `PreparedTrace.points` as private data even after pseudonymization.
- Do not publish raw or individual traces. Future candidate roads must require a
  minimum number of independent trips and human review before export.
- Pseudonymization reduces direct identification risk; it is not anonymization.

## Cleaning behavior

`TraceCleaningConfig.max_speed_m_s` defaults to 70 metres per second. A point is
removed when its speed from the last retained point exceeds the configured
limit. Exact duplicates at the same timestamp are removed; different positions
at the same timestamp are rejected. At least two points must remain.

The algorithm preserves input order. It does not smooth coordinates, split long
recording gaps, map-match points, or infer missing roads.

The model-independent map-matching boundary is now defined in
[stage four](STAGE_4.md). Stage three remains responsible for preparing the
sensitive input before any adapter runs.

## Next implementation slices

1. Split traces around configurable recording gaps.
2. Integrate the offline OSM fixture from issue #3.
3. Cluster unmatched segments only after a minimum trip threshold.
4. Export reviewed aggregate centerlines without raw-trace provenance.
