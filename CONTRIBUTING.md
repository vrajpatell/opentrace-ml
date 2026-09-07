# Contributing to OpenTrace ML

Thank you for helping build open road-intelligence tooling.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m pytest
```

Run formatting and static checks before opening a pull request:

```bash
ruff check .
```

For Go changes, use Go 1.26 or newer and run:

```bash
cd go
gofmt -w .
go vet ./...
go test -race ./...
go test -run '^$' -bench . -benchmem ./...
```

Changes to shared behavior must update `spec/v1/` when the portable contract
changes and add a case to `go/testdata/conformance/v1/` that passes in both languages.

## Pull requests

- Keep changes focused and include tests for new behavior.
- Preserve the separation between vision, forecasting, and mapping backends.
- Document public APIs and externally visible behavior.
- Do not commit datasets, private GPS traces, access tokens, or model weights.
- Never log raw trace coordinates, raw trip IDs, device IDs, or pseudonym keys.
- Preserve the explicit consent gate for every private trace-ingestion path.
- Publish only thresholded aggregate geometry after documented human review.
- Update `DATA_LICENSES.md` when adding a public-data adapter.
- Preserve the documented complexity bound for every Go hot-path operation;
  include benchmark comparisons for performance-sensitive changes.
- Keep the Go core standard-library-only. Propose optional dependencies in a
  separate adapter package with measurements and licence analysis.

## Dataset contributions

Every proposed dataset integration must document:

- authoritative source and version;
- licence and required attribution;
- expected download size and checksum when available;
- personally identifiable or sensitive data considerations;
- whether derived outputs have share-alike requirements.

## Licensing

By submitting a contribution to OpenTrace ML, you agree that your contribution
will be licensed under the Apache License, Version 2.0.
