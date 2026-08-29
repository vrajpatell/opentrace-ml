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

## Pull requests

- Keep changes focused and include tests for new behavior.
- Preserve the separation between vision, forecasting, and mapping backends.
- Document public APIs and externally visible behavior.
- Do not commit datasets, private GPS traces, access tokens, or model weights.
- Update `DATA_LICENSES.md` when adding a public-data adapter.

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

