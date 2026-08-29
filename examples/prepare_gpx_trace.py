"""Prepare a private GPX trace without printing its coordinates."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

from opentrace_ml import load_gpx_points, prepare_trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gpx_path")
    parser.add_argument("--trip-id", required=True, help="Trip-scoped ID; never use a device ID")
    parser.add_argument("--consent", action="store_true", help="Confirm the trace owner consented")
    args = parser.parse_args()

    secret_key = os.environ.get("OPENTRACE_PSEUDONYM_KEY")
    if not secret_key:
        parser.error("Set OPENTRACE_PSEUDONYM_KEY to a secret containing at least 16 bytes")

    prepared = prepare_trace(
        load_gpx_points(args.gpx_path),
        trip_id=args.trip_id,
        secret_key=secret_key,
        consent_granted=args.consent,
    )
    print(json.dumps({"trip_id": prepared.trip_id, "report": asdict(prepared.report)}, indent=2))


if __name__ == "__main__":
    main()
