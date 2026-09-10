"""Export a fitted traffic model for native Go or portable Python inference."""

import argparse

import pandas as pd

from opentrace_ml import OnlineTrafficForecaster


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", help="User-supplied traffic CSV (for example licensed UCI data)")
    source.add_argument("--synthetic-demo", action="store_true", help="Use original synthetic data")
    parser.add_argument("--output", required=True, help="Destination JSON file; replaces existing file")
    parser.add_argument("--lags", type=int, default=6)
    parser.add_argument("--timestamp-column", default="date_time")
    parser.add_argument("--target-column", default="traffic_volume")
    args = parser.parse_args()
    if args.synthetic_demo:
        timestamps = pd.date_range("2026-08-31T00:00:00Z", periods=96, freq="h")
        frame = pd.DataFrame({
            args.timestamp_column: timestamps,
            args.target_column: [100 + timestamp.hour * 5 for timestamp in timestamps],
        })
    else:
        frame = pd.read_csv(args.csv)
    model = OnlineTrafficForecaster(lags=args.lags).fit_frame(
        frame, timestamp_column=args.timestamp_column, target_column=args.target_column,
    )
    model.export_model().save(args.output)
    print(f"Exported inference snapshot with {args.lags} lags to {args.output}")


if __name__ == "__main__":
    main()
