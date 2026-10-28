"""Install built distributions or a registry requirement in fresh consumer venvs."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import venv
from pathlib import Path

SMOKE = """
import importlib.metadata
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import opentrace_ml as m

assert sys.prefix != sys.base_prefix, "Consumer must run in a virtual environment"
assert Path(m.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), m.__file__
assert importlib.metadata.version("opentrace-ml") == sys.argv[1]
assert m.__version__ == sys.argv[1]
assert all(hasattr(m, name) for name in m.__all__)
frame = pd.DataFrame({
    "date_time": pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC"),
    "traffic_volume": np.tile(np.arange(24, dtype=float) + 100, 2),
})
model = m.SeasonalNaiveForecaster().fit_frame(frame)
forecast = model.forecast(frame["date_time"].iloc[-1] + pd.Timedelta(hours=1), periods=24)
np.testing.assert_array_equal(forecast["predicted_traffic_volume"], np.arange(24) + 100)
assert m.optimal_model_path(np.array([[1, 9], [2, 8]])).indices == (0, 0)
assert m.score_route(m.RouteSignals("consumer", 100, 100))["reliability_score"] == 100
print(f"Verified opentrace-ml {m.__version__} from {m.__file__}")
"""


def verify(requirement: str, version: str, *, index_url: str | None) -> None:
    with tempfile.TemporaryDirectory(prefix="opentrace-consumer-") as directory:
        root = Path(directory)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        command = [str(python), "-I", "-m", "pip", "--isolated", "install", "--no-cache-dir"]
        if index_url:
            command.extend(["--index-url", index_url])
        # Dependencies must be resolved normally; never use --no-deps here.
        subprocess.run([*command, requirement], cwd=root, check=True)
        subprocess.run([str(python), "-I", "-m", "pip", "check"], cwd=root, check=True)
        subprocess.run([str(python), "-I", "-c", SMOKE, version], cwd=root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dist-dir", type=Path)
    source.add_argument("--requirement", help="Exact published package requirement")
    parser.add_argument("--version", required=True)
    parser.add_argument("--index-url")
    args = parser.parse_args()
    if args.dist_dir:
        wheels = list(args.dist_dir.glob("*.whl"))
        sources = list(args.dist_dir.glob("*.tar.gz"))
        if len(wheels) != 1 or len(sources) != 1:
            parser.error("dist-dir must contain exactly one wheel and one source archive")
        requirements = [str(path.resolve()) for path in wheels + sources]
    else:
        requirements = [args.requirement]
    for requirement in requirements:
        print(f"Checking {requirement} in an independent environment", flush=True)
        verify(requirement, args.version, index_url=args.index_url)


if __name__ == "__main__":
    main()
