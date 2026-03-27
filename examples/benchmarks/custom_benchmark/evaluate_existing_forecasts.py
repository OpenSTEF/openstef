"""Evaluate pre-existing forecasts without running backtesting.

If you already have forecast predictions (e.g. from your own model or an external
system), you can point the benchmark pipeline at them and run only the evaluation
and analysis steps.

How it works:
  1. Place your prediction parquet files in the expected directory layout (see below).
  2. Run this script — the pipeline detects existing backtest output and
     automatically skips to evaluation + analysis.

Expected directory layout::

    benchmark_results/MyForecasts/
    └── backtest/
        └── <group_name>/           # e.g. "solar_park"
            └── <target_name>/      # e.g. "Within 15 kilometers of Opmeer_normalized"
                └── predictions.parquet

Expected parquet format::

    Index:   pd.DatetimeIndex (name="timestamp", tz-naive UTC, 15-min intervals)
    Columns:
      - "available_at" (datetime)  — when the prediction was generated
      - "quantile_P05" (float)     — 5th percentile prediction
      - "quantile_P50" (float)     — median prediction (REQUIRED)
      - "quantile_P95" (float)     — 95th percentile prediction
      - ...one column per quantile, named with Quantile(x).format()

Example row::

    timestamp (index)      available_at          quantile_P05  quantile_P50  quantile_P95
    2023-01-15 12:00:00    2023-01-14 06:00:00   0.5           1.2           2.0

You can list the expected target names and group names by checking the targets.yaml
in your dataset, or by running::

    runner = create_custom_benchmark_runner()
    for t in runner.target_provider.get_targets(["solar_park"]):
        print(t.group_name, t.name)

The pipeline still needs a "forecaster factory" to know which quantiles were used,
but fit() and predict() are never called. We use DummyForecaster for this.
"""

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import logging
import multiprocessing
import os
from pathlib import Path

from examples.benchmarks.custom_benchmark.example_benchmark import create_custom_benchmark_runner
from openstef_beam.backtesting.backtest_forecaster import DummyForecaster
from openstef_beam.benchmarking import BenchmarkContext, BenchmarkTarget, LocalBenchmarkStorage
from openstef_core.types import Q

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

_logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

# Path to the folder that contains the backtest/ directory with your parquets.
OUTPUT_PATH = Path("./benchmark_results/MyForecasts")
N_PROCESSES = multiprocessing.cpu_count()

# Quantiles your forecasts were generated for (must include 0.5 = median).
# Adjust this list to match whatever quantiles are in your parquet columns.
PREDICTION_QUANTILES = [Q(0.05), Q(0.1), Q(0.3), Q(0.5), Q(0.7), Q(0.9), Q(0.95)]


def stub_factory(_context: BenchmarkContext, _target: BenchmarkTarget) -> DummyForecaster:
    """Factory that returns a DummyForecaster (backtesting is skipped).

    DummyForecaster provides quantile info to the pipeline but never runs
    fit() or predict() since backtest output already exists on disk.

    Returns:
        DummyForecaster with the configured quantiles.
    """
    return DummyForecaster(predict_quantiles=PREDICTION_QUANTILES)


if __name__ == "__main__":
    # Point the storage at your results folder.
    # The pipeline reads parquets from:
    #   OUTPUT_PATH / backtest / <group_name> / <target_name> / predictions.parquet
    storage = LocalBenchmarkStorage(base_path=OUTPUT_PATH)

    runner = create_custom_benchmark_runner(storage=storage)

    # Run the pipeline — backtesting is auto-skipped for every target that
    # already has a predictions.parquet on disk.
    runner.run(
        forecaster_factory=stub_factory,
        run_name="my_forecasts",
        n_processes=N_PROCESSES,
        filter_args=["solar_park"],
    )
