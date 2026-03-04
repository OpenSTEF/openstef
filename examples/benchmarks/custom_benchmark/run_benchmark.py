"""Run the custom benchmark: example baseline vs OpenSTEF GBLinear.

Uses the custom benchmark pipeline from example_benchmark.py (which extends
SimpleTargetProvider) instead of the built-in Liander 2024 runner.
"""

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import os

# Prevent thread contention when running multiple targets in parallel
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging
import multiprocessing
from pathlib import Path

from examples.benchmarks.custom_benchmark.example_baseline import ExampleBenchmarkForecaster
from examples.benchmarks.custom_benchmark.example_benchmark import MyCategory, create_custom_benchmark_runner
from openstef_beam.benchmarking import BenchmarkContext, BenchmarkTarget, LocalBenchmarkStorage
from openstef_beam.benchmarking.baselines import create_openstef4_preset_backtest_forecaster
from openstef_core.types import LeadTime, Q
from openstef_models.presets import ForecastingWorkflowConfig

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

OUTPUT_PATH = Path("./benchmark_results")
N_PROCESSES = multiprocessing.cpu_count()

# Optional: filter to specific target categories (None = run all)
BENCHMARK_FILTER: list[MyCategory] | None = ["solar_park"]

# Quantiles define the probabilistic forecast bands
PREDICTION_QUANTILES = [Q(0.05), Q(0.1), Q(0.3), Q(0.5), Q(0.7), Q(0.9), Q(0.95)]

# --- GBLinear config ---
# Map column names in your data to what OpenSTEF expects
gblinear_config = ForecastingWorkflowConfig(
    model_id="custom_benchmark_",
    run_name=None,
    model="gblinear",
    horizons=[LeadTime.from_string("P3D")],
    quantiles=PREDICTION_QUANTILES,
    model_reuse_enable=True,
    radiation_column="shortwave_radiation",
    wind_speed_column="wind_speed_80m",
    pressure_column="surface_pressure",
    temperature_column="temperature_2m",
    relative_humidity_column="relative_humidity_2m",
    energy_price_column="EPEX_NL",
    rolling_aggregate_features=["mean", "median", "max", "min"],
)


# --- Example baseline factory ---
def example_factory(_context: BenchmarkContext, _target: BenchmarkTarget) -> ExampleBenchmarkForecaster:
    """Create an example forecaster for a benchmark target.

    Returns:
        Configured ExampleBenchmarkForecaster instance.
    """
    return ExampleBenchmarkForecaster(predict_quantiles=PREDICTION_QUANTILES)


if __name__ == "__main__":
    # 1. Run example baseline using the custom benchmark pipeline
    create_custom_benchmark_runner(
        storage=LocalBenchmarkStorage(base_path=OUTPUT_PATH / "ExampleBaseline"),
    ).run(
        forecaster_factory=example_factory,
        run_name="example_baseline",
        n_processes=N_PROCESSES,
        filter_args=BENCHMARK_FILTER,
    )

    # 2. Run GBLinear using the same custom pipeline
    create_custom_benchmark_runner(
        storage=LocalBenchmarkStorage(base_path=OUTPUT_PATH / "GBLinear"),
    ).run(
        forecaster_factory=create_openstef4_preset_backtest_forecaster(
            workflow_config=gblinear_config,
            cache_dir=OUTPUT_PATH / "cache",
        ),
        run_name="gblinear",
        n_processes=N_PROCESSES,
        filter_args=BENCHMARK_FILTER,
    )
