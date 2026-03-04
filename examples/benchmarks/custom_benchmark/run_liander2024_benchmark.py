"""Example: run the built-in Liander 2024 benchmark with a custom baseline and GBLinear.

Uses create_liander2024_benchmark_runner() which pre-configures everything:
backtest settings, evaluation windows, metrics, analysis plots, and target
definitions. Data is auto-downloaded from HuggingFace.
"""

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging
import multiprocessing
from pathlib import Path

from examples.benchmarks.custom_benchmark.example_baseline import ExampleBenchmarkForecaster
from openstef_beam.benchmarking import BenchmarkContext, BenchmarkTarget, LocalBenchmarkStorage
from openstef_beam.benchmarking.baselines import create_openstef4_preset_backtest_forecaster
from openstef_beam.benchmarking.benchmarks.liander2024 import Liander2024Category, create_liander2024_benchmark_runner
from openstef_beam.benchmarking.callbacks.strict_execution_callback import StrictExecutionCallback
from openstef_core.types import LeadTime, Q
from openstef_models.presets import ForecastingWorkflowConfig

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

OUTPUT_PATH = Path("./benchmark_results")
N_PROCESSES = multiprocessing.cpu_count()

# Optional: filter to specific target categories (None = run all)
BENCHMARK_FILTER: list[Liander2024Category] | None = None

# Quantiles define the probabilistic forecast bands
# Q(0.05) = 5th percentile, Q(0.5) = median, Q(0.95) = 95th percentile
PREDICTION_QUANTILES = [Q(0.05), Q(0.1), Q(0.3), Q(0.5), Q(0.7), Q(0.9), Q(0.95)]

# --- GBLinear model config ---
# Map column names in your data to what OpenSTEF expects
gblinear_config = ForecastingWorkflowConfig(
    model_id="liander_benchmark_",
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


def example_factory(_context: BenchmarkContext, _target: BenchmarkTarget) -> ExampleBenchmarkForecaster:
    """Create the example baseline forecaster.

    Returns:
        Configured ExampleBenchmarkForecaster instance.
    """
    return ExampleBenchmarkForecaster(predict_quantiles=PREDICTION_QUANTILES)


if __name__ == "__main__":
    # 1. Run custom baseline on Liander 2024
    # create_liander2024_benchmark_runner() sets up everything: data download, configs, metrics
    # LocalBenchmarkStorage writes results as parquet files to disk
    create_liander2024_benchmark_runner(
        storage=LocalBenchmarkStorage(base_path=OUTPUT_PATH / "ExampleBaseline"),
        callbacks=[StrictExecutionCallback()],  # Fail fast on errors
    ).run(
        forecaster_factory=example_factory,  # Your model factory (called per target)
        run_name="example_baseline",         # Label for this run
        n_processes=N_PROCESSES,              # Parallel targets
        filter_args=BENCHMARK_FILTER,        # None = all categories
    )

    # 2. Run GBLinear on Liander 2024
    # create_openstef4_preset_backtest_forecaster returns a factory that wraps OpenSTEF models
    create_liander2024_benchmark_runner(
        storage=LocalBenchmarkStorage(base_path=OUTPUT_PATH / "GBLinear"),
        callbacks=[StrictExecutionCallback()],
    ).run(
        forecaster_factory=create_openstef4_preset_backtest_forecaster(
            workflow_config=gblinear_config,
            cache_dir=OUTPUT_PATH / "cache",
        ),
        run_name="gblinear",
        n_processes=N_PROCESSES,
        filter_args=BENCHMARK_FILTER,
    )
