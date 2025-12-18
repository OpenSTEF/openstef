"""Liander 2024 Benchmark Example.

====================================

This example demonstrates how to set up and run the Liander 2024 STEF benchmark using OpenSTEF BEAM.
The benchmark will evaluate XGBoost and GBLinear models on the dataset from HuggingFace.
"""

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import os

os.environ["OMP_NUM_THREADS"] = "1"  # Set OMP_NUM_THREADS to 1 to avoid issues with parallel execution and xgboost
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging
import multiprocessing
from pathlib import Path

from openstef_beam.benchmarking.baselines import (
    create_openstef4_preset_backtest_forecaster,
)
from openstef_beam.benchmarking.benchmarks.liander2024 import Liander2024Category, create_liander2024_benchmark_runner
from openstef_beam.benchmarking.callbacks.strict_execution_callback import StrictExecutionCallback
from openstef_beam.benchmarking.storage.local_storage import LocalBenchmarkStorage
from openstef_core.types import LeadTime, Q
from openstef_models.integrations.mlflow.mlflow_storage import MLFlowStorage
from openstef_models.presets import (
    ForecastingWorkflowConfig,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

OUTPUT_PATH = Path("./benchmark_results_test_convenience")

BENCHMARK_RESULTS_PATH_XGBOOST = OUTPUT_PATH / "XGBoost"
BENCHMARK_RESULTS_PATH_GBLINEAR = OUTPUT_PATH / "GBLinear"
N_PROCESSES = multiprocessing.cpu_count()  # Amount of parallel processes to use for the benchmark


# Model configuration
FORECAST_HORIZONS = [LeadTime.from_string("P3D")]  # Forecast horizon(s)
PREDICTION_QUANTILES = [
    Q(0.05),
    Q(0.1),
    Q(0.3),
    Q(0.5),
    Q(0.7),
    Q(0.9),
    Q(0.95),
]  # Quantiles for probabilistic forecasts

BENCHMARK_FILTER: list[Liander2024Category] | None = None

USE_MLFLOW_STORAGE = False

if USE_MLFLOW_STORAGE:
    storage = MLFlowStorage(
        tracking_uri=str(OUTPUT_PATH / "mlflow_artifacts"),
        local_artifacts_path=OUTPUT_PATH / "mlflow_tracking_artifacts",
    )
else:
    storage = None

common_config = ForecastingWorkflowConfig(
    model_id="common_model_",
    run_name=None,
    model="flatliner",
    horizons=FORECAST_HORIZONS,
    quantiles=PREDICTION_QUANTILES,
    model_reuse_enable=True,
    mlflow_storage=storage,
    radiation_column="shortwave_radiation",
    rolling_aggregate_features=["mean", "median", "max", "min"],
    wind_speed_column="wind_speed_80m",
    pressure_column="surface_pressure",
    temperature_column="temperature_2m",
    relative_humidity_column="relative_humidity_2m",
    energy_price_column="EPEX_NL",
)

xgboost_config = common_config.model_copy(update={"model": "xgboost"})

gblinear_config = common_config.model_copy(update={"model": "gblinear"})

if __name__ == "__main__":
    # Run for XGBoost model
    create_liander2024_benchmark_runner(
        storage=LocalBenchmarkStorage(base_path=BENCHMARK_RESULTS_PATH_XGBOOST),
        callbacks=[StrictExecutionCallback()],
    ).run(
        forecaster_factory=create_openstef4_preset_backtest_forecaster(
            workflow_config=xgboost_config,
            cache_dir=OUTPUT_PATH / "cache",
        ),
        run_name="xgboost",
        n_processes=N_PROCESSES,
        filter_args=BENCHMARK_FILTER,
    )

    # # Run for GBLinear model
    create_liander2024_benchmark_runner(
        storage=LocalBenchmarkStorage(base_path=BENCHMARK_RESULTS_PATH_GBLINEAR),
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
