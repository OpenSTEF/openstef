"""Liander 2024 Benchmark Example.

====================================

This example demonstrates how to set up and run the Liander 2024 STEF benchmark using OpenSTEF BEAM.
The benchmark will evaluate XGBoost and GBLinear models on the dataset from HuggingFace.
"""

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import os
import time

os.environ["OMP_NUM_THREADS"] = "1"  # Set OMP_NUM_THREADS to 1 to avoid issues with parallel execution and xgboost
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging
import multiprocessing
from datetime import timedelta
from pathlib import Path

from pydantic_extra_types.coordinate import Coordinate
from pydantic_extra_types.country import CountryAlpha2

from openstef_beam.backtesting.backtest_forecaster import BacktestForecasterConfig, OpenSTEF4BacktestForecaster
from openstef_beam.benchmarking.benchmark_pipeline import BenchmarkContext
from openstef_beam.benchmarking.benchmarks.liander2024 import Liander2024Category, create_liander2024_benchmark_runner
from openstef_beam.benchmarking.callbacks.strict_execution_callback import StrictExecutionCallback
from openstef_beam.benchmarking.models.benchmark_target import BenchmarkTarget
from openstef_beam.benchmarking.storage.local_storage import LocalBenchmarkStorage
from openstef_core.types import LeadTime, Q
from openstef_meta.presets import (
    EnsembleWorkflowConfig,
    create_ensemble_workflow,
)
from openstef_models.integrations.mlflow.mlflow_storage import MLFlowStorage
from openstef_models.presets.forecasting_workflow import LocationConfig
from openstef_models.workflows import CustomForecastingWorkflow

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

OUTPUT_PATH = Path("./benchmark_results")

N_PROCESSES = 1 if True else multiprocessing.cpu_count()  # Amount of parallel processes to use for the benchmark

ensemble_type = "stacking"  # "stacking", "learned_weights" or "rules"
base_models = ["lgbm", "gblinear"]  # combination of "lgbm", "gblinear", "xgboost" and "lgbm_linear"
combiner_model = (
    "gblinear"  # "lgbm", "xgboost", "rf" or "logistic" for learned weights combiner, gblinear for stacking combiner
)

model = "Ensemble_" + "_".join(base_models) + "_" + ensemble_type + "_" + combiner_model

# Model configuration
FORECAST_HORIZONS = [LeadTime.from_string("PT36H")]  # Forecast horizon(s)
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

common_config = EnsembleWorkflowConfig(
    model_id="common_model_",
    ensemble_type=ensemble_type,
    base_models=base_models,  # type: ignore
    combiner_model=combiner_model,
    horizons=FORECAST_HORIZONS,
    quantiles=PREDICTION_QUANTILES,
    model_reuse_enable=False,
    mlflow_storage=None,
    radiation_column="shortwave_radiation",
    rolling_aggregate_features=["mean", "median", "max", "min"],
    wind_speed_column="wind_speed_80m",
    pressure_column="surface_pressure",
    temperature_column="temperature_2m",
    relative_humidity_column="relative_humidity_2m",
    energy_price_column="EPEX_NL",
    forecast_combiner_sample_weight_exponent=0,
    forecaster_sample_weight_exponent={"gblinear": 1, "lgbm": 0, "xgboost": 0, "lgbm_linear": 0},
)


# Create the backtest configuration
backtest_config = BacktestForecasterConfig(
    requires_training=True,
    predict_length=timedelta(days=7),
    predict_min_length=timedelta(minutes=15),
    predict_context_length=timedelta(days=14),  # Context needed for lag features
    predict_context_min_coverage=0.5,
    training_context_length=timedelta(days=90),  # Three months of training data
    training_context_min_coverage=0.5,
    predict_sample_interval=timedelta(minutes=15),
)


def _target_forecaster_factory(
    context: BenchmarkContext,
    target: BenchmarkTarget,
) -> OpenSTEF4BacktestForecaster:
    # Factory function that creates a forecaster for a given target.
    prefix = context.run_name
    base_config = common_config

    def _create_workflow() -> CustomForecastingWorkflow:
        # Create a new workflow instance with fresh model.
        return create_ensemble_workflow(
            config=base_config.model_copy(
                update={
                    "model_id": f"{prefix}_{target.name}",
                    "location": LocationConfig(
                        name=target.name,
                        description=target.description,
                        coordinate=Coordinate(
                            latitude=target.latitude,
                            longitude=target.longitude,
                        ),
                        country_code=CountryAlpha2("NL"),
                    ),
                }
            )
        )

    return OpenSTEF4BacktestForecaster(
        config=backtest_config,
        workflow_factory=_create_workflow,
        debug=False,
        contributions=False,
        cache_dir=OUTPUT_PATH / "cache" / f"{context.run_name}_{target.name}",
    )


if __name__ == "__main__":
    start_time = time.time()
    create_liander2024_benchmark_runner(
        storage=LocalBenchmarkStorage(base_path=OUTPUT_PATH / model),
        data_dir=Path("../data/liander2024-energy-forecasting-benchmark"),
        callbacks=[StrictExecutionCallback()],
    ).run(
        forecaster_factory=_target_forecaster_factory,
        run_name=model,
        n_processes=N_PROCESSES,
        filter_args=BENCHMARK_FILTER,
    )

    end_time = time.time()
    print(f"Benchmark completed in {end_time - start_time:.2f} seconds.")
