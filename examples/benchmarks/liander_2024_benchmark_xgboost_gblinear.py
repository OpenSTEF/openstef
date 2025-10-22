"""Liander 2024 Benchmark Example.

====================================

This example demonstrates how to set up and run the Liander 2024 STEF benchmark using OpenSTEF BEAM.
The benchmark will evaluate XGBoost and GBLinear models on the dataset from HuggingFace.
"""

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import logging
from datetime import timedelta
from pathlib import Path

from pydantic_extra_types.country import CountryAlpha2

from openstef_beam.backtesting.backtest_forecaster import BacktestForecasterConfig, OpenSTEF4Forecaster
from openstef_beam.benchmarking.benchmark_pipeline import BenchmarkContext
from openstef_beam.benchmarking.benchmarks.liander2024 import create_liander2024_benchmark_runner
from openstef_beam.benchmarking.callbacks.strict_execution_callback import StrictExecutionCallback
from openstef_beam.benchmarking.models.benchmark_target import BenchmarkTarget
from openstef_beam.benchmarking.storage.local_storage import LocalBenchmarkStorage
from openstef_core.types import LeadTime, Q
from openstef_models.integrations.mlflow.mlflow_storage import MLFlowStorage
from openstef_models.integrations.mlflow.mlflow_storage_callback import MLFlowStorageCallback
from openstef_models.models.forecasting.gblinear_forecaster import (
    GBLinearForecaster,
    GBLinearForecasterConfig,
    GBLinearHyperParams,
)
from openstef_models.models.forecasting.xgboost_forecaster import (
    XGBoostForecaster,
    XGBoostForecasterConfig,
    XGBoostHyperParams,
)
from openstef_models.models.forecasting_model import ForecastingModel, PostprocessingPipeline
from openstef_models.transforms.feature_engineering_pipeline import FeatureEngineeringPipeline
from openstef_models.transforms.general import Scaler
from openstef_models.transforms.time_domain.holiday_features_adder import HolidayFeatureAdder
from openstef_models.transforms.time_domain.versioned_lags_adder import VersionedLagsAdder
from openstef_models.workflows.custom_forecasting_workflow import CustomForecastingWorkflow

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

BENCHMARK_RESULTS_PATH_XGBOOST = Path("./benchmark_results/XGBoost")
BENCHMARK_RESULTS_PATH_GBLINEAR = Path("./benchmark_results/GBLinear")
N_PROCESSES = 1  # Amount of parallel processes to use for the benchmark

# Model configuration
FORECAST_HORIZONS = [LeadTime.from_string("PT12H")]  # Forecast horizon(s)
PREDICTION_QUANTILES = [Q(0.1), Q(0.3), Q(0.5), Q(0.7), Q(0.9)]  # Quantiles for probabilistic forecasts
LAG_FEATURES = [timedelta(days=-7)]  # Lag features to include
XGBOOST_HYPERPARAMS = XGBoostHyperParams()  # XGBoost hyperparameters
XGBOOST_VERBOSITY = 1
GBLINEAR_HYPERPARAMS = GBLinearHyperParams()  # GBLinear hyperparameters
GBLINEAR_VERBOSITY = 1


def xgboost_forecaster_factory(
    context: BenchmarkContext,  # noqa: ARG001
    target: BenchmarkTarget,
) -> OpenSTEF4Forecaster:
    """Factory function that creates a WorkflowBacktestAdapter with XGBoost model.

    Args:
        context: Benchmark context with run metadata
        target: Target specification for the benchmark

    Returns:
        OpenSTEF4Forecaster: Forecaster wrapping the configured forecasting workflow
    """

    def create_workflow() -> CustomForecastingWorkflow:
        """Create a new workflow instance with fresh model.

        Returns:
            A configured forecasting workflow with XGBoost model
        """
        # Create the forecasting model with preprocessing, forecaster, and postprocessing
        model = ForecastingModel(
            preprocessing=FeatureEngineeringPipeline.create(
                horizons=FORECAST_HORIZONS,
                horizon_transforms=[
                    Scaler(method="standard"),
                    HolidayFeatureAdder(country_code=CountryAlpha2("NL")),
                ],
                versioned_transforms=[VersionedLagsAdder(feature="load", lags=LAG_FEATURES)],
            ),
            forecaster=XGBoostForecaster(
                config=XGBoostForecasterConfig(
                    horizons=FORECAST_HORIZONS,
                    quantiles=PREDICTION_QUANTILES,
                    hyperparams=XGBOOST_HYPERPARAMS,
                    verbosity=XGBOOST_VERBOSITY,
                )
            ),
            postprocessing=PostprocessingPipeline(transforms=[]),
            target_column="load",
        )

        # Create the forecasting workflow with the model
        return CustomForecastingWorkflow(
            model=model,
            model_id=f"xgboost_{target.name}",
            callbacks=[
                MLFlowStorageCallback(
                    storage=MLFlowStorage(
                        tracking_uri=str(BENCHMARK_RESULTS_PATH_XGBOOST / "mlflow_tracking"),
                        local_artifacts_path=BENCHMARK_RESULTS_PATH_XGBOOST / "mlflow_tracking_artifacts",
                    ),
                    model_reuse_enable=False,
                )
            ],
        )

    # Create the backtest configuration
    backtest_config = BacktestForecasterConfig(
        requires_training=True,
        horizon_length=timedelta(days=7),
        horizon_min_length=timedelta(minutes=15),
        predict_context_length=timedelta(days=14),  # Context needed for lag features
        predict_context_min_coverage=0.5,
        training_context_length=timedelta(days=90),  # Three months of training data
        training_context_min_coverage=0.5,
        predict_sample_interval=timedelta(minutes=15),
    )

    return OpenSTEF4Forecaster(config=backtest_config, workflow_factory=create_workflow)


def gblinear_forecaster_factory(
    context: BenchmarkContext,  # noqa: ARG001
    target: BenchmarkTarget,
) -> OpenSTEF4Forecaster:
    """Factory function that creates a WorkflowBacktestAdapter with GBLinear model.

    Args:
        context: Benchmark context with run metadata
        target: Target specification for the benchmark

    Returns:
        OpenSTEF4Forecaster: Forecaster wrapping the configured forecasting workflow
    """

    def create_workflow() -> CustomForecastingWorkflow:
        """Create a new workflow instance with fresh model.

        Returns:
            A configured forecasting workflow with GBLinear model
        """
        # Create the forecasting model with preprocessing, forecaster, and postprocessing
        model = ForecastingModel(
            preprocessing=FeatureEngineeringPipeline.create(
                horizons=FORECAST_HORIZONS,
                horizon_transforms=[
                    Scaler(method="standard"),
                    HolidayFeatureAdder(country_code=CountryAlpha2("NL")),
                ],
                versioned_transforms=[VersionedLagsAdder(feature="load", lags=LAG_FEATURES)],
            ),
            forecaster=GBLinearForecaster(
                config=GBLinearForecasterConfig(
                    horizons=FORECAST_HORIZONS,
                    quantiles=PREDICTION_QUANTILES,
                    hyperparams=GBLINEAR_HYPERPARAMS,
                    verbosity=GBLINEAR_VERBOSITY,
                )
            ),
            postprocessing=PostprocessingPipeline(transforms=[]),
            target_column="load",
        )

        # Create the forecasting workflow with the model
        return CustomForecastingWorkflow(
            model=model,
            model_id=f"gblinear_{target.name}",
            callbacks=[
                MLFlowStorageCallback(
                    storage=MLFlowStorage(
                        tracking_uri=str(BENCHMARK_RESULTS_PATH_GBLINEAR / "mlflow_tracking"),
                        local_artifacts_path=BENCHMARK_RESULTS_PATH_GBLINEAR / "mlflow_tracking_artifacts",
                    ),
                    model_reuse_enable=False,
                )
            ],
        )

    # Create the backtest configuration
    backtest_config = BacktestForecasterConfig(
        requires_training=True,
        horizon_length=timedelta(days=7),
        horizon_min_length=timedelta(minutes=15),
        predict_context_length=timedelta(days=14),  # Context needed for lag features
        predict_context_min_coverage=0.5,
        training_context_length=timedelta(days=90),  # Three months of training data
        training_context_min_coverage=0.5,
        predict_sample_interval=timedelta(minutes=15),
    )

    return OpenSTEF4Forecaster(config=backtest_config, workflow_factory=create_workflow)


if __name__ == "__main__":
    # Run for XGBoost model
    create_liander2024_benchmark_runner(
        storage=LocalBenchmarkStorage(base_path=BENCHMARK_RESULTS_PATH_XGBOOST),
        callbacks=[StrictExecutionCallback()],
    ).run(
        forecaster_factory=xgboost_forecaster_factory,
        run_name="OpenSTEF_XGBoost",
        n_processes=N_PROCESSES,
    )

    # Run for GBLinear model
    create_liander2024_benchmark_runner(
        storage=LocalBenchmarkStorage(base_path=BENCHMARK_RESULTS_PATH_GBLINEAR),
        callbacks=[StrictExecutionCallback()],
    ).run(
        forecaster_factory=gblinear_forecaster_factory,
        run_name="OpenSTEF_GBLinear",
        n_processes=N_PROCESSES,
    )
