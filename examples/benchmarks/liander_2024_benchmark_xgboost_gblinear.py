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

from openstef_beam.backtesting.backtest_forecaster import BacktestForecasterConfig, BacktestForecasterMixin
from openstef_beam.backtesting.restricted_horizon_timeseries import RestrictedHorizonVersionedTimeSeries
from openstef_beam.benchmarking.benchmark_pipeline import BenchmarkContext
from openstef_beam.benchmarking.benchmarks.liander2024 import create_liander2024_benchmark_runner
from openstef_beam.benchmarking.callbacks.strict_execution_callback import StrictExecutionCallback
from openstef_beam.benchmarking.models.benchmark_target import BenchmarkTarget
from openstef_beam.benchmarking.storage.local_storage import LocalBenchmarkStorage
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.types import LeadTime, Q
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

BENCHMARK_RESULTS_PATH = Path("./benchmark_results")
N_PROCESSES = 1  # Amount of parallel processes to use for the benchmark

# Model configuration
FORECAST_HORIZONS = [LeadTime.from_string("PT12H")]  # Forecast horizon(s)
PREDICTION_QUANTILES = [Q(0.1), Q(0.3), Q(0.5), Q(0.7), Q(0.9)]  # Quantiles for probabilistic forecasts
LAG_FEATURES = [timedelta(days=-7)]  # Lag features to include
XGBOOST_HYPERPARAMS = XGBoostHyperParams()  # XGBoost hyperparameters
XGBOOST_VERBOSITY = 1


class OpenSTEF4Forecaster(BacktestForecasterMixin):
    """Forecaster that allows using a ForecastingWorkflow to be used in backtesting, specifically for OpenSTEF4 models.

    This bridges the gap between openstef_models' ForecastingWorkflow interface
    and openstef_beam's BacktestForecasterMixin interface, allowing workflows to be used
    in benchmark pipelines.
    """

    def __init__(self, config: BacktestForecasterConfig, workflow: CustomForecastingWorkflow):
        """Initialize the forecaster.

        Args:
            config: Configuration for the backtest forecaster interface
            workflow: The CustomForecastingWorkflow to adapt
        """
        self.config = config
        self.workflow = workflow

    @property
    def quantiles(self) -> list[Q]:
        """Return the list of quantiles that this forecaster predicts."""
        # Extract quantiles from the workflow's model
        return self.workflow.model.forecaster.config.quantiles

    def fit(self, data: RestrictedHorizonVersionedTimeSeries) -> None:
        """Train the model using data from the restricted horizon time series.

        Args:
            data: Time series data with horizon restrictions for training.
        """
        # Get training data window based on config
        training_end = data.horizon
        training_start = training_end - self.config.training_context_length

        # Extract the versioned dataset for training
        training_data = data.get_window_versioned(start=training_start, end=training_end, available_before=data.horizon)

        # Use the workflow's fit method
        self.workflow.fit(data=training_data)

    def predict(self, data: RestrictedHorizonVersionedTimeSeries) -> TimeSeriesDataset | None:
        """Generate predictions using the workflow.

        Args:
            data: Time series data with horizon restrictions for prediction.

        Returns:
            TimeSeriesDataset with predictions, or None if prediction cannot be performed.
        """
        # Define the time windows:
        # - Historical context: used for features (lags, etc.)
        # - Forecast period: the period we want to predict
        predict_context_start = data.horizon - self.config.predict_context_length
        forecast_end = data.horizon + self.config.horizon_length

        # Extract the dataset including both historical context and forecast period
        predict_data = data.get_window_versioned(
            start=predict_context_start,
            end=forecast_end,  # Include the forecast period
            available_before=data.horizon,  # Only use data available at prediction time (prevents lookahead bias)
        )

        forecast = self.workflow.predict(
            data=predict_data,
            forecast_start=data.horizon,  # Where historical data ends and forecasting begins
        )

        return forecast


class Liander2024ForecastingWorkflow(CustomForecastingWorkflow):
    """For now just the same as CustomForecastingWorkflow until something specific for the benchmark is needed."""


def xgboost_forecaster_factory(
    context: BenchmarkContext,
    target: BenchmarkTarget,
) -> OpenSTEF4Forecaster:
    """Factory function that creates a WorkflowBacktestAdapter with XGBoost model.

    Args:
        context: Benchmark context with run metadata
        target: Target specification for the benchmark

    Returns:
        WorkflowBacktestAdapter wrapping the configured forecasting workflow
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
    workflow = Liander2024ForecastingWorkflow(model=model, model_id=f"xgboost_{target.name}", callbacks=[])

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

    # Wrap the workflow in the adapter to make it compatible with BacktestForecasterMixin
    return OpenSTEF4Forecaster(config=backtest_config, workflow=workflow)


# TODO: def gblinear_forecaster_factory(


if __name__ == "__main__":
    storage = LocalBenchmarkStorage(base_path=BENCHMARK_RESULTS_PATH)
    runner = create_liander2024_benchmark_runner(
        data_dir=Path(
            "../liander2024-stef-benchmark"
        ),  # TODO: remove in final version to force download from HuggingFace
        storage=storage,
        callbacks=[StrictExecutionCallback()],
    )
    # Run for XGBoost model
    runner.run(
        forecaster_factory=xgboost_forecaster_factory,
        run_name="OpenSTEF_XGBoost",
        n_processes=N_PROCESSES,
    )
    # Run for GBLinear model
    # runner.run(
    #     forecaster_factory=gblinear_forecaster_factory,
    #     run_name="OpenSTEF_GBLinear",
    #     n_processes=N_PROCESSES,
    # )
