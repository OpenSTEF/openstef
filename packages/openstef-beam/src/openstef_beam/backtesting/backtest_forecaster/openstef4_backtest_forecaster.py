# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""OpenSTEF 4.0 forecaster for backtesting pipelines."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, override

from pydantic import Field, PrivateAttr

from openstef_beam.backtesting.backtest_forecaster.mixins import BacktestForecasterConfig, BacktestForecasterMixin
from openstef_beam.backtesting.restricted_horizon_timeseries import RestrictedHorizonVersionedTimeSeries
from openstef_core.base_model import BaseModel
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.types import Q
from openstef_models.models.forecasting_model import restore_target
from openstef_models.workflows.custom_forecasting_workflow import CustomForecastingWorkflow


class OpenSTEF4BacktestForecaster(BaseModel, BacktestForecasterMixin):
    """Forecaster that allows using a ForecastingWorkflow to be used in backtesting, specifically for OpenSTEF4 models.

    A new workflow is created each time fit() is called using the provided workflow_factory,
    ensuring fresh model instances for each training cycle during benchmarking.
    """

    config: BacktestForecasterConfig = Field(
        description="Configuration for the backtest forecaster interface",
    )
    workflow_factory: Callable[[], CustomForecastingWorkflow] = Field(
        description="Factory function that creates a new CustomForecastingWorkflow instance",
    )
    cache_dir: Path = Field(
        description="Directory to use for caching model artifacts during backtesting",
    )
    debug: bool = Field(
        default=False,
        description="When True, saves intermediate input data for debugging",
    )

    _workflow: CustomForecastingWorkflow | None = PrivateAttr(default=None)

    @override
    def model_post_init(self, context: Any) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    @override
    def quantiles(self) -> list[Q]:
        # Create a workflow instance if needed to get quantiles
        if self._workflow is None:
            self._workflow = self.workflow_factory()
        # Extract quantiles from the workflow's model
        return self._workflow.model.forecaster.config.quantiles

    @override
    def fit(self, data: RestrictedHorizonVersionedTimeSeries) -> None:
        # Create a new workflow for this training cycle
        self._workflow = self.workflow_factory()

        # Get training data window based on config
        training_end = data.horizon
        training_start = training_end - self.config.training_context_length

        # Extract the versioned dataset for training
        training_data_versioned = data.get_window_versioned(
            start=training_start, end=training_end, available_before=data.horizon
        )
        # Convert to horizons
        training_data = training_data_versioned.to_horizons(horizons=self._workflow.model.config.horizons)
        training_data = restore_target(
            dataset=training_data,
            original_dataset=training_data_versioned.select_version(),
            target_column=self._workflow.model.target_column,
        )

        if self.debug:
            id_str = data.horizon.strftime("%Y%m%d%H%M%S")
            training_data.to_parquet(path=self.cache_dir / f"debug_{id_str}_training.parquet")

        # Use the workflow's fit method
        self._workflow.fit(data=training_data)

        if self.debug:
            id_str = data.horizon.strftime("%Y%m%d%H%M%S")
            self._workflow.model.prepare_input(training_data).to_parquet(  # pyright: ignore[reportPrivateUsage]
                path=self.cache_dir / f"debug_{id_str}_prepared_training.parquet"
            )

    @override
    def predict(self, data: RestrictedHorizonVersionedTimeSeries) -> TimeSeriesDataset | None:
        if self._workflow is None:
            raise NotFittedError("Must call fit() before predict()")

        # Define the time windows:
        # - Historical context: used for features (lags, etc.)
        # - Forecast period: the period we want to predict
        predict_context_start = data.horizon - self.config.predict_context_length
        forecast_end = data.horizon + self.config.horizon_length

        # Extract the dataset including both historical context and forecast period
        predict_data_versioned = data.get_window_versioned(
            start=predict_context_start,
            end=forecast_end,  # Include the forecast period
            available_before=data.horizon,  # Only use data available at prediction time (prevents lookahead bias)
        )
        # Convert to horizons
        predict_data = predict_data_versioned.to_horizons(horizons=self._workflow.model.config.horizons)
        predict_data = restore_target(
            dataset=predict_data,
            original_dataset=predict_data_versioned.select_version(),
            target_column=self._workflow.model.target_column,
        )

        forecast = self._workflow.predict(
            data=predict_data,
            forecast_start=data.horizon,  # Where historical data ends and forecasting begins
        )

        if self.debug:
            id_str = data.horizon.strftime("%Y%m%d%H%M%S")
            predict_data.to_parquet(path=self.cache_dir / f"debug_{id_str}_predict.parquet")
            forecast.to_parquet(path=self.cache_dir / f"debug_{id_str}_forecast.parquet")

        return forecast


__all__ = ["OpenSTEF4BacktestForecaster"]
