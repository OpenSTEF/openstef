# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""OpenSTEF 4.0 forecaster for backtesting pipelines."""

from collections.abc import Callable
from typing import override

from openstef_beam.backtesting.backtest_forecaster.mixins import BacktestForecasterConfig, BacktestForecasterMixin
from openstef_beam.backtesting.restricted_horizon_timeseries import RestrictedHorizonVersionedTimeSeries
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.types import Q
from openstef_models.workflows.custom_forecasting_workflow import CustomForecastingWorkflow


class OpenSTEF4BacktestForecaster(BacktestForecasterMixin):
    """Forecaster that allows using a ForecastingWorkflow to be used in backtesting, specifically for OpenSTEF4 models.

    A new workflow is created each time fit() is called using the provided workflow_factory,
    ensuring fresh model instances for each training cycle during benchmarking.
    """

    def __init__(self, config: BacktestForecasterConfig, workflow_factory: Callable[[], CustomForecastingWorkflow]):
        """Initialize the forecaster.

        Args:
            config: Configuration for the backtest forecaster interface
            workflow_factory: Factory function that creates a new CustomForecastingWorkflow instance
        """
        self.config = config
        self.workflow_factory = workflow_factory
        self._workflow: CustomForecastingWorkflow | None = None

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
        training_data = data.get_window_versioned(start=training_start, end=training_end, available_before=data.horizon)

        # Use the workflow's fit method
        self._workflow.fit(data=training_data)

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
        predict_data = data.get_window_versioned(
            start=predict_context_start,
            end=forecast_end,  # Include the forecast period
            available_before=data.horizon,  # Only use data available at prediction time (prevents lookahead bias)
        )

        return self._workflow.predict(
            data=predict_data,
            forecast_start=data.horizon,  # Where historical data ends and forecasting begins
        )
