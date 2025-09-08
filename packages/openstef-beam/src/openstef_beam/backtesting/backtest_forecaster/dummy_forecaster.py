# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Minimal forecaster for testing backtesting pipelines.

Provides a no-op implementation that satisfies the backtesting interface
without requiring actual model training or prediction logic.
"""

from datetime import timedelta

from openstef_beam.backtesting.backtest_forecaster.mixins import BacktestForecasterConfig, BacktestForecasterMixin
from openstef_beam.backtesting.restricted_horizon_timeseries import RestrictedHorizonVersionedTimeSeries
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.types import Q, Quantile


class DummyForecaster(BacktestForecasterMixin):
    """Simple forecaster implementation for testing and development purposes.

    Provides a minimal implementation of the backtesting forecaster interface
    without actual prediction logic. Useful for testing pipeline components,
    debugging, and as a reference implementation.

    Attributes:
        config: Configuration parameters for the forecasting interface.
        predict_quantiles: List of quantiles to return in predictions.
    """

    def __init__(
        self,
        config: BacktestForecasterConfig | None = None,
        predict_quantiles: list[Quantile] | None = None,
    ) -> None:
        """Initialize the dummy forecaster with default configuration.

        Args:
            config: Forecaster configuration. If None, uses default no-training config.
            predict_quantiles: Quantiles to include in predictions. If None, uses
                             standard quantiles [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95].
        """
        super().__init__()
        self.config = config or BacktestForecasterConfig(
            requires_training=False,
            horizon_length=timedelta(days=7),
            horizon_min_length=timedelta(days=0),
            predict_context_length=timedelta(days=0),
            predict_context_min_coverage=0.0,
            training_context_length=timedelta(days=0),
            training_context_min_coverage=0.0,
            predict_sample_interval=timedelta(minutes=15),
        )
        self.predict_quantiles = predict_quantiles or [Q(0.05), Q(0.1), Q(0.3), Q(0.5), Q(0.7), Q(0.9), Q(0.95)]

    @property
    def quantiles(self) -> list[Quantile]:
        """Return the list of quantiles for predictions."""
        return self.predict_quantiles

    def predict(self, data: RestrictedHorizonVersionedTimeSeries) -> TimeSeriesDataset | None:
        """Placeholder prediction method that raises NotImplementedError.

        Args:
            data: Time series data with horizon restrictions for prediction.

        Returns:
            None - this implementation does not provide actual predictions.

        Raises:
            NotImplementedError: Always raised as this is a dummy implementation.
        """
        raise NotImplementedError
