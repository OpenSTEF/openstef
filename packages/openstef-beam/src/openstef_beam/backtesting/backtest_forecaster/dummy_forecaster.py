# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

from openstef_beam.backtesting.backtest_forecaster.mixins import BacktestForecasterConfig, BacktestForecasterMixin
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.versioned_timeseries.accessors import RestrictedHorizonVersionedTimeSeries


class DummyForecaster(BacktestForecasterMixin):
    def __init__(
        self,
        config: BacktestForecasterConfig | None = None,
        predict_quantiles: list[float] | None = None,
    ) -> None:
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
        self.predict_quantiles = predict_quantiles or [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]

    @property
    def quantiles(self) -> list[float]:
        return self.predict_quantiles

    def predict(self, data: RestrictedHorizonVersionedTimeSeries) -> TimeSeriesDataset | None:
        raise NotImplementedError
