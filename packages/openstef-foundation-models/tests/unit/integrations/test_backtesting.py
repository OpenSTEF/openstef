# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the foundation-model backtesting adapter."""

from datetime import datetime, timedelta
from typing import ClassVar, override

import numpy as np
import pandas as pd
import pytest
from pydantic import Field, PrivateAttr

from openstef_beam.backtesting.restricted_horizon_timeseries import RestrictedHorizonVersionedTimeSeries
from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.datasets.versioned_timeseries_dataset import VersionedTimeSeriesDataset
from openstef_core.mixins.predictor import BatchResult, HyperParams
from openstef_core.types import LeadTime, Quantile
from openstef_foundation_models.integrations.backtesting import (
    FoundationModelBacktestForecaster,
    create_foundation_model_backtest_forecaster,
)
from openstef_models.models.forecasting.forecaster import Forecaster

SAMPLE_INTERVAL = timedelta(minutes=15)
QUANTILES = [Quantile(0.1), Quantile(0.5), Quantile(0.9)]


class CountingForecaster(Forecaster):
    """Zero-shot fake forecaster that counts backend calls (one per batch run)."""

    HyperParams: ClassVar[type[HyperParams]] = HyperParams

    hyperparams: HyperParams = Field(default_factory=HyperParams)
    _predict_batch_calls: int = PrivateAttr(default=0)
    _last_batch_size: int = PrivateAttr(default=0)

    @property
    @override
    def hparams(self) -> HyperParams:
        return self.hyperparams

    @property
    @override
    def is_fitted(self) -> bool:
        return True

    @override
    def fit(self, data: ForecastInputDataset, data_val: ForecastInputDataset | None = None) -> None:
        pass

    @override
    def predict(self, data: ForecastInputDataset) -> ForecastDataset:
        return self._build_forecast(data)

    @override
    def predict_batch(self, data: list[ForecastInputDataset]) -> BatchResult[ForecastDataset]:
        self._predict_batch_calls += 1
        self._last_batch_size = len(data)
        results: BatchResult[ForecastDataset] = []
        results.extend(self._build_forecast(item) for item in data)
        return results

    def _build_forecast(self, data: ForecastInputDataset) -> ForecastDataset:
        index = data.create_forecast_range(self.max_horizon)
        predictions = np.tile(np.array([1.0, 2.0, 3.0]), (len(index), 1))
        return ForecastDataset.from_quantile_predictions(
            predictions=predictions,
            index=index,
            quantiles=self.quantiles,
            sample_interval=data.sample_interval,
            target_column=data.target_column,
        )


def _make_dataset(periods: int = 200) -> tuple[VersionedTimeSeriesDataset, pd.DatetimeIndex]:
    """Build a versioned dataset with a 'load' target available at each timestamp."""
    timestamps = pd.date_range(start="2025-01-01", periods=periods, freq=SAMPLE_INTERVAL, name="timestamp")
    data = pd.DataFrame(
        {
            "available_at": timestamps,
            "load": np.arange(periods, dtype=float),
        },
        index=timestamps,
    )
    dataset = VersionedTimeSeriesDataset.from_dataframe(data=data, sample_interval=SAMPLE_INTERVAL)
    return dataset, timestamps


def _restricted(dataset: VersionedTimeSeriesDataset, horizon: datetime) -> RestrictedHorizonVersionedTimeSeries:
    return RestrictedHorizonVersionedTimeSeries(dataset=dataset, horizon=horizon)


@pytest.fixture
def forecaster() -> CountingForecaster:
    return CountingForecaster(quantiles=QUANTILES, horizons=[LeadTime.from_string("PT2H")])


def test_predict_returns_forecast_indexed_from_horizon(forecaster: CountingForecaster) -> None:
    """A single window predict returns a quantile forecast starting at the horizon."""
    # Arrange
    dataset, timestamps = _make_dataset()
    horizon = timestamps[150].to_pydatetime()
    adapter = create_foundation_model_backtest_forecaster(forecaster)

    # Act
    forecast = adapter.predict(_restricted(dataset, horizon))

    # Assert
    assert forecast is not None
    assert forecast.data.index[0].to_pydatetime() == horizon
    assert set(forecast.data.columns) == {q.format() for q in QUANTILES}


def test_predict_returns_none_when_no_history(forecaster: CountingForecaster) -> None:
    """With no observed target before the horizon, no forecast can be produced."""
    # Arrange
    dataset, timestamps = _make_dataset()
    horizon = timestamps[0].to_pydatetime()  # nothing strictly before the first timestamp
    adapter = create_foundation_model_backtest_forecaster(forecaster)

    # Act
    forecast = adapter.predict(_restricted(dataset, horizon))

    # Assert
    assert forecast is None


def test_predict_batch_uses_single_backend_call(forecaster: CountingForecaster) -> None:
    """A batch of windows is forecast in one backend call (load-once), order preserved."""
    # Arrange
    dataset, timestamps = _make_dataset()
    horizons = [timestamps[i].to_pydatetime() for i in (120, 140, 160)]
    adapter = create_foundation_model_backtest_forecaster(forecaster, batch_size=8)

    # Act
    results = adapter.predict_batch([_restricted(dataset, h) for h in horizons])

    # Assert
    assert forecaster._predict_batch_calls == 1
    assert forecaster._last_batch_size == 3
    assert [r.data.index[0].to_pydatetime() for r in results] == horizons


def test_predict_batch_preserves_none_positions(forecaster: CountingForecaster) -> None:
    """Windows without history yield None while valid windows run in one batch call."""
    # Arrange
    dataset, timestamps = _make_dataset()
    horizons = [
        timestamps[0].to_pydatetime(),  # no history -> None
        timestamps[140].to_pydatetime(),
        timestamps[160].to_pydatetime(),
    ]
    adapter = create_foundation_model_backtest_forecaster(forecaster, batch_size=8)

    # Act
    results = adapter.predict_batch([_restricted(dataset, h) for h in horizons])

    # Assert
    assert results[0] is None
    assert results[1] is not None
    assert results[2] is not None
    assert forecaster._predict_batch_calls == 1
    assert forecaster._last_batch_size == 2  # only the two valid windows reach the backend


def test_factory_disables_training_and_derives_predict_length(forecaster: CountingForecaster) -> None:
    """The default config is zero-shot and its horizon matches the forecaster."""
    # Arrange / Act
    adapter = create_foundation_model_backtest_forecaster(forecaster)

    # Assert
    assert adapter.config.requires_training is False
    assert adapter.config.predict_length == forecaster.max_horizon.value


def test_quantiles_delegate_to_forecaster(forecaster: CountingForecaster) -> None:
    """The adapter exposes the wrapped forecaster's quantiles."""
    # Arrange / Act
    adapter = create_foundation_model_backtest_forecaster(forecaster)

    # Assert
    assert adapter.quantiles == QUANTILES


def test_adapter_reuses_single_forecaster_instance(forecaster: CountingForecaster) -> None:
    """The same forecaster instance backs every window (no per-window rebuild)."""
    # Arrange
    adapter = create_foundation_model_backtest_forecaster(forecaster)

    # Act / Assert
    assert isinstance(adapter, FoundationModelBacktestForecaster)
    assert adapter.forecaster is forecaster
