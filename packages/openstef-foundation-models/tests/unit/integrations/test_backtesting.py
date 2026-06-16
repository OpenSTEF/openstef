# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the foundation-model backtesting adapter."""

from datetime import datetime, timedelta
from typing import ClassVar, override

import numpy as np
import pandas as pd
import pytest
from pydantic import Field

from openstef_beam.backtesting.restricted_horizon_timeseries import RestrictedHorizonVersionedTimeSeries
from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.datasets.versioned_timeseries_dataset import VersionedTimeSeriesDataset
from openstef_core.mixins.predictor import HyperParams
from openstef_core.types import LeadTime, Quantile
from openstef_foundation_models.integrations.backtesting import (
    FoundationModelBacktestForecaster,
    create_foundation_model_backtest_forecaster,
)
from openstef_models.models.forecasting.forecaster import Forecaster
from openstef_models.models.forecasting_model import ForecastingModel
from openstef_models.workflows.custom_forecasting_workflow import CustomForecastingWorkflow

SAMPLE_INTERVAL = timedelta(minutes=15)
QUANTILES = [Quantile(0.1), Quantile(0.5), Quantile(0.9)]


class CountingForecaster(Forecaster):
    """Zero-shot fake forecaster producing a fixed quantile forecast per window."""

    HyperParams: ClassVar[type[HyperParams]] = HyperParams

    hyperparams: HyperParams = Field(default_factory=HyperParams)

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


def _make_workflow(forecaster: Forecaster) -> CustomForecastingWorkflow:
    """Wrap a forecaster in a minimal single-forecaster workflow."""
    return CustomForecastingWorkflow(
        model=ForecastingModel(forecaster=forecaster, target_column="load"),
        model_id="test",
    )


@pytest.fixture
def forecaster() -> CountingForecaster:
    return CountingForecaster(quantiles=QUANTILES, horizons=[LeadTime.from_string("PT2H")])


def test_predict_returns_forecast_indexed_from_horizon(forecaster: CountingForecaster) -> None:
    """A single window predict returns a quantile forecast starting at the horizon."""
    # Arrange
    dataset, timestamps = _make_dataset()
    horizon = timestamps[150].to_pydatetime()
    adapter = create_foundation_model_backtest_forecaster(_make_workflow(forecaster))

    # Act
    forecast = adapter.predict(_restricted(dataset, horizon))

    # Assert
    assert forecast is not None
    assert forecast.data.index[0].to_pydatetime() == horizon
    assert {q.format() for q in QUANTILES} <= set(forecast.data.columns)


def test_predict_returns_none_when_no_history(forecaster: CountingForecaster) -> None:
    """With no observed target before the horizon, no forecast can be produced."""
    # Arrange
    dataset, timestamps = _make_dataset()
    horizon = timestamps[0].to_pydatetime()  # nothing strictly before the first timestamp
    adapter = create_foundation_model_backtest_forecaster(_make_workflow(forecaster))

    # Act
    forecast = adapter.predict(_restricted(dataset, horizon))

    # Assert
    assert forecast is None


def test_factory_disables_training_and_derives_predict_length(forecaster: CountingForecaster) -> None:
    """The default config is zero-shot and its horizon matches the forecaster."""
    # Arrange / Act
    adapter = create_foundation_model_backtest_forecaster(_make_workflow(forecaster))

    # Assert
    assert adapter.config.requires_training is False
    assert adapter.config.predict_length == forecaster.max_horizon.value


def test_quantiles_delegate_to_forecaster(forecaster: CountingForecaster) -> None:
    """The adapter exposes the wrapped forecaster's quantiles."""
    # Arrange / Act
    adapter = create_foundation_model_backtest_forecaster(_make_workflow(forecaster))

    # Assert
    assert adapter.quantiles == QUANTILES


def test_adapter_reuses_single_forecaster_instance(forecaster: CountingForecaster) -> None:
    """The same forecaster instance backs every window (no per-window rebuild)."""
    # Arrange
    adapter = create_foundation_model_backtest_forecaster(_make_workflow(forecaster))

    # Act / Assert
    assert isinstance(adapter, FoundationModelBacktestForecaster)
    assert adapter.workflow.model.forecaster is forecaster
