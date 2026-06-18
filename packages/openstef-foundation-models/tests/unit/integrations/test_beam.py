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

from openstef_beam.backtesting.backtest_forecaster.mixins import BacktestForecasterConfig
from openstef_beam.backtesting.restricted_horizon_timeseries import RestrictedHorizonVersionedTimeSeries
from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.datasets.versioned_timeseries_dataset import VersionedTimeSeriesDataset
from openstef_core.mixins.predictor import HyperParams
from openstef_core.types import LeadTime, Quantile
from openstef_foundation_models.integrations.beam import FoundationModelBacktestForecaster
from openstef_models.models.forecasting.forecaster import Forecaster
from openstef_models.models.forecasting_model import ForecastingModel
from openstef_models.workflows.custom_forecasting_workflow import CustomForecastingWorkflow

SAMPLE_INTERVAL = timedelta(minutes=15)
QUANTILES = [Quantile(0.1), Quantile(0.5), Quantile(0.9)]


class CountingForecaster(Forecaster):
    """Zero-shot fake forecaster producing a fixed quantile forecast per window."""

    HyperParams: ClassVar[type[HyperParams]] = HyperParams

    hyperparams: HyperParams = Field(default_factory=HyperParams)

    received_inputs: list[ForecastInputDataset] = Field(default_factory=list)

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
        self.received_inputs.append(data)
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


def _make_dataset_with_weather_forecast(
    periods: int = 200,
) -> tuple[VersionedTimeSeriesDataset, pd.DatetimeIndex]:
    """Build a dataset with a target measured per-timestamp and a weather forecast.

    The target's ``available_at`` equals its own timestamp (so future target values
    are not available before the horizon), while the weather covariate is a forecast
    issued at the very start (available well before any horizon, including for future
    timestamps).
    """
    timestamps = pd.date_range(start="2025-01-01", periods=periods, freq=SAMPLE_INTERVAL, name="timestamp")
    target = pd.DataFrame({"available_at": timestamps, "load": np.arange(periods, dtype=float)}, index=timestamps)
    weather = pd.DataFrame(
        {"available_at": timestamps[0], "temperature": np.arange(periods, dtype=float) * 0.1},
        index=timestamps,
    )
    dataset = VersionedTimeSeriesDataset(
        data_parts=[
            VersionedTimeSeriesDataset.from_dataframe(target, SAMPLE_INTERVAL).data_parts[0],
            VersionedTimeSeriesDataset.from_dataframe(weather, SAMPLE_INTERVAL).data_parts[0],
        ]
    )
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
    adapter = FoundationModelBacktestForecaster(workflow=_make_workflow(forecaster))

    # Act
    forecast = adapter.predict(_restricted(dataset, horizon))

    # Assert
    assert forecast is not None
    assert forecast.data.index[0].to_pydatetime() == horizon
    assert {q.format() for q in QUANTILES} <= set(forecast.data.columns)


def test_predict_window_includes_future_covariates_without_leaking_target(
    forecaster: CountingForecaster,
) -> None:
    """The window extends past the horizon so future weather forecasts reach the model.

    Future covariate values (available before the horizon) must be present for the
    prediction period, while future target actuals (only available after the horizon)
    must stay absent to avoid look-ahead leakage.
    """
    # Arrange
    dataset, timestamps = _make_dataset_with_weather_forecast()
    horizon = timestamps[150].to_pydatetime()
    adapter = FoundationModelBacktestForecaster(workflow=_make_workflow(forecaster))

    # Act
    adapter.predict(_restricted(dataset, horizon))

    # Assert
    received = forecaster.received_inputs[-1].data
    future = received[received.index >= horizon]
    assert not future.empty, "window must include rows at/after the horizon"
    # Weather forecast is known for the future horizon...
    assert future["temperature"].notna().all()
    # ...but target actuals strictly after the horizon must not leak.
    assert received[received.index > horizon]["load"].isna().all()


def test_predict_returns_none_when_no_history(forecaster: CountingForecaster) -> None:
    """With no observed target before the horizon, no forecast can be produced."""
    # Arrange
    dataset, timestamps = _make_dataset()
    horizon = timestamps[0].to_pydatetime()  # nothing strictly before the first timestamp
    adapter = FoundationModelBacktestForecaster(workflow=_make_workflow(forecaster))

    # Act
    forecast = adapter.predict(_restricted(dataset, horizon))

    # Assert
    assert forecast is None


def test_default_config_is_zero_shot_and_load_once(forecaster: CountingForecaster) -> None:
    """Omitting the config yields a zero-shot default with a non-zero prediction length."""
    # Arrange / Act
    adapter = FoundationModelBacktestForecaster(workflow=_make_workflow(forecaster))

    # Assert
    assert adapter.config.requires_training is False
    assert adapter.config.predict_length == timedelta(hours=48)


def test_from_workflow_sizes_predict_length_to_the_workflow_horizon(forecaster: CountingForecaster) -> None:
    """from_workflow defaults the prediction length to the workflow's maximum horizon."""
    # Arrange / Act
    adapter = FoundationModelBacktestForecaster.from_workflow(_make_workflow(forecaster))

    # Assert
    assert adapter.config.requires_training is False
    assert adapter.config.predict_length == forecaster.max_horizon.value


def test_from_workflow_accepts_an_explicit_predict_length(forecaster: CountingForecaster) -> None:
    """An explicit predict_length overrides the workflow-derived default."""
    # Arrange / Act
    adapter = FoundationModelBacktestForecaster.from_workflow(
        _make_workflow(forecaster), predict_length=timedelta(hours=6)
    )

    # Assert
    assert adapter.config.predict_length == timedelta(hours=6)


def test_explicit_config_overrides_the_derived_default(forecaster: CountingForecaster) -> None:
    """A config passed at construction is kept verbatim, not replaced by the default."""
    # Arrange
    config = BacktestForecasterConfig(
        requires_training=False,
        predict_length=timedelta(hours=6),
        predict_min_length=timedelta(minutes=15),
        predict_context_length=timedelta(days=1),
        predict_context_min_coverage=0.0,
        training_context_length=timedelta(0),
        training_context_min_coverage=0.0,
    )

    # Act
    adapter = FoundationModelBacktestForecaster(workflow=_make_workflow(forecaster), config=config)

    # Assert
    assert adapter.config is config
    assert adapter.config.predict_length == timedelta(hours=6)


def test_quantiles_delegate_to_forecaster(forecaster: CountingForecaster) -> None:
    """The adapter exposes the wrapped forecaster's quantiles."""
    # Arrange / Act
    adapter = FoundationModelBacktestForecaster(workflow=_make_workflow(forecaster))

    # Assert
    assert adapter.quantiles == QUANTILES


def test_adapter_reuses_single_forecaster_instance(forecaster: CountingForecaster) -> None:
    """The same forecaster instance backs every window (no per-window rebuild)."""
    # Arrange
    adapter = FoundationModelBacktestForecaster(workflow=_make_workflow(forecaster))

    # Act / Assert
    assert isinstance(adapter, FoundationModelBacktestForecaster)
    assert isinstance(adapter.workflow.model, ForecastingModel)
    assert adapter.workflow.model.forecaster is forecaster
