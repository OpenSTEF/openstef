# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Integration test: a real Chronos-2 forecaster backtested through the adapter.

Verifies the load-once guarantee with the real ONNX session — a single backend
is shared across every backtest window — and that the adapter yields valid
forecasts for each window.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_beam.backtesting.restricted_horizon_timeseries import RestrictedHorizonVersionedTimeSeries
from openstef_core.datasets.versioned_timeseries_dataset import VersionedTimeSeriesDataset
from openstef_core.types import LeadTime, Quantile
from openstef_foundation_models.inference.onnx_backend import OnnxBackend
from openstef_foundation_models.integrations.beam import FoundationModelBacktestForecaster
from openstef_foundation_models.models.forecasting import Chronos2Forecaster
from openstef_models.models.forecasting_model import ForecastingModel
from openstef_models.workflows.custom_forecasting_workflow import CustomForecastingWorkflow

pytestmark = [pytest.mark.slow, pytest.mark.integration]

SAMPLE_INTERVAL = timedelta(minutes=15)
QUANTILES = [Quantile(0.1), Quantile(0.5), Quantile(0.9)]


def _make_dataset(periods: int) -> tuple[VersionedTimeSeriesDataset, pd.DatetimeIndex]:
    """Build a versioned 'load' dataset available at each timestamp."""
    timestamps = pd.date_range(start="2025-01-01", periods=periods, freq=SAMPLE_INTERVAL, name="timestamp")
    steps_per_day = int(timedelta(days=1) / SAMPLE_INTERVAL)
    load = 100.0 + 30.0 * np.sin(2 * np.pi * np.arange(periods) / steps_per_day)
    data = pd.DataFrame({"available_at": timestamps, "load": load}, index=timestamps)
    dataset = VersionedTimeSeriesDataset.from_dataframe(data=data, sample_interval=SAMPLE_INTERVAL)
    return dataset, timestamps


def _restricted(dataset: VersionedTimeSeriesDataset, horizon: datetime) -> RestrictedHorizonVersionedTimeSeries:
    return RestrictedHorizonVersionedTimeSeries(dataset=dataset, horizon=horizon)


def test_adapter_runs_real_forecaster_load_once(onnx_backend: OnnxBackend) -> None:
    """A multi-window backtest reuses one ONNX session and forecasts every window."""
    # Arrange
    forecaster = Chronos2Forecaster(
        backend=onnx_backend,
        quantiles=QUANTILES,
        horizons=[LeadTime.from_string("PT6H")],
    )
    workflow = CustomForecastingWorkflow(
        model=ForecastingModel(forecaster=forecaster, target_column="load"),
        model_id="chronos2-backtest",
    )
    adapter = FoundationModelBacktestForecaster(workflow=workflow)
    dataset, timestamps = _make_dataset(periods=10 * 96)
    horizons = [timestamps[index].to_pydatetime() for index in (4 * 96, 6 * 96, 8 * 96)]
    windows = [_restricted(dataset, horizon) for horizon in horizons]
    session_before = onnx_backend._session  # the same session object must be reused across windows

    # Act
    forecasts = [adapter.predict(window) for window in windows]

    # Assert
    assert len(forecasts) == len(windows)
    for forecast, horizon in zip(forecasts, horizons, strict=True):
        assert forecast is not None
        assert forecast.data.index[0].to_pydatetime() == horizon
        assert {q.format() for q in QUANTILES} <= set(forecast.data.columns)
    # Load-once: the same forecaster (and its single ONNX session) served every window.
    assert isinstance(adapter.workflow.model, ForecastingModel)
    assert adapter.workflow.model.forecaster is forecaster
    assert onnx_backend._session is session_before
