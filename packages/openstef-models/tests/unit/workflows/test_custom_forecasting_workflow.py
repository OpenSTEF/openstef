# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, override

import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.validated_datasets import ForecastDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.testing import assert_timeseries_equal
from openstef_core.types import LeadTime, Quantile
from openstef_models.models.forecasting.constant_quantile_forecaster import ConstantQuantileForecaster
from openstef_models.models.forecasting_model import ForecastingModel
from openstef_models.testing import SimpleForecaster, create_sample_timeseries_dataset
from openstef_models.workflows.custom_forecasting_workflow import (
    CustomForecastingWorkflow,
    ForecastingCallback,
)

if TYPE_CHECKING:
    from openstef_models.mixins.callbacks import WorkflowContext


class _RecordingCallback(ForecastingCallback):
    """Callback that records batch lifecycle invocations."""

    def __init__(self) -> None:
        self.batch_start_calls = 0
        self.batch_end_calls = 0
        self.last_start_size: int | None = None
        self.last_end_size: int | None = None

    @override
    def on_predict_batch_start(
        self,
        context: "WorkflowContext[CustomForecastingWorkflow]",
        data: list[TimeSeriesDataset],
    ) -> None:
        self.batch_start_calls += 1
        self.last_start_size = len(data)

    @override
    def on_predict_batch_end(
        self,
        context: "WorkflowContext[CustomForecastingWorkflow]",
        data: list[TimeSeriesDataset],
        result: list[ForecastDataset],
    ) -> None:
        self.batch_end_calls += 1
        self.last_end_size = len(result)


@pytest.fixture
def sample_timeseries_dataset() -> TimeSeriesDataset:
    """Create sample time series data for workflow tests."""
    return create_sample_timeseries_dataset()


def _make_workflow(callbacks: list[ForecastingCallback] | None = None) -> CustomForecastingWorkflow:
    horizons = [LeadTime(timedelta(hours=6))]
    forecaster = SimpleForecaster(quantiles=[Quantile(0.5)], horizons=horizons)
    model = ForecastingModel(forecaster=forecaster)
    return CustomForecastingWorkflow(model_id="test_model", model=model, callbacks=callbacks or [])


def test_workflow__predict_batch_matches_loop(sample_timeseries_dataset: TimeSeriesDataset):
    """predict_batch returns a list aligned to input, equal to looping predict."""
    # Arrange
    workflow = _make_workflow()
    workflow.fit(sample_timeseries_dataset)
    items = [sample_timeseries_dataset, sample_timeseries_dataset]
    forecast_start = datetime.fromisoformat("2025-01-01T12:00:00")

    # Act
    batch = workflow.predict_batch(items, forecast_start=[forecast_start] * len(items))
    looped = [workflow.predict(item, forecast_start=forecast_start) for item in items]

    # Assert
    assert len(batch) == len(items)
    for batched, single in zip(batch, looped, strict=True):
        assert_timeseries_equal(batched, single)


def test_workflow__predict_batch_fires_callbacks(sample_timeseries_dataset: TimeSeriesDataset):
    """on_predict_batch_start / on_predict_batch_end fire with the correct sizes."""
    # Arrange
    callback = _RecordingCallback()
    workflow = _make_workflow(callbacks=[callback])
    workflow.fit(sample_timeseries_dataset)
    items = [sample_timeseries_dataset, sample_timeseries_dataset, sample_timeseries_dataset]

    # Act
    workflow.predict_batch(items, forecast_start=[datetime.fromisoformat("2025-01-01T12:00:00")] * len(items))

    # Assert
    assert callback.batch_start_calls == 1
    assert callback.batch_end_calls == 1
    assert callback.last_start_size == len(items)
    assert callback.last_end_size == len(items)


def test_workflow__predict_batch_not_fitted_raises_error(sample_timeseries_dataset: TimeSeriesDataset):
    """predict_batch raises NotFittedError when the underlying model is not fitted."""
    workflow = _make_workflow()

    with pytest.raises(NotFittedError):
        workflow.predict_batch(
            [sample_timeseries_dataset],
            forecast_start=[datetime.fromisoformat("2025-01-01T12:00:00")],
        )


def test_workflow__predict_batch_constant_forecaster(sample_timeseries_dataset: TimeSeriesDataset):
    """predict_batch works through a default-loop forecaster (no real batching)."""
    # Arrange
    horizons = [LeadTime(timedelta(hours=6))]
    forecaster = ConstantQuantileForecaster(horizons=horizons, quantiles=[Quantile(0.5)])
    model = ForecastingModel(forecaster=forecaster)
    workflow = CustomForecastingWorkflow(model_id="test_model", model=model)
    workflow.fit(sample_timeseries_dataset)
    items = [sample_timeseries_dataset, sample_timeseries_dataset]
    forecast_start = datetime.fromisoformat("2025-01-01T12:00:00")

    # Act
    batch = workflow.predict_batch(items, forecast_start=[forecast_start] * len(items))
    looped = [workflow.predict(item, forecast_start=forecast_start) for item in items]

    # Assert
    assert len(batch) == len(items)
    for batched, single in zip(batch, looped, strict=True):
        assert_timeseries_equal(batched, single)
