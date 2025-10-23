# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for batch prediction functionality in backtesting pipeline."""

from datetime import datetime, time, timedelta

import pandas as pd
import pytest

from openstef_beam.backtesting import BacktestConfig, BacktestPipeline
from openstef_beam.backtesting.backtest_forecaster.mixins import (
    BacktestBatchForecasterMixin,
    BacktestForecasterConfig,
    BacktestForecasterMixin,
)
from openstef_beam.backtesting.restricted_horizon_timeseries import RestrictedHorizonVersionedTimeSeries
from openstef_core.datasets import TimeSeriesDataset, VersionedTimeSeriesDataset


class MockModelConfig(BacktestForecasterConfig):
    """Mock configuration for testing."""

    requires_training: bool = True
    batch_size: int | None = 4

    horizon_length: timedelta = timedelta(hours=6)
    horizon_min_length: timedelta = timedelta(hours=1)

    predict_context_length: timedelta = timedelta(hours=1)
    predict_context_min_coverage: float = 0.8

    training_context_length: timedelta = timedelta(hours=2)
    training_context_min_coverage: float = 0.9


class MockModel(BacktestBatchForecasterMixin, BacktestForecasterMixin):
    """Mock model for testing."""

    def __init__(self, batch_size: int | None = 4):
        # Create a simple config and allow overriding batch_size
        self.config = MockModelConfig()
        self.batch_size = batch_size
        self.fit_calls: list[datetime] = []
        self.predict_calls: list[datetime] = []
        self.predict_batch_calls: list[int] = []

    def fit(self, data: RestrictedHorizonVersionedTimeSeries) -> None:
        self.fit_calls.append(data.horizon)

    def predict(self, data: RestrictedHorizonVersionedTimeSeries) -> TimeSeriesDataset:
        self.predict_calls.append(data.horizon)
        # Return mock prediction as TimeSeriesDataset
        timestamps = pd.date_range(start=data.horizon, periods=2, freq="1h")
        return TimeSeriesDataset(
            data=pd.DataFrame({"quantile_P50": [0.5, 0.5]}, index=timestamps),
            sample_interval=self.config.predict_sample_interval,
        )

    def predict_batch(self, batch: list[RestrictedHorizonVersionedTimeSeries]) -> list[TimeSeriesDataset]:
        self.predict_batch_calls.append(len(batch))
        results: list[TimeSeriesDataset] = []
        for data in batch:
            timestamps = pd.date_range(start=data.horizon, periods=2, freq="1h")
            results.append(
                TimeSeriesDataset(
                    data=pd.DataFrame({"quantile_P50": [0.5, 0.5]}, index=timestamps),
                    sample_interval=self.config.predict_sample_interval,
                )
            )
        return results


@pytest.fixture
def mock_data() -> tuple[VersionedTimeSeriesDataset, VersionedTimeSeriesDataset]:
    """Create mock datasets for testing."""
    timestamps = pd.date_range(start="2024-01-01", periods=100, freq="1h")

    ground_truth = VersionedTimeSeriesDataset.from_dataframe(
        data=pd.DataFrame({"available_at": timestamps, "target": range(100)}, index=timestamps),
        sample_interval=timedelta(hours=1),
    )

    predictors = VersionedTimeSeriesDataset.from_dataframe(
        data=pd.DataFrame({"available_at": timestamps, "feature": range(100)}, index=timestamps),
        sample_interval=timedelta(hours=1),
    )

    return ground_truth, predictors


def test_batch_processing_efficiency():
    """Test that batch processing handles mixed events correctly and efficiently."""
    # This test validates the behavior without testing internal implementation details
    ground_truth_data = pd.DataFrame(
        {
            "available_at": pd.date_range(start="2024-01-01", periods=10, freq="1h"),
            "target": range(10),
        },
        index=pd.date_range(start="2024-01-01", periods=10, freq="1h"),
    )
    ground_truth = VersionedTimeSeriesDataset.from_dataframe(data=ground_truth_data, sample_interval=timedelta(hours=1))

    predictors_data = pd.DataFrame(
        {
            "available_at": pd.date_range(start="2024-01-01", periods=10, freq="1h"),
            "feature": range(10),
        },
        index=pd.date_range(start="2024-01-01", periods=10, freq="1h"),
    )
    predictors = VersionedTimeSeriesDataset.from_dataframe(data=predictors_data, sample_interval=timedelta(hours=1))

    # Test with batch size 3
    model = MockModel(batch_size=3)
    config = BacktestConfig(
        predict_interval=timedelta(hours=2),
        train_interval=timedelta(hours=4),
        align_time=time.fromisoformat("00:00+00"),
    )
    pipeline = BacktestPipeline(config=config, forecaster=model)

    # Run the pipeline - this will exercise the internal batching logic
    result = pipeline.run(
        ground_truth=ground_truth,
        predictors=predictors,
        start=datetime.fromisoformat("2024-01-01T00:00:00"),
        end=datetime.fromisoformat("2024-01-01T08:00:00"),
        show_progress=False,
    )

    # Verify that batching was used (should have batch calls)
    assert len(model.predict_batch_calls) > 0
    assert len(model.predict_calls) == 0  # Should not use individual prediction calls

    # Verify we got results
    assert len(result.data) > 0


def test_batch_prediction_is_used(mock_data: tuple[VersionedTimeSeriesDataset, VersionedTimeSeriesDataset]):
    """Test that batch prediction is used when supported."""
    ground_truth, predictors = mock_data

    model = MockModel(batch_size=4)
    config = BacktestConfig(
        predict_interval=timedelta(hours=6),
        train_interval=timedelta(days=1),
    )
    pipeline = BacktestPipeline(config=config, forecaster=model)

    # Run pipeline
    pipeline.run(
        ground_truth=ground_truth,
        predictors=predictors,
        start=datetime.fromisoformat("2024-01-01T06:00:00"),
        end=datetime.fromisoformat("2024-01-02T06:00:00"),
        show_progress=False,
    )

    # Verify batch prediction was used
    assert len(model.predict_batch_calls) > 0
    assert len(model.predict_calls) == 0  # Serial prediction should not be used


def test_serial_prediction_fallback(mock_data: tuple[VersionedTimeSeriesDataset, VersionedTimeSeriesDataset]):
    """Test that serial prediction is used when batching is not supported."""
    ground_truth, predictors = mock_data

    model = MockModel(batch_size=None)  # No batch support
    config = BacktestConfig(
        predict_interval=timedelta(hours=6),
        train_interval=timedelta(days=1),
    )
    pipeline = BacktestPipeline(config=config, forecaster=model)

    # Run pipeline
    pipeline.run(
        ground_truth=ground_truth,
        predictors=predictors,
        start=datetime.fromisoformat("2024-01-01T06:00:00"),
        end=datetime.fromisoformat("2024-01-02T06:00:00"),
        show_progress=False,
    )

    # Verify serial prediction was used
    assert len(model.predict_calls) > 0
    assert len(model.predict_batch_calls) == 0  # Batch prediction should not be used


def test_batch_size_one_uses_serial(mock_data: tuple[VersionedTimeSeriesDataset, VersionedTimeSeriesDataset]):
    """Test that batch_size=1 uses serial prediction."""
    ground_truth, predictors = mock_data

    model = MockModel(batch_size=1)  # Batch size 1 should trigger serial mode
    config = BacktestConfig(
        predict_interval=timedelta(hours=6),
        train_interval=timedelta(days=1),
    )
    pipeline = BacktestPipeline(config=config, forecaster=model)

    # Run pipeline
    pipeline.run(
        ground_truth=ground_truth,
        predictors=predictors,
        start=datetime.fromisoformat("2024-01-01T06:00:00"),
        end=datetime.fromisoformat("2024-01-02T06:00:00"),
        show_progress=False,
    )

    # Verify serial prediction was used (batch_size=1 doesn't benefit from batching)
    assert len(model.predict_calls) > 0
    assert len(model.predict_batch_calls) == 0
