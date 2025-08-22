# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import pytest

from openstef_beam.backtesting import BacktestConfig, BacktestPipeline
from openstef_beam.backtesting.backtest_forecaster.mixins import (
    BacktestForecasterConfig,
    BacktestForecasterMixin,
)
from openstef_core.datasets import TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.datasets.versioned_timeseries_accessors import RestrictedHorizonVersionedTimeSeries


class MockForecaster(BacktestForecasterMixin):
    """Test implementation of the new forecaster interface for pipeline testing."""

    def __init__(self, config: BacktestForecasterConfig):
        self.config = config
        self.train_calls: list[Any] = []
        self.predict_calls: list[Any] = []
        self._predict_return_value: TimeSeriesDataset | None = None
        self._predict_side_effect: Callable[[RestrictedHorizonVersionedTimeSeries], TimeSeriesDataset | None] | None = (
            None
        )

    @property
    def quantiles(self) -> list[float]:
        return [0.5]

    def fit(self, data: RestrictedHorizonVersionedTimeSeries) -> None:
        self.train_calls.append(data)

    def predict(self, data: RestrictedHorizonVersionedTimeSeries) -> TimeSeriesDataset | None:
        """Return a TimeSeriesDataset or None (to simulate model unable to predict)."""
        self.predict_calls.append(data)

        if self._predict_side_effect:
            return self._predict_side_effect(data)

        if self._predict_return_value is not None:
            return self._predict_return_value

        # Default prediction: single timestamp at the horizon with a P50 quantile
        return TimeSeriesDataset(
            data=pd.DataFrame({"quantile_P50": [100.0]}, index=pd.DatetimeIndex([data.horizon])),
            sample_interval=self.config.predict_sample_interval,
        )

    def set_predict_return_value(self, value: TimeSeriesDataset | None) -> None:
        self._predict_return_value = value
        self._predict_side_effect = None

    def set_predict_side_effect(
        self, func: Callable[[RestrictedHorizonVersionedTimeSeries], TimeSeriesDataset | None]
    ) -> None:
        # func should accept the horizon-transformed data and return TimeSeriesDataset | None
        self._predict_side_effect = func
        self._predict_return_value = None

    @property
    def train_call_count(self) -> int:
        return len(self.train_calls)

    @property
    def predict_call_count(self) -> int:
        return len(self.predict_calls)


@pytest.fixture
def datasets() -> tuple[VersionedTimeSeriesDataset, VersionedTimeSeriesDataset]:
    """Combined fixture for both ground truth and predictors datasets."""
    timestamps = pd.date_range("2025-01-01", "2025-01-05", freq="1h")

    ground_truth = VersionedTimeSeriesDataset(
        data=pd.DataFrame({"timestamp": timestamps, "available_at": timestamps, "target": range(len(timestamps))}),
        sample_interval=timedelta(hours=1),
    )

    predictors = VersionedTimeSeriesDataset(
        data=pd.DataFrame({
            "timestamp": timestamps,
            "available_at": timestamps,
            "feature1": range(len(timestamps)),
            "feature2": range(len(timestamps), 2 * len(timestamps)),
        }),
        sample_interval=timedelta(hours=1),
    )

    return ground_truth, predictors


@pytest.mark.parametrize(
    ("requires_training", "expected_train_calls", "expected_predict_calls"),
    [
        pytest.param(True, ">0", ">0", id="with_training"),
        pytest.param(False, "==0", ">0", id="without_training"),
    ],
)
def test_run_training_scenarios(
    datasets: tuple[VersionedTimeSeriesDataset, VersionedTimeSeriesDataset],
    requires_training: bool,
    expected_train_calls: str,
    expected_predict_calls: str,
) -> None:
    # Arrange
    ground_truth, predictors = datasets
    config = BacktestConfig(predict_interval=timedelta(hours=6), train_interval=timedelta(hours=12))
    forecaster_config = BacktestForecasterConfig(
        requires_training=requires_training,
        horizon_length=timedelta(hours=24),
        horizon_min_length=timedelta(hours=1),
        predict_context_length=timedelta(hours=6),
        predict_context_min_coverage=0.5,
        training_context_length=timedelta(hours=12),
        training_context_min_coverage=0.5,
    )

    mock_forecaster = MockForecaster(forecaster_config)
    pipeline = BacktestPipeline(config=config, forecaster=mock_forecaster)

    # Act
    result = pipeline.run(
        ground_truth=ground_truth,
        predictors=predictors,
        start=datetime.fromisoformat("2025-01-02T00:00:00"),
        end=datetime.fromisoformat("2025-01-02T18:00:00"),
        show_progress=False,
    )

    # Assert
    assert isinstance(result, VersionedTimeSeriesDataset)
    assert result.sample_interval == forecaster_config.predict_sample_interval

    # Validate call counts
    if expected_train_calls == ">0":
        assert mock_forecaster.train_call_count > 0
    else:
        assert mock_forecaster.train_call_count == 0

    if expected_predict_calls == ">0":
        assert mock_forecaster.predict_call_count > 0
        assert len(result.data) > 0
        assert "quantile_P50" in result.data.columns


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (None, None),
        ("2025-01-02T00:00:00", None),
        (None, "2025-01-03T00:00:00"),
        ("2025-01-02T00:00:00", "2025-01-03T00:00:00"),
    ],
)
def test_run_date_boundary_handling(
    datasets: tuple[VersionedTimeSeriesDataset, VersionedTimeSeriesDataset],
    start: str | None,
    end: str | None,
) -> None:
    # Arrange
    ground_truth, predictors = datasets
    config = BacktestConfig()
    mock_forecaster = MockForecaster(
        BacktestForecasterConfig(
            requires_training=True,
            horizon_length=timedelta(hours=24),
            horizon_min_length=timedelta(hours=1),
            predict_context_length=timedelta(hours=6),
            predict_context_min_coverage=0.5,
            training_context_length=timedelta(hours=12),
            training_context_min_coverage=0.5,
        )
    )
    pipeline = BacktestPipeline(config=config, forecaster=mock_forecaster)

    start_dt = datetime.fromisoformat(start) if start else None
    end_dt = datetime.fromisoformat(end) if end else None

    # Act
    result = pipeline.run(
        ground_truth=ground_truth, predictors=predictors, start=start_dt, end=end_dt, show_progress=False
    )

    # Assert
    assert isinstance(result, VersionedTimeSeriesDataset)

    # Validate timestamps are within expected bounds
    if len(result.data) > 0:
        result_timestamps = result.data["timestamp"]
        if start_dt:
            assert (result_timestamps >= start_dt).all()
        if end_dt:
            assert (result_timestamps <= end_dt).all()


def test_run_output_validation_and_concatenation(
    datasets: tuple[VersionedTimeSeriesDataset, VersionedTimeSeriesDataset],
) -> None:
    # Arrange
    ground_truth, predictors = datasets
    config = BacktestConfig(predict_interval=timedelta(hours=6), train_interval=timedelta(hours=12))
    mock_forecaster = MockForecaster(
        BacktestForecasterConfig(
            requires_training=True,
            horizon_length=timedelta(hours=24),
            horizon_min_length=timedelta(hours=1),
            predict_context_length=timedelta(hours=6),
            predict_context_min_coverage=0.5,
            training_context_length=timedelta(hours=12),
            training_context_min_coverage=0.5,
        )
    )

    # Configure incremental predictions to test concatenation
    call_counter = 0

    def create_prediction(data: RestrictedHorizonVersionedTimeSeries) -> TimeSeriesDataset:
        nonlocal call_counter
        value = 100.0 + call_counter * 10
        call_counter += 1
        return TimeSeriesDataset(
            data=pd.DataFrame({"quantile_P50": [value]}, index=pd.DatetimeIndex([data.horizon])),
            sample_interval=mock_forecaster.config.predict_sample_interval,
        )

    mock_forecaster.set_predict_side_effect(create_prediction)
    pipeline = BacktestPipeline(config=config, forecaster=mock_forecaster)

    # Act
    result = pipeline.run(
        ground_truth=ground_truth,
        predictors=predictors,
        start=datetime.fromisoformat("2025-01-02T00:00:00"),
        end=datetime.fromisoformat("2025-01-02T18:00:00"),
        show_progress=False,
    )

    # Assert - Basic structure
    assert isinstance(result, VersionedTimeSeriesDataset)
    assert result.sample_interval == mock_forecaster.config.predict_sample_interval
    assert mock_forecaster.predict_call_count >= 2

    # Assert - Output validation
    result_data = result.data
    required_columns = ["timestamp", "available_at", "quantile_P50"]
    assert all(col in result_data.columns for col in required_columns)

    # Assert - Data types
    assert pd.api.types.is_datetime64_any_dtype(result_data["timestamp"])
    assert pd.api.types.is_datetime64_any_dtype(result_data["available_at"])
    assert pd.api.types.is_numeric_dtype(result_data["quantile_P50"])

    # Assert - Data quality
    assert not result_data["timestamp"].isna().any()
    assert not result_data["available_at"].isna().any()
    assert result_data["timestamp"].is_monotonic_increasing
    assert (result_data["quantile_P50"] >= 0).all()

    # Assert - Concatenation worked (multiple predictions with incremental values)
    assert len(result_data) >= 2
    prediction_values = result_data["quantile_P50"].tolist()
    assert len(set(prediction_values)) > 1
    assert prediction_values == sorted(prediction_values)

    # Verify incremental pattern
    for i in range(1, len(prediction_values)):
        assert prediction_values[i] - prediction_values[i - 1] == 10.0


def test_run_handles_none_predictions(datasets: tuple[VersionedTimeSeriesDataset, VersionedTimeSeriesDataset]) -> None:
    # Arrange
    ground_truth, predictors = datasets
    config = BacktestConfig()
    mock_forecaster = MockForecaster(
        BacktestForecasterConfig(
            requires_training=True,
            horizon_length=timedelta(hours=24),
            horizon_min_length=timedelta(hours=1),
            predict_context_length=timedelta(hours=6),
            predict_context_min_coverage=0.5,
            training_context_length=timedelta(hours=12),
            training_context_min_coverage=0.5,
        )
    )

    def return_none(data: RestrictedHorizonVersionedTimeSeries) -> TimeSeriesDataset | None:
        return None

    mock_forecaster.set_predict_side_effect(return_none)
    pipeline = BacktestPipeline(config=config, forecaster=mock_forecaster)

    # Act
    result = pipeline.run(
        ground_truth=ground_truth,
        predictors=predictors,
        start=datetime.fromisoformat("2025-01-02T00:00:00"),
        end=datetime.fromisoformat("2025-01-03T00:00:00"),
        show_progress=False,
    )

    # Assert
    assert isinstance(result, VersionedTimeSeriesDataset)
    result_data = result.data
    required_columns = ["timestamp", "available_at"]
    assert all(col in result_data.columns for col in required_columns)
    assert len(result_data) == 0


@pytest.mark.parametrize(
    ("scenario", "config_overrides", "dataset_type", "expected_calls"),
    [
        pytest.param(
            "insufficient_context",
            {"training_context_length": timedelta(days=10), "training_context_min_coverage": 0.8},
            "minimal",
            0,
            id="no_events_insufficient_context",
        ),
        pytest.param(
            "insufficient_coverage",
            {
                "predict_context_length": timedelta(hours=6),
                "predict_context_min_coverage": 0.9,
                "requires_training": False,
            },
            "sparse",
            0,
            id="no_events_insufficient_coverage",
        ),
    ],
)
def test_run_edge_cases(
    scenario: str,
    config_overrides: dict[str, Any],
    dataset_type: str,
    expected_calls: int,
) -> None:
    # Arrange
    config = BacktestConfig(predict_interval=timedelta(hours=6), train_interval=timedelta(hours=12))

    base_model_config = BacktestForecasterConfig(
        requires_training=True,
        horizon_length=timedelta(hours=24),
        horizon_min_length=timedelta(hours=1),
        predict_context_length=timedelta(hours=1),
        predict_context_min_coverage=0.8,
        training_context_length=timedelta(hours=12),
        training_context_min_coverage=0.5,
    )
    base_model_config = base_model_config.model_copy(update=config_overrides)

    mock_forecaster = MockForecaster(config=base_model_config)
    pipeline = BacktestPipeline(config=config, forecaster=mock_forecaster)

    # Create appropriate dataset
    if dataset_type == "minimal":
        timestamps = pd.date_range("2025-01-01T12:00:00", "2025-01-01T15:00:00", freq="1h")
        start_time = "2025-01-01T12:00:00"
        end_time = "2025-01-01T15:00:00"
    else:  # sparse
        timestamps = pd.DatetimeIndex(["2025-01-01T12:00:00", "2025-01-01T18:00:00"])
        start_time = "2025-01-01T18:00:00"
        end_time = "2025-01-01T20:00:00"

    ground_truth = VersionedTimeSeriesDataset(
        data=pd.DataFrame({"timestamp": timestamps, "available_at": timestamps, "target": range(len(timestamps))}),
        sample_interval=timedelta(hours=1),
    )
    predictors = VersionedTimeSeriesDataset(
        data=pd.DataFrame({"timestamp": timestamps, "available_at": timestamps, "feature1": range(len(timestamps))}),
        sample_interval=timedelta(hours=1),
    )

    # Act
    result = pipeline.run(
        ground_truth=ground_truth,
        predictors=predictors,
        start=datetime.fromisoformat(start_time),
        end=datetime.fromisoformat(end_time),
        show_progress=False,
    )

    # Assert
    assert isinstance(result, VersionedTimeSeriesDataset)
    assert mock_forecaster.predict_call_count == expected_calls
    assert len(result.data) == 0

    # Validate empty result structure
    required_columns = ["timestamp", "available_at"]
    assert all(col in result.data.columns for col in required_columns)
