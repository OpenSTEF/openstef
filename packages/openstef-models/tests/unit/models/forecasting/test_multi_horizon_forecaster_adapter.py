# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta
from typing import Self, cast, override

import pandas as pd
import pytest

from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.mixins import State
from openstef_core.types import LeadTime, Quantile
from openstef_models.models.forecasting import HorizonForecaster, HorizonForecasterConfig
from openstef_models.models.forecasting.multi_horizon_forecaster_adapter import (
    MultiHorizonForecasterAdapter,
    MultiHorizonForecasterConfig,
    combine_horizon_forecasts,
)


class PredictableConstantForecaster(HorizonForecaster):
    """Always predicts constant 110.0 after fitting. Minimal test implementation.

    Behavior:
    - Before fit: raises error on predict
    - After fit: predicts [110.0, 110.0, ...] for all data points
    - State serialization: lazy (returns self)
    """

    def __init__(self, config: HorizonForecasterConfig, initial_value: float = 42.0) -> None:
        self._config = config
        self._initial_value = initial_value
        self._fitted_constant = initial_value  # Value used for predictions after fitting
        self._is_fitted = False

    @property
    @override
    def config(self) -> HorizonForecasterConfig:
        return self._config

    @property
    @override
    def is_fitted(self) -> bool:
        return self._is_fitted

    @override
    def to_state(self) -> State:
        return self  # Lazy: just return self for testing

    @override
    def from_state(self, state: State) -> Self:
        return cast(Self, state)

    @override
    def fit(self, data: ForecastInputDataset) -> None:
        # Learn from first data point: fitted_constant = first_target_value * 1.1
        # Use the target column from the input dataset when available, otherwise
        # fall back to the first column. We know the test data contains numeric
        # values, so this is safe for predictable testing.
        if data.target_column and data.target_column in data.data.columns:
            series = data.data[data.target_column]
        else:
            series = data.data.iloc[:, 0]
        first_value = float(series.iloc[0])
        self._fitted_constant = first_value * 1.1
        self._is_fitted = True

    @override
    def predict(self, data: ForecastInputDataset) -> ForecastDataset:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        return ForecastDataset(
            data=pd.DataFrame(
                data={quantile.format(): self._fitted_constant for quantile in self.config.quantiles},
                index=data.index,
            ),
            sample_interval=data.sample_interval,
            forecast_start=data.forecast_start,
        )


def model_factory(config: HorizonForecasterConfig) -> PredictableConstantForecaster:
    initial_value = 100.0 + config.horizons[0].value.total_seconds() / 3600  # 100 + hours
    return PredictableConstantForecaster(config=config, initial_value=initial_value)


# Fixtures for tests
@pytest.fixture
def sample_multi_horizon_config() -> MultiHorizonForecasterConfig[HorizonForecasterConfig]:
    """Create sample multi-horizon forecaster configuration."""
    return MultiHorizonForecasterConfig[HorizonForecasterConfig](
        horizons=[LeadTime(timedelta(hours=3)), LeadTime(timedelta(hours=6))],
        quantiles=[Quantile(0.5)],
        forecaster_config=HorizonForecasterConfig(
            quantiles=[Quantile(0.5)],
            horizons=[LeadTime(timedelta(hours=1))],  # Will be overridden
        ),
    )


@pytest.fixture
def sample_input_data() -> dict[LeadTime, ForecastInputDataset]:
    """Create sample input data for multiple horizons."""
    data_3h = pd.DataFrame(
        {"load": [100.0, 200.0, 300.0]},
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1h"),
    )
    data_6h = pd.DataFrame(
        {"load": [400.0, 500.0, 600.0]},
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1h"),
    )

    return {
        LeadTime(timedelta(hours=3)): ForecastInputDataset(
            data=data_3h, sample_interval=timedelta(hours=1), target_column="load"
        ),
        LeadTime(timedelta(hours=6)): ForecastInputDataset(
            data=data_6h, sample_interval=timedelta(hours=1), target_column="load"
        ),
    }


# Tests for MultiHorizonForecasterAdapter
def test_multi_horizon_forecaster__create_and_workflow(
    sample_multi_horizon_config: MultiHorizonForecasterConfig[HorizonForecasterConfig],
    sample_input_data: dict[LeadTime, ForecastInputDataset],
):
    """Test creating MultiHorizonForecasterAdapter and complete fit-predict workflow with exact value validation."""
    # Arrange
    forecaster = MultiHorizonForecasterAdapter.create(config=sample_multi_horizon_config, model_factory=model_factory)

    # Assert initial state
    assert len(forecaster._horizon_forecasters) == 2
    assert LeadTime(timedelta(hours=3)) in forecaster._horizon_forecasters
    assert LeadTime(timedelta(hours=6)) in forecaster._horizon_forecasters
    assert not forecaster.is_fitted

    # Act - fit and predict
    forecaster.fit(sample_input_data)
    result = forecaster.predict(sample_input_data)

    # Assert fitted state and prediction results
    assert forecaster.is_fitted
    assert isinstance(result, ForecastDataset)
    assert len(result.data) > 0
    assert "quantile_P50" in result.data.columns

    # Assert exact prediction values
    # Each forecaster gets fitted_constant = 110.0 (100.0 * 1.1) after fitting
    # Combined forecast should have this value for all data points
    result_values = result.data["quantile_P50"].tolist()
    expected_value = 110.0  # All forecasters predict the same after fitting
    assert all(abs(val - expected_value) < 0.001 for val in result_values), (
        f"Expected all values to be ~{expected_value}, got {result_values}"
    )
    assert len(result_values) == 3, f"Expected 3 forecast points, got {len(result_values)}"


def test_multi_horizon_forecaster__state_serialize_restore(
    sample_multi_horizon_config: MultiHorizonForecasterConfig[HorizonForecasterConfig],
    sample_input_data: dict[LeadTime, ForecastInputDataset],
):
    """Test state serialization preserves learned behavior across different input data."""
    # Arrange - create and train original forecaster
    original_forecaster = MultiHorizonForecasterAdapter.create(
        config=sample_multi_horizon_config, model_factory=model_factory
    )
    original_forecaster.fit(sample_input_data)

    # Create different input data to test serialization meaningfully
    different_data = {
        LeadTime(timedelta(hours=3)): ForecastInputDataset(
            data=pd.DataFrame(
                {"load": [50.0, 60.0, 70.0]},
                index=pd.date_range(datetime.fromisoformat("2025-01-02T00:00:00"), periods=3, freq="1h"),
            ),
            sample_interval=timedelta(hours=1),
            target_column="load",
        ),
        LeadTime(timedelta(hours=6)): ForecastInputDataset(
            data=pd.DataFrame(
                {"load": [80.0, 90.0, 100.0]},
                index=pd.date_range(datetime.fromisoformat("2025-01-02T00:00:00"), periods=3, freq="1h"),
            ),
            sample_interval=timedelta(hours=1),
            target_column="load",
        ),
    }

    # Act
    state = original_forecaster.to_state()
    restored_forecaster = MultiHorizonForecasterAdapter.create(
        config=sample_multi_horizon_config, model_factory=model_factory
    )
    restored_forecaster = restored_forecaster.from_state(state)

    # Compare predictions on new data to verify state restoration
    original_result = original_forecaster.predict(different_data)
    restored_result = restored_forecaster.predict(different_data)

    # Assert
    assert restored_forecaster.is_fitted
    pd.testing.assert_frame_equal(original_result.data, restored_result.data)
    assert original_result.sample_interval == restored_result.sample_interval


def test_multi_horizon_forecaster__predict_not_fitted_raises_error(
    sample_multi_horizon_config: MultiHorizonForecasterConfig[HorizonForecasterConfig],
    sample_input_data: dict[LeadTime, ForecastInputDataset],
):
    """Test that predicting without fitting raises error."""
    # Arrange
    forecaster = MultiHorizonForecasterAdapter.create(config=sample_multi_horizon_config, model_factory=model_factory)

    # Act & Assert
    with pytest.raises(RuntimeError, match="Model not fitted"):
        forecaster.predict(sample_input_data)


def _create_test_forecast_dataset(
    start_time: datetime,
    num_points: int,
    sample_interval: timedelta,
    forecast_start: datetime | None = None,
    quantile_value: float = 0.5,
    constant_value: float = 100.0,
) -> ForecastDataset:
    return ForecastDataset(
        data=pd.DataFrame(
            data={Quantile(quantile_value).format(): [constant_value] * num_points},
            index=pd.date_range(start_time, periods=num_points, freq=sample_interval),
        ),
        sample_interval=sample_interval,
        forecast_start=forecast_start or start_time,
    )


@pytest.mark.parametrize(
    ("horizon_configs", "expected_values"),
    [
        pytest.param([(timedelta(hours=6), 100.0)], [100.0] * 6, id="single_forecast"),
        pytest.param(
            [(timedelta(hours=6), 100.0), (timedelta(hours=12), 200.0)], [100.0] * 6 + [200.0] * 6, id="two_horizons"
        ),
        pytest.param(
            [(timedelta(hours=3), 300.0), (timedelta(hours=6), 400.0), (timedelta(hours=9), 500.0)],
            [300.0] * 3 + [400.0] * 3 + [500.0] * 3,
            id="three_progressive_horizons",
        ),
        pytest.param(
            [(timedelta(days=1), 150.0), (timedelta(days=2), 250.0)], [150.0] * 24 + [250.0] * 24, id="day_horizons"
        ),
    ],
)
def test_combine_horizon_forecasts(horizon_configs: list[tuple[timedelta, float]], expected_values: list[float]):
    """Test combining forecasts with predictable constant values per horizon."""
    # Arrange
    forecast_start = datetime.fromisoformat("2025-01-01T10:00:00")
    sample_interval = timedelta(hours=1)
    total_points = len(expected_values)
    forecasts: dict[LeadTime, ForecastDataset] = {
        LeadTime(lead_time_delta): _create_test_forecast_dataset(
            start_time=forecast_start,
            num_points=total_points,
            sample_interval=sample_interval,
            forecast_start=forecast_start,
            constant_value=constant_value,
        )
        for lead_time_delta, constant_value in horizon_configs
    }

    # Act
    result = combine_horizon_forecasts(forecasts)

    # Assert
    assert len(result.data) == total_points
    assert result.sample_interval == sample_interval
    assert result.forecast_start == forecast_start

    # Check that the combined data matches expected values
    actual_values = result.data.iloc[:, 0].tolist()
    assert actual_values == expected_values


def test_combine_horizon_forecasts__empty_raises_error():
    """Test combining empty forecasts raises ValueError."""
    # Arrange
    forecasts: dict[LeadTime, ForecastDataset] = {}

    # Act & Assert
    with pytest.raises(ValueError, match="No forecasts to combine"):
        combine_horizon_forecasts(forecasts)
