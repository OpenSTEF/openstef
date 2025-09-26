# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.types import LeadTime, Quantile
from openstef_models.models.forecasting.base_case_forecaster import (
    BaseCaseForecaster,
    BaseCaseForecasterConfig,
    BaseCaseForecasterHyperParams,
)
from openstef_models.models.forecasting.multi_horizon_forecaster_adapter import MultiHorizonForecasterConfig


@pytest.fixture
def sample_forecast_input_dataset() -> ForecastInputDataset:
    # Create 14 days of quarter-hourly data
    num_samples = 14 * 24 * 4
    dates = pd.date_range(
        start=datetime.fromisoformat("2025-01-01T00:00:00"),
        periods=num_samples,
        freq="15min"
    )

    # 1's for 1st week and 2 for 2nd week
    data = pd.DataFrame(
        {"load": [1] * (num_samples // 2) + [2] * (len(dates) // 2)},
        index=dates
    )

    return ForecastInputDataset(
        data=data,
        sample_interval=timedelta(minutes=15),
        target_column="load",
        forecast_start=datetime.fromisoformat("2025-01-08T00:00:00"),  # Middle of dataset
    )


@pytest.fixture
def sample_forecaster_config() -> BaseCaseForecasterConfig:
    """Create sample forecaster configuration with standard quantiles."""
    return BaseCaseForecasterConfig(
        quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
        horizons=[LeadTime(timedelta(hours=6))],
        hyperparams=BaseCaseForecasterHyperParams(lookback_days=7),
    )


def test_base_case_forecaster__fit_predict(
    sample_forecaster_config: BaseCaseForecasterConfig,
    sample_forecast_input_dataset: ForecastInputDataset,
):
    """Test that forecaster trains on data and produces predictions from 1 week ago."""
    # Arrange
    forecaster = BaseCaseForecaster(config=sample_forecaster_config)

    # Act
    forecaster.fit(sample_forecast_input_dataset)
    result = forecaster.predict(sample_forecast_input_dataset)

    # Assert
    # Check that model is fitted and produces forecast
    assert forecaster.is_fitted
    assert isinstance(result, ForecastDataset)

    # Should have predictions for all hours after forecast_start
    forecast_start = datetime.fromisoformat("2025-01-08T00:00:00")
    expected_hours = len([t for t in sample_forecast_input_dataset.index if t > forecast_start])
    assert len(result.data) == expected_hours

    # Check that predictions match values from exactly 1 week ago
    first_prediction_time = result.data.index[0]  # First forecast timestamp
    one_week_ago = first_prediction_time - timedelta(weeks=1)

    # Get the historical value from 1 week ago
    historical_value = sample_forecast_input_dataset.data.loc[one_week_ago, "load"]

    # P50 (median) should match the historical value exactly
    actual_values = result.data.iloc[0]  # First forecast row
    assert actual_values["quantile_P50"] == historical_value

    # P10 and P90 should be different from P50 due to confidence intervals
    assert actual_values["quantile_P10"] != actual_values["quantile_P50"]
    assert actual_values["quantile_P90"] != actual_values["quantile_P50"]

    # P10 should be lower than P50, P90 should be higher than P50
    assert actual_values["quantile_P10"] < actual_values["quantile_P50"]
    assert actual_values["quantile_P90"] > actual_values["quantile_P50"]


def test_base_case_forecaster__predict_without_historical_data():
    """Test that forecaster handles missing historical data gracefully."""
    # Arrange
    # Create data with only 3 days (not enough for 1-week lookback)
    short_data = pd.DataFrame(
        {"load": [100.0, 110.0, 120.0]},
        index=pd.date_range(
            start=datetime.fromisoformat("2025-01-01T00:00:00"),
            periods=3,
            freq="1h"
        )
    )

    input_dataset = ForecastInputDataset(
        data=short_data,
        sample_interval=timedelta(hours=1),
        target_column="load",
        forecast_start=datetime.fromisoformat("2025-01-01T01:00:00"),
    )

    config = BaseCaseForecasterConfig(
        quantiles=[Quantile(0.5)],
        horizons=[LeadTime(timedelta(hours=1))],
        hyperparams=BaseCaseForecasterHyperParams(lookback_days=168),  # 7 days * 24 hours
    )
    forecaster = BaseCaseForecaster(config=config)
    forecaster.fit(input_dataset)

    # Act
    result = forecaster.predict(input_dataset)

    # Assert
    # Should produce predictions but with NaN values (no historical data available)
    # Data has 3 hours starting at 00:00, forecast_start at 01:00, so 2 hours after: 02:00, 03:00
    # But actually data only goes to 02:00 (3 periods starting at 00:00: 00:00, 01:00, 02:00)
    assert len(result.data) == 1  # 1 hour after forecast_start (only 02:00 available)
    assert pd.isna(result.data.iloc[0]["quantile_P50"])  # Should be NaN


def test_base_case_forecaster__predict_not_fitted_raises_error(
    sample_forecaster_config: BaseCaseForecasterConfig,
):
    """Test that predicting without fitting raises NotFittedError."""
    # Arrange
    forecaster = BaseCaseForecaster(config=sample_forecaster_config)
    dummy_data = pd.DataFrame(
        {"load": [100.0]},
        index=pd.date_range(
            start=datetime.fromisoformat("2025-01-01T00:00:00"),
            periods=1,
            freq="1h"
        )
    )
    input_dataset = ForecastInputDataset(
        data=dummy_data,
        sample_interval=timedelta(hours=1),
        target_column="load"
    )

    # Act & Assert
    with pytest.raises(NotFittedError, match="BaseCaseForecaster"):
        forecaster.predict(input_dataset)


def test_base_case_forecaster__state_serialize_restore(
    sample_forecaster_config: BaseCaseForecasterConfig,
    sample_forecast_input_dataset: ForecastInputDataset,
):
    """Test that forecaster state can be serialized and restored with preserved functionality."""
    # Arrange
    original_forecaster = BaseCaseForecaster(config=sample_forecaster_config)
    original_forecaster.fit(sample_forecast_input_dataset)

    # Act
    # Serialize state and create new forecaster from state
    state = original_forecaster.to_state()

    restored_forecaster = BaseCaseForecaster(config=sample_forecaster_config)
    restored_forecaster = restored_forecaster.from_state(state)

    # Assert
    # Check that restored forecaster produces identical predictions
    assert restored_forecaster.is_fitted
    original_result = original_forecaster.predict(sample_forecast_input_dataset)
    restored_result = restored_forecaster.predict(sample_forecast_input_dataset)

    pd.testing.assert_frame_equal(original_result.data, restored_result.data)
    assert original_result.sample_interval == restored_result.sample_interval


def test_base_case_forecaster__config_properties(sample_forecaster_config: BaseCaseForecasterConfig):
    """Test that forecaster correctly exposes configuration properties."""
    # Arrange
    forecaster = BaseCaseForecaster(config=sample_forecaster_config)

    # Assert
    assert forecaster.config == sample_forecaster_config
    assert forecaster.hyperparams == sample_forecaster_config.hyperparams
    assert forecaster.hyperparams.lookback_days == 168  # 7 days * 24 hours


def test_base_case_forecaster__different_lookback_hours():
    """Test that forecaster respects different lookback_hours hyperparameter."""
    # Arrange
    # Create 4 weeks of data
    weeks = 4
    hours = weeks * 7 * 24
    dates = pd.date_range(
        start=datetime.fromisoformat("2025-01-01T00:00:00"),
        periods=hours,
        freq="1h"
    )

    # Create data with predictable values based on week number
    load_values = []
    for timestamp in dates:
        week_number = (timestamp - dates[0]).days // 7 + 1
        load_values.append(week_number * 100.0)  # Week 1: 100, Week 2: 200, etc.

    data = pd.DataFrame({"load": load_values}, index=dates)
    input_dataset = ForecastInputDataset(
        data=data,
        sample_interval=timedelta(hours=1),
        target_column="load",
        forecast_start=datetime.fromisoformat("2025-01-22T00:00:00"),  # Start of week 4
    )

    # Test with 14 days (336 hours) lookback
    config_14days = BaseCaseForecasterConfig(
        quantiles=[Quantile(0.5)],
        horizons=[LeadTime(timedelta(hours=1))],
        hyperparams=BaseCaseForecasterHyperParams(lookback_days=336),  # 14 days * 24 hours
    )

    forecaster = BaseCaseForecaster(config=config_14days)
    forecaster.fit(input_dataset)

    # Act
    result = forecaster.predict(input_dataset)

    # Assert
    # Should predict values from 14 days ago (week 2 values = 200.0)
    first_prediction = result.data.iloc[0]["quantile_P50"]
    assert first_prediction == 200.0  # Week 2 value


@pytest.fixture
def multi_horizon_config() -> MultiHorizonForecasterConfig[BaseCaseForecasterConfig]:
    """Create multi-horizon forecaster configuration for testing."""
    return MultiHorizonForecasterConfig[BaseCaseForecasterConfig](
        horizons=[LeadTime(timedelta(hours=3)), LeadTime(timedelta(hours=6))],
        quantiles=[Quantile(0.5)],
        forecaster_config=BaseCaseForecasterConfig(
            quantiles=[Quantile(0.5)],
            horizons=[LeadTime(timedelta(hours=1))],  # Will be overridden
            hyperparams=BaseCaseForecasterHyperParams(lookback_days=168),  # 7 days * 24 hours
        ),
    )


@pytest.fixture
def multi_horizon_input_data() -> dict[LeadTime, ForecastInputDataset]:
    """Create horizon-specific input data for multi-horizon testing."""
    # Create 2 weeks of hourly data
    hours = 14 * 24
    dates = pd.date_range(
        start=datetime.fromisoformat("2025-01-01T00:00:00"),
        periods=hours,
        freq="1h"
    )

    # Create different load patterns for different horizons
    base_load = [100.0 + i for i in range(hours)]  # Increasing trend

    horizon_3h_data = pd.DataFrame(
        {"load": [val * 1.1 for val in base_load]},  # 10% higher
        index=dates
    )

    horizon_6h_data = pd.DataFrame(
        {"load": [val * 1.2 for val in base_load]},  # 20% higher
        index=dates
    )

    return {
        LeadTime(timedelta(hours=3)): ForecastInputDataset(
            data=horizon_3h_data,
            sample_interval=timedelta(hours=1),
            target_column="load",
            forecast_start=datetime.fromisoformat("2025-01-08T00:00:00"),
        ),
        LeadTime(timedelta(hours=6)): ForecastInputDataset(
            data=horizon_6h_data,
            sample_interval=timedelta(hours=1),
            target_column="load",
            forecast_start=datetime.fromisoformat("2025-01-08T00:00:00"),
        ),
    }


def test_base_case_forecaster__minimal_config():
    """Test that forecaster works with minimal configuration."""
    # Arrange
    config = BaseCaseForecasterConfig(
        quantiles=[Quantile(0.5)],
        horizons=[LeadTime(timedelta(hours=1))],
    )
    forecaster = BaseCaseForecaster(config=config)

    # Create minimal test data
    data = pd.DataFrame(
        {"load": [100.0, 110.0]},
        index=pd.date_range(
            start=datetime.fromisoformat("2025-01-01T00:00:00"),
            periods=2,
            freq="1h"
        )
    )
    input_dataset = ForecastInputDataset(
        data=data,
        sample_interval=timedelta(hours=1),
        target_column="load"
    )

    # Act
    forecaster.fit(input_dataset)

    # Assert
    assert forecaster.is_fitted
    assert forecaster.config.hyperparams.lookback_days == 168  # Default value (7 days * 24 hours)
    assert len(forecaster.config.quantiles) == 1  # Single quantile
    assert forecaster.config.quantiles[0] == Quantile(0.5)  # Median
