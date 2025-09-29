# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import MissingColumnsError
from openstef_core.types import LeadTime, Quantile
from openstef_models.models.forecasting.base_case_forecaster import (
    BaseCaseForecaster,
    BaseCaseForecasterConfig,
    BaseCaseForecasterHyperParams,
)


@pytest.fixture
def sample_forecast_input_dataset() -> ForecastInputDataset:
    # Create 14 days of quarter-hourly data
    num_samples = 14 * 24 * 4
    dates = pd.date_range(start=datetime.fromisoformat("2025-01-01T00:00:00"), periods=num_samples, freq="15min")

    # Create base load data with hourly variation: base + hour-based pattern
    base_load = []
    for i, timestamp in enumerate(dates):
        # Week pattern: week 1 = 100-110, week 2 = 200-210 based on hour
        week = 1 if i < (num_samples // 2) else 2
        hour_variation = (timestamp.hour % 12) * 2  # 0-22 variation
        base_load.append(week * 100 + hour_variation)

    # Create lag features simulating LagTransform output
    # 7-day lag (168 hours = P7D): values from 7 days ago
    lag_7d = [np.nan] * (7 * 24 * 4) + base_load[: -7 * 24 * 4]

    # 14-day lag (336 hours = P14D): values from 14 days ago with some variation
    lag_14d_base = [np.nan] * (14 * 24 * 4) + base_load[: -14 * 24 * 4]
    # Add some variation to enable std calculation
    lag_14d = [val + (i % 5 - 2) if not pd.isna(val) else val for i, val in enumerate(lag_14d_base)]

    data = pd.DataFrame(
        {
            "load": base_load,
            "load_lag_-P7D": lag_7d,  # 7-day lag feature
            "load_lag_-P14D": lag_14d,  # 14-day lag feature
        },
        index=dates,
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
        hyperparams=BaseCaseForecasterHyperParams(),  # Auto-detect lag columns
    )


def test_base_case_forecaster__fit_predict(
    sample_forecaster_config: BaseCaseForecasterConfig,
    sample_forecast_input_dataset: ForecastInputDataset,
):
    """Test that forecaster trains on data and produces predictions using lag features."""
    # Arrange
    forecaster = BaseCaseForecaster(config=sample_forecaster_config)

    # Act
    forecaster.fit(sample_forecast_input_dataset)
    result = forecaster.predict(sample_forecast_input_dataset)

    # Assert
    # Check that model is fitted and produces forecast
    assert forecaster.is_fitted
    assert isinstance(result, ForecastDataset)

    # Should have predictions for all periods from forecast_start onward
    forecast_start = datetime.fromisoformat("2025-01-08T00:00:00")
    expected_periods = len([t for t in sample_forecast_input_dataset.index if t >= forecast_start])
    assert len(result.data) == expected_periods

    # Check that predictions use lag features correctly
    # The forecaster should have detected the 7-day lag column
    assert forecaster._primary_lag_target_column_name == "load_lag_-P7D"
    assert forecaster._fallback_lag_target_column_name == "load_lag_-P14D"

    # Check first prediction matches 7-day lag value
    first_prediction_time = result.data.index[0]
    expected_value = sample_forecast_input_dataset.data.loc[first_prediction_time, "load_lag_-P7D"]

    # P50 (median) should match the lag value exactly (no std adjustment for median)
    actual_median = result.data.iloc[0]["quantile_P50"]
    assert actual_median == expected_value

    # Check that all quantiles are present in results
    actual_values = result.data.iloc[0]
    assert "quantile_P10" in actual_values
    assert "quantile_P50" in actual_values
    assert "quantile_P90" in actual_values

    # All quantile values should be finite numbers
    assert not pd.isna(actual_values["quantile_P10"])
    assert not pd.isna(actual_values["quantile_P50"])
    assert not pd.isna(actual_values["quantile_P90"])


def test_base_case_forecaster__predict_without_lag_columns():
    """Test that forecaster handles missing lag columns gracefully."""
    # Arrange
    # Create data with NO lag columns
    data = pd.DataFrame(
        {"load": [100.0, 110.0, 120.0]},
        index=pd.date_range(start=datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1h"),
    )

    input_dataset = ForecastInputDataset(
        data=data,
        sample_interval=timedelta(hours=1),
        target_column="load",
        forecast_start=datetime.fromisoformat("2025-01-01T01:00:00"),
    )

    config = BaseCaseForecasterConfig(
        quantiles=[Quantile(0.5)],
        horizons=[LeadTime(timedelta(hours=1))],
    )
    forecaster = BaseCaseForecaster(config=config)

    # Act & Assert
    with pytest.raises(MissingColumnsError):
        forecaster.fit_predict(input_dataset)


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
    assert not restored_forecaster.is_fitted
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
    # Check that hyperparameters use timedelta values
    assert forecaster.hyperparams.primary_lag == timedelta(days=7)
    assert forecaster.hyperparams.fallback_lag == timedelta(days=14)


def test_base_case_forecaster__explicit_lag_column_configuration():
    """Test that forecaster respects explicit lag column configuration."""
    # Arrange
    # Create data with custom lag column names
    dates = pd.date_range(
        start=datetime.fromisoformat("2025-01-01T00:00:00"),
        periods=24,  # 1 day
        freq="1h",
    )

    data = pd.DataFrame(
        {
            "load": list(range(24)),  # 0, 1, 2, ..., 23
            "load_lag_-P7D": [100.0] * 12 + [200.0] * 12,  # 7-day lag column
            "load_lag_-P14D": [300.0] * 24,  # 14-day lag column
        },
        index=dates,
    )

    input_dataset = ForecastInputDataset(
        data=data,
        sample_interval=timedelta(hours=1),
        target_column="load",
        forecast_start=datetime.fromisoformat("2025-01-01T12:00:00"),  # Middle of day
    )

    # NOTE: The refactored version auto-detects column names from target + lag timedelta
    # We need to create data that matches the expected lag column naming pattern
    config = BaseCaseForecasterConfig(
        quantiles=[Quantile(0.5)],
        horizons=[LeadTime(timedelta(hours=1))],
    )

    forecaster = BaseCaseForecaster(config=config)
    forecaster.fit(input_dataset)

    # Act
    result = forecaster.predict(input_dataset)

    # Assert
    # Should have detected the lag columns based on target + timedelta
    assert forecaster._primary_lag_target_column_name == "load_lag_-P7D"
    assert forecaster._fallback_lag_target_column_name == "load_lag_-P14D"

    # Should predict values from the 7d lag column
    first_prediction = result.data.iloc[0]["quantile_P50"]
    assert first_prediction == 200.0  # Second half of custom_7d_lag


def test_base_case_forecaster__fallback_to_14d_lag():
    """Test that forecaster falls back to 14-day lag when 7-day lag has gaps."""
    # Arrange
    dates = pd.date_range(start=datetime.fromisoformat("2025-01-01T00:00:00"), periods=24, freq="1h")

    # Create data where 7-day lag has NaN gaps, but 14-day lag is complete
    data = pd.DataFrame(
        {
            "load": list(range(24)),
            "load_lag_-P7D": [np.nan] * 12 + [100.0] * 12,  # Has gaps in first half
            "load_lag_-P14D": [200.0] * 24,  # Complete data
        },
        index=dates,
    )

    input_dataset = ForecastInputDataset(
        data=data,
        sample_interval=timedelta(hours=1),
        target_column="load",
        forecast_start=datetime.fromisoformat("2025-01-01T06:00:00"),  # In the gap area
    )

    forecaster = BaseCaseForecaster(
        BaseCaseForecasterConfig(
            quantiles=[Quantile(0.5)],
            horizons=[LeadTime(timedelta(hours=1))],
        )
    )
    forecaster.fit(input_dataset)

    # Act
    result = forecaster.predict(input_dataset)

    # Assert
    # Should detect both lag columns
    assert forecaster._primary_lag_target_column_name == "load_lag_-P7D"
    assert forecaster._fallback_lag_target_column_name == "load_lag_-P14D"

    # In gap area, should use 14-day lag as fallback
    first_prediction = result.data.iloc[0]["quantile_P50"]
    assert first_prediction == 200.0  # Fallback value

    # Later predictions should use 7-day lag where available
    later_prediction = result.data.iloc[6]["quantile_P50"]  # At 12:00, 7d lag available
    assert later_prediction == 100.0  # Primary lag value


def test_base_case_forecaster__minimal_config():
    """Test that forecaster works with minimal configuration."""
    # Arrange
    config = BaseCaseForecasterConfig(
        quantiles=[Quantile(0.5)],
        horizons=[LeadTime(timedelta(hours=1))],
    )
    forecaster = BaseCaseForecaster(config=config)

    # Create minimal test data with lag columns
    data = pd.DataFrame(
        {
            "load": [100.0, 110.0],
            "load_lag_-P7D": [90.0, 95.0],  # Primary lag data
            "load_lag_-P14D": [80.0, 85.0],  # Fallback lag data
        },
        index=pd.date_range(start=datetime.fromisoformat("2025-01-01T00:00:00"), periods=2, freq="1h"),
    )

    input_dataset = ForecastInputDataset(data=data, sample_interval=timedelta(hours=1), target_column="load")

    # Act
    forecaster.fit(input_dataset)

    # Assert
    assert forecaster.is_fitted
    # Check that hyperparameters use default timedelta values
    assert forecaster.hyperparams.primary_lag == timedelta(days=7)
    assert forecaster.hyperparams.fallback_lag == timedelta(days=14)
    assert len(forecaster.config.quantiles) == 1  # Single quantile
    assert forecaster.config.quantiles[0] == Quantile(0.5)  # Median
