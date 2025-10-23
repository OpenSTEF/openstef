# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from pandas import DatetimeIndex

from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.types import LeadTime, Quantile
from openstef_models.models.forecasting.base_case_forecaster import (
    BaseCaseForecaster,
    BaseCaseForecasterConfig,
    BaseCaseForecasterHyperParams,
)


@pytest.fixture
def sample_forecast_input_dataset() -> ForecastInputDataset:
    """Create test dataset with different patterns for week 1 and week 2 + 1 week forecast."""
    # Create 3 weeks of hourly data with distinct patterns
    dates: DatetimeIndex = pd.date_range(
        start=datetime.fromisoformat("2025-01-01T00:00:00"), periods=3 * 7 * 24, freq="1h"
    )
    forecast_start = dates[2 * 7 * 24]  # Start of 3rd week (2 weeks of history)

    load_values: list[float] = []
    for timestamp in dates:
        if timestamp < forecast_start:
            # Week 1: Pattern 200 + hour (should NOT be used for forecasting)
            if timestamp < dates[7 * 24]:
                load_values.append(200 + timestamp.hour)
            # Week 2: Pattern 300 + hour (should be used for forecasting - most recent)
            else:
                load_values.append(300 + timestamp.hour)
        else:
            # Forecast period
            load_values.append(np.nan)

    data = pd.DataFrame({"load": load_values}, index=dates)

    return ForecastInputDataset(
        data=data, sample_interval=timedelta(hours=1), target_column="load", forecast_start=forecast_start
    )


@pytest.fixture
def base_case_forecaster() -> BaseCaseForecaster:
    """Create sample forecaster configuration with standard quantiles."""
    return BaseCaseForecaster(
        config=BaseCaseForecasterConfig(
            horizons=[LeadTime(timedelta(days=1))],
            hyperparams=BaseCaseForecasterHyperParams(),
        )
    )


def test_base_case_forecaster__initialization_custom(base_case_forecaster: BaseCaseForecaster):
    """Test forecaster initialization with custom configuration."""
    # Arrange
    forecaster = base_case_forecaster

    # Assert
    assert forecaster.config == base_case_forecaster.config
    assert forecaster.hyperparams.primary_lag == timedelta(days=7)
    assert forecaster.hyperparams.fallback_lag == timedelta(days=14)


def test_base_case_forecaster__fit_predict(
    base_case_forecaster: BaseCaseForecaster,
    sample_forecast_input_dataset: ForecastInputDataset,
):
    """Test that forecaster produces predictions using weekly pattern repetition."""
    # Arrange - Fixtures provide forecaster and dataset

    # Act
    base_case_forecaster.fit(sample_forecast_input_dataset)
    result = base_case_forecaster.predict(sample_forecast_input_dataset)

    # Assert
    assert base_case_forecaster.is_fitted
    assert isinstance(result, ForecastDataset)

    pd.testing.assert_index_equal(
        result.index, sample_forecast_input_dataset.create_forecast_range(base_case_forecaster.config.max_horizon)
    )

    expected_columns = [q.format() for q in base_case_forecaster.config.quantiles]
    assert list(result.data.columns) == expected_columns

    actual_values = result.data.iloc[0]
    for col in expected_columns:
        assert not pd.isna(actual_values[col])


def test_base_case_forecaster__weekly_pattern_repetition(
    base_case_forecaster: BaseCaseForecaster,
    sample_forecast_input_dataset: ForecastInputDataset,
):
    """Test that forecaster uses the MOST RECENT week's pattern (week 2: 300s) NOT the older week (week 1: 200s)."""
    # Act
    result = base_case_forecaster.predict(sample_forecast_input_dataset)

    # Assert - Simple check: Are we getting 300s (correct) or 200s (wrong)?
    np.testing.assert_array_equal(
        actual=result.median_series.iloc[[0, 1, 23]].to_numpy(),  # type: ignore
        desired=np.array([300.0, 301.0, 323.0]),
    )


def test_base_case_forecaster__no_forecast_start(base_case_forecaster: BaseCaseForecaster):
    """Test behavior when no forecast_start is specified."""
    # Arrange
    input_dataset = ForecastInputDataset(
        data=pd.DataFrame(
            {"load": [100.0, 110.0, 120.0]},
            index=pd.date_range(start=datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1h"),
        ),
        sample_interval=timedelta(hours=1),
        target_column="load",
        forecast_start=None,
    )

    # Act
    result = base_case_forecaster.predict(input_dataset)

    # Assert
    assert len(result.data) == 25
    assert result.sample_interval == input_dataset.sample_interval


def test_base_case_forecaster__no_historical_data(base_case_forecaster: BaseCaseForecaster):
    """Test behavior when no historical data is available before forecast_start."""
    # Arrange
    data = pd.DataFrame(
        {"load": [100.0, 110.0, 120.0]},
        index=pd.date_range(start=datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1h"),
    )

    input_dataset = ForecastInputDataset(
        data=data,
        sample_interval=timedelta(hours=1),
        target_column="load",
        forecast_start=data.index[0],
    )

    # Act
    result = base_case_forecaster.predict(input_dataset)

    # Assert
    assert len(result.data) == 25
    assert all(pd.isna(result.median_series))


def test_base_case_forecaster__fallback_lag_usage():
    """Test that fallback lag is used when primary lag period has insufficient data."""
    # Arrange
    dates = pd.date_range(start=datetime.fromisoformat("2025-01-01T00:00:00"), periods=3 * 7 * 24, freq="1h")

    load_values: list[float] = []
    for i, timestamp in enumerate(dates):
        if i < 7 * 24:
            load_values.append(100.0 + timestamp.hour)
        elif i < 14 * 24:
            load_values.append(200.0 + timestamp.hour)
        else:
            load_values.append(np.nan)

    data = pd.DataFrame({"load": load_values}, index=dates)

    config = BaseCaseForecasterConfig(
        quantiles=[Quantile(0.5)],
        horizons=[LeadTime(timedelta(hours=1))],
        hyperparams=BaseCaseForecasterHyperParams(
            primary_lag=timedelta(hours=6),
            fallback_lag=timedelta(days=7),
        ),
    )

    input_dataset = ForecastInputDataset(
        data=data,
        sample_interval=timedelta(hours=1),
        target_column="load",
        forecast_start=dates[2 * 7 * 24],
    )

    forecaster = BaseCaseForecaster(config=config)

    # Act
    result = forecaster.predict(input_dataset)

    # Assert
    assert len(result.data) > 0
    assert not result.data["quantile_P50"].isna().all()
    assert forecaster.is_fitted
    assert isinstance(result, ForecastDataset)

    # Verify fallback lag is used - predictions should use Week 2 data (200.0 + hour pattern)
    median_predictions = result.data["quantile_P50"]

    # Check that we're getting values from Week 2 range (200-223) not Week 1 (100-123)
    assert all(median_predictions >= 200.0), "Should use Week 2 data (≥200)"
    assert all(median_predictions <= 223.0), "Should use Week 2 data (≤223)"


def test_base_case_forecaster__state_serialization(base_case_forecaster: BaseCaseForecaster):
    """Test state serialization and deserialization."""
    # Arrange - Fixture provides forecaster

    # Act
    state = base_case_forecaster.to_state()
    new_forecaster = BaseCaseForecaster(
        config=BaseCaseForecasterConfig(
            quantiles=[Quantile(0.5)],
            horizons=[LeadTime(timedelta(hours=6))],
            hyperparams=BaseCaseForecasterHyperParams(),
        )
    ).from_state(state)

    # Assert
    assert isinstance(state, dict)
    assert "version" in state
    assert "config" in state
    assert new_forecaster.config.quantiles == base_case_forecaster.config.quantiles
    assert new_forecaster.hyperparams.primary_lag == base_case_forecaster.hyperparams.primary_lag


def test_base_case_forecaster__different_frequencies(base_case_forecaster: BaseCaseForecaster):
    """Test forecaster with different data frequencies."""
    # Arrange
    dates = pd.date_range(start=datetime.fromisoformat("2025-01-01T00:00:00"), periods=2 * 7 * 48, freq="30min")
    load_values = [100.0 + (i % 48) for i in range(len(dates))]

    data = pd.DataFrame({"load": load_values}, index=dates)

    input_dataset = ForecastInputDataset(
        data=data,
        sample_interval=timedelta(minutes=30),
        target_column="load",
        forecast_start=dates[7 * 48],
    )

    # Act
    result = base_case_forecaster.predict(input_dataset)

    # Assert
    assert result.sample_interval == timedelta(minutes=30)
    assert len(result.data) > 0
    assert not result.median_series.isna().all()


@pytest.mark.parametrize(
    ("primary_lag", "fallback_lag"),
    [
        (timedelta(days=7), timedelta(days=14)),
        (timedelta(days=1), timedelta(days=7)),
        (timedelta(hours=24), timedelta(hours=48)),
    ],
)
def test_base_case_forecaster__different_lag_configurations(primary_lag: timedelta, fallback_lag: timedelta):
    """Test forecaster with different lag configurations."""
    # Arrange
    dates = pd.date_range(start=datetime.fromisoformat("2025-01-01T00:00:00"), periods=3 * 7 * 24, freq="1h")
    load_values = [100.0 + i % 24 for i in range(len(dates))]

    data = pd.DataFrame({"load": load_values}, index=dates)

    config = BaseCaseForecasterConfig(
        quantiles=[Quantile(0.5)],
        horizons=[LeadTime(timedelta(hours=1))],
        hyperparams=BaseCaseForecasterHyperParams(
            primary_lag=primary_lag,
            fallback_lag=fallback_lag,
        ),
    )

    input_dataset = ForecastInputDataset(
        data=data,
        sample_interval=timedelta(hours=1),
        target_column="load",
        forecast_start=dates[2 * 7 * 24],
    )

    forecaster = BaseCaseForecaster(config=config)

    # Act
    result = forecaster.predict(input_dataset)

    # Assert
    assert len(result.data) > 0
    assert not result.data["quantile_P50"].isna().all()
