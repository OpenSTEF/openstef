# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta

import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.types import EnergyComponentType
from openstef_models.models.component_splitting.constant_component_splitter import (
    ConstantComponentSplitter,
    ConstantComponentSplitterConfig,
)


@pytest.fixture
def sample_timeseries_dataset() -> TimeSeriesDataset:
    """Create sample time series data with typical energy load values.

    Returns:
        TimeSeriesDataset with load values spanning 4 hours for component splitting tests.
    """
    data = pd.DataFrame(
        {"load": [100.0, 200.0, 300.0, 400.0]},
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=4, freq="1h"),
    )

    return TimeSeriesDataset(data=data, sample_interval=timedelta(hours=1))


@pytest.fixture
def sample_splitter_config() -> ConstantComponentSplitterConfig:
    """Create sample component splitter configuration with solar and wind components."""
    return ConstantComponentSplitterConfig(
        source_column="load",
        component_ratios={
            EnergyComponentType.SOLAR: 0.5,
            EnergyComponentType.WIND: 0.25,
            EnergyComponentType.OTHER: 0.25,
        },
    )


def test_constant_component_splitter__config_validation_ratios_must_sum_to_one():
    """Test that configuration validation requires component ratios to sum to 1.0."""
    # Act & Assert - Ratios summing to less than 1.0 should raise ValueError
    with pytest.raises(ValueError, match=r"Component ratios must sum to 1\.0"):
        ConstantComponentSplitterConfig(
            source_column="load",
            component_ratios={
                EnergyComponentType.SOLAR: 0.5,
                EnergyComponentType.WIND: 0.3,  # Sum = 0.8, not 1.0
            },
        )

    # Ratios summing to more than 1.0 should also raise ValueError
    with pytest.raises(ValueError, match=r"Component ratios must sum to 1\.0"):
        ConstantComponentSplitterConfig(
            source_column="load",
            component_ratios={
                EnergyComponentType.SOLAR: 0.7,
                EnergyComponentType.WIND: 0.5,  # Sum = 1.2, not 1.0
            },
        )


def test_constant_component_splitter__predict_splits_correctly(
    sample_splitter_config: ConstantComponentSplitterConfig,
    sample_timeseries_dataset: TimeSeriesDataset,
):
    """Test that predict splits load according to configured ratios."""
    # Arrange
    splitter = ConstantComponentSplitter(config=sample_splitter_config)

    # Act
    result = splitter.predict(sample_timeseries_dataset)

    # Assert - Check that all components are present
    assert EnergyComponentType.SOLAR in result.data.columns
    assert EnergyComponentType.WIND in result.data.columns
    assert EnergyComponentType.OTHER in result.data.columns

    # Check that ratios are applied correctly
    original_load = sample_timeseries_dataset.data["load"]
    expected_solar = pd.Series([50.0, 100.0, 150.0, 200.0], index=original_load.index)
    expected_wind = pd.Series([25.0, 50.0, 75.0, 100.0], index=original_load.index)
    expected_other = pd.Series([25.0, 50.0, 75.0, 100.0], index=original_load.index)
    pd.testing.assert_series_equal(result.data[EnergyComponentType.SOLAR], expected_solar, check_names=False)
    pd.testing.assert_series_equal(result.data[EnergyComponentType.WIND], expected_wind, check_names=False)
    pd.testing.assert_series_equal(result.data[EnergyComponentType.OTHER], expected_other, check_names=False)

    # Verify components sum back to original (within floating point precision)
    reconstructed = (
        result.data[EnergyComponentType.SOLAR]
        + result.data[EnergyComponentType.WIND]
        + result.data[EnergyComponentType.OTHER]
    )
    pd.testing.assert_series_equal(reconstructed, original_load, check_names=False)


def test_constant_component_splitter__known_solar_park():
    """Test the known_solar_park class method creates correct configuration."""
    # Act
    splitter = ConstantComponentSplitter.known_solar_park()

    # Assert - Configuration is correct for solar park
    assert splitter.is_fitted
    assert splitter.config.source_column == "load"
    assert splitter.config.component_ratios == {EnergyComponentType.SOLAR: 1.0}

    # Test prediction
    data = pd.DataFrame(
        {"load": [100.0, 200.0, 300.0]},
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1h"),
    )
    dataset = TimeSeriesDataset(data=data, sample_interval=timedelta(hours=1))
    result = splitter.predict(dataset)

    # Solar component should equal load
    pd.testing.assert_series_equal(result.data[EnergyComponentType.SOLAR], dataset.data["load"], check_names=False)


def test_constant_component_splitter__known_wind_farm():
    """Test the known_wind_farm class method creates correct configuration."""
    # Act
    splitter = ConstantComponentSplitter.known_wind_farm()

    # Assert - Configuration is correct for wind farm
    assert splitter.is_fitted
    assert splitter.config.source_column == "load"
    assert splitter.config.component_ratios == {EnergyComponentType.WIND: 1.0}

    # Test prediction
    data = pd.DataFrame(
        {"load": [100.0, 200.0, 300.0]},
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1h"),
    )
    dataset = TimeSeriesDataset(data=data, sample_interval=timedelta(hours=1))
    result = splitter.predict(dataset)

    # Wind component should equal load
    pd.testing.assert_series_equal(result.data[EnergyComponentType.WIND], dataset.data["load"], check_names=False)
