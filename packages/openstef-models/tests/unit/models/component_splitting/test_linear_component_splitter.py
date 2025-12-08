# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta

import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.types import EnergyComponentType
from openstef_models.models.component_splitting.linear_component_splitter import (
    LinearComponentSplitter,
    LinearComponentSplitterConfig,
)


@pytest.fixture
def sample_timeseries_dataset() -> TimeSeriesDataset:
    """Create sample time series data with typical energy load values and weather features.

    Returns:
        TimeSeriesDataset with load, radiation, and windspeed values spanning 24 hours.
    """
    data = pd.DataFrame(
        {
            "load": [100.0, 150.0, 200.0, 250.0, 300.0],
            "radiation": [0.0, 0.0, 10.0, 50.0, 100.0],
            "windspeed_100m": [10.0, 5.0, 0.0, 3.0, 7.0],
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=5, freq="1h"),
    )

    return TimeSeriesDataset(data=data, sample_interval=timedelta(hours=1))


@pytest.fixture
def sample_splitter_config() -> LinearComponentSplitterConfig:
    """Create sample linear component splitter configuration."""
    return LinearComponentSplitterConfig(
        source_column="load",
        radiation_column="radiation",
        windspeed_100m_column="windspeed_100m",
    )


def test_linear_component_splitter__predict_returns_correct_components(
    sample_splitter_config: LinearComponentSplitterConfig,
    sample_timeseries_dataset: TimeSeriesDataset,
):
    """Test that predict returns wind, solar, and other components."""
    # Arrange
    splitter = LinearComponentSplitter(config=sample_splitter_config)

    # Act
    result = splitter.predict(sample_timeseries_dataset)

    # Assert - Check that all components are present
    assert EnergyComponentType.SOLAR in result.data.columns
    assert EnergyComponentType.WIND in result.data.columns
    assert EnergyComponentType.OTHER in result.data.columns

    # Check that result has same index as input
    pd.testing.assert_index_equal(result.data.index, sample_timeseries_dataset.data.index)

    # Check that components are negative
    assert (result.data[EnergyComponentType.SOLAR] <= 0).all()
    assert (result.data[EnergyComponentType.WIND] <= 0).all()


def test_linear_component_splitter__create_input_features(
    sample_splitter_config: LinearComponentSplitterConfig,
    sample_timeseries_dataset: TimeSeriesDataset,
):
    """Test that input features are created correctly."""
    # Arrange
    splitter = LinearComponentSplitter(config=sample_splitter_config)

    # Act
    input_features = splitter._create_input_features(sample_timeseries_dataset)

    # Assert - Check expected columns exist
    expected_columns = ["radiation", "windspeed_100m", "total_load"]
    assert list(input_features.columns) == expected_columns

    # Check that values match input data
    assert (input_features["radiation"] == sample_timeseries_dataset.data["radiation"]).all()
    assert (input_features["windspeed_100m"] == sample_timeseries_dataset.data["windspeed_100m"]).all()
    assert (input_features["total_load"] == sample_timeseries_dataset.data["load"]).all()
