# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import MissingColumnsError
from openstef_core.feature_engineering.forecasting_transforms.rolling_aggregate_transform import (
    RollingAggregateTransform,
)


def test_rolling_aggregate_features_basic():
    """Test basic rolling aggregation with simple data."""
    # Arrange
    data = pd.DataFrame(
        {"load": [1.0, 2.0, 3.0, 4.0, 5.0]},
        index=pd.date_range("2023-01-01 00:00:00", periods=5, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, sample_interval=timedelta(hours=1))

    transform = RollingAggregateTransform(
        columns=["load"],
        rolling_window_size=timedelta(hours=2),  # 2-hour window
        aggregation_functions=["mean", "max", "min"],
    )

    # Act
    result = transform.transform(dataset)

    # Assert
    assert result.sample_interval == dataset.sample_interval

    # Check columns exist
    assert "rolling_mean_load_PT2H" in result.data.columns
    assert "rolling_max_load_PT2H" in result.data.columns
    assert "rolling_min_load_PT2H" in result.data.columns

    # Check values (2-hour rolling window)
    # Index 0: [1] -> mean=1, max=1, min=1
    # Index 1: [1,2] -> mean=1.5, max=2, min=1
    # Index 2: [2,3] -> mean=2.5, max=3, min=2
    # Index 3: [3,4] -> mean=3.5, max=4, min=3
    # Index 4: [4,5] -> mean=4.5, max=5, min=4
    expected_mean = [1.0, 1.5, 2.5, 3.5, 4.5]
    expected_max = [1.0, 2.0, 3.0, 4.0, 5.0]
    expected_min = [1.0, 1.0, 2.0, 3.0, 4.0]

    assert result.data["rolling_mean_load_PT2H"].tolist() == expected_mean
    assert result.data["rolling_max_load_PT2H"].tolist() == expected_max
    assert result.data["rolling_min_load_PT2H"].tolist() == expected_min


def test_rolling_aggregate_features_with_nan():
    """Test rolling aggregation handles NaN values correctly."""
    # Arrange
    data = pd.DataFrame(
        {"load": [1.0, np.nan, 3.0, 4.0]},
        index=pd.date_range("2023-01-01 00:00:00", periods=4, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, sample_interval=timedelta(hours=1))

    transform = RollingAggregateTransform(
        columns=["load"],
        rolling_window_size=timedelta(hours=2),
        aggregation_functions=["mean"],
    )

    # Act
    result = transform.transform(dataset)

    # Assert
    # Index 0: [1] -> mean=1.0
    # Index 1: [1, NaN] -> mean=1.0 (pandas ignores NaN)
    # Index 2: [NaN, 3] -> mean=3.0 (pandas ignores NaN)
    # Index 3: [3, 4] -> mean=3.5
    expected_mean = [1.0, 1.0, 3.0, 3.5]

    assert result.data["rolling_mean_load_PT2H"].tolist() == expected_mean


def test_rolling_aggregate_features_multiple_columns():
    """Test rolling aggregation on multiple columns."""
    # Arrange
    data = pd.DataFrame(
        {
            "load": [10.0, 20.0, 30.0],
            "temperature": [1.0, 2.0, 3.0],
        },
        index=pd.date_range("2023-01-01 00:00:00", periods=3, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, sample_interval=timedelta(hours=1))

    transform = RollingAggregateTransform(
        columns=["load", "temperature"],
        rolling_window_size=timedelta(hours=2),
        aggregation_functions=["mean"],
    )

    # Act
    result = transform.transform(dataset)

    # Assert
    assert "rolling_mean_load_PT2H" in result.data.columns
    assert "rolling_mean_temperature_PT2H" in result.data.columns

    # Original columns should still be present
    assert "load" in result.data.columns
    assert "temperature" in result.data.columns

    # Check values
    assert result.data["rolling_mean_load_PT2H"].tolist() == [10.0, 15.0, 25.0]
    assert result.data["rolling_mean_temperature_PT2H"].tolist() == [1.0, 1.5, 2.5]


def test_rolling_aggregate_features_missing_column_raises_error():
    """Test that transform raises error when required column is missing."""
    # Arrange
    data = pd.DataFrame(
        {"not_load": [1.0, 2.0, 3.0]},
        index=pd.date_range("2023-01-01 00:00:00", periods=3, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, sample_interval=timedelta(minutes=15))
    transform = RollingAggregateTransform(columns=["load"])

    # Act & Assert
    with pytest.raises(MissingColumnsError, match="Missing required columns"):
        transform.transform(dataset)


def test_rolling_aggregate_features_default_parameters():
    """Test transform works with default parameters."""
    # Arrange
    data = pd.DataFrame(
        {"load": [1.0, 2.0, 3.0]},
        index=pd.date_range("2023-01-01 00:00:00", periods=3, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, sample_interval=timedelta(hours=1))

    transform = RollingAggregateTransform(columns=["load"])

    # Act
    result = transform.transform(dataset)

    # Assert - default is 24-hour window with median, min, max
    assert "rolling_median_load_P1D" in result.data.columns
    assert "rolling_min_load_P1D" in result.data.columns
    assert "rolling_max_load_P1D" in result.data.columns
