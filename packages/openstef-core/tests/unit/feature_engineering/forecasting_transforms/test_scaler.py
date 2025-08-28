# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the Scaler transform."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.feature_engineering.forecasting_transforms.scaler import Scaler, ScalingMethod


@pytest.fixture
def sample_dataset() -> TimeSeriesDataset:
    """
    Create a sample TimeSeriesDataset for testing.

    Returns:
        TimeSeriesDataset: A dataset with realistic energy load and temperature data.
    """
    data = pd.DataFrame(
        {
            "load": [100.0, 120.0, 110.0, 130.0, 125.0],
            "temperature": [20.0, 22.0, 21.0, 23.0, 24.0],
            "wind_speed": [5.0, 8.0, 6.0, 10.0, 7.0],
        },
        index=pd.date_range("2025-01-01", periods=5, freq="1h"),
    )
    return TimeSeriesDataset(data, timedelta(hours=1))


def test_scaler_invalid_method():
    """Test Scaler raises ValueError for unsupported scaling method."""
    with pytest.raises(ValidationError):
        Scaler(method="invalid_method")


def test_standard_scaler_fit_transform(sample_dataset: TimeSeriesDataset):
    """Test StandardScaler produces zero mean and unit variance."""
    # Arrange
    scaler = Scaler(method=ScalingMethod.Standard)

    # Act
    scaler.fit(sample_dataset)
    result = scaler.transform(sample_dataset)

    # Assert
    assert isinstance(result, TimeSeriesDataset)
    assert result.sample_interval == sample_dataset.sample_interval
    assert list(result.data.columns) == list(sample_dataset.data.columns)
    assert len(result.data) == len(sample_dataset.data)

    # Check scaling properties (StandardScaler should give ~0 mean, 1 std)
    for col in result.data.columns:
        assert np.allclose(result.data[col].mean(), 0.0)
        assert np.allclose(result.data[col].std(ddof=0), 1.0)  # Population std as used by sklearn


def test_minmax_scaler_fit_transform(sample_dataset: TimeSeriesDataset):
    """Test MinMaxScaler produces values between 0 and 1."""
    # Arrange
    scaler = Scaler(method=ScalingMethod.MinMax)

    # Act
    scaler.fit(sample_dataset)
    result = scaler.transform(sample_dataset)

    # Assert
    assert isinstance(result, TimeSeriesDataset)
    assert result.sample_interval == sample_dataset.sample_interval
    assert list(result.data.columns) == list(sample_dataset.data.columns)
    assert len(result.data) == len(sample_dataset.data)

    # Check that min and max values are actually 0 and 1 for each column
    for col in result.data.columns:
        values = result.data[col]
        assert np.allclose(values.min(), 0.0)
        assert np.allclose(values.max(), 1.0)


def test_maxabs_scaler_fit_transform(sample_dataset: TimeSeriesDataset):
    """Test MaxAbsScaler produces values between -1 and 1."""
    # Arrange
    scaler = Scaler(method=ScalingMethod.MaxAbs)

    # Act
    scaler.fit(sample_dataset)
    result = scaler.transform(sample_dataset)

    # Assert
    assert isinstance(result, TimeSeriesDataset)
    assert result.sample_interval == sample_dataset.sample_interval
    assert list(result.data.columns) == list(sample_dataset.data.columns)
    assert len(result.data) == len(sample_dataset.data)

    # Check that the maximum absolute value is 1 for each column
    for col in result.data.columns:
        values = result.data[col]
        assert np.allclose(values.abs().max(), 1.0)


def test_robust_scaler_fit_transform(sample_dataset: TimeSeriesDataset):
    """Test RobustScaler uses median and IQR for scaling."""
    # Arrange
    scaler = Scaler(method=ScalingMethod.Robust)

    # Act
    scaler.fit(sample_dataset)
    result = scaler.transform(sample_dataset)

    # Assert
    assert isinstance(result, TimeSeriesDataset)
    assert result.sample_interval == sample_dataset.sample_interval
    assert list(result.data.columns) == list(sample_dataset.data.columns)

    # For RobustScaler, median should be close to 0
    for col in result.data.columns:
        median_val = result.data[col].median()
        assert np.allclose(median_val, 0.0)


def test_different_datasets_fit_transform():
    """Test fitting on one dataset and transforming another."""
    # Arrange
    # Training dataset
    train_data = pd.DataFrame(
        {
            "load": [100.0, 200.0],  # Range: 100
            "temperature": [10.0, 30.0],  # Range: 20
        },
        index=pd.date_range("2025-01-01", periods=2, freq="1h"),
    )
    train_dataset = TimeSeriesDataset(train_data, timedelta(hours=1))

    # Test dataset with different range
    test_data = pd.DataFrame(
        {
            "load": [150.0, 250.0],  # Different values
            "temperature": [20.0, 40.0],  # Different values
        },
        index=pd.date_range("2025-01-02", periods=2, freq="1h"),
    )
    test_dataset = TimeSeriesDataset(test_data, timedelta(hours=1))

    # Act
    scaler = Scaler(method=ScalingMethod.MinMax)
    scaler.fit(train_dataset)
    result = scaler.transform(test_dataset)

    # Assert
    # Values should be scaled based on training data ranges
    # load: 150 -> (150-100)/(200-100) = 0.5, 250 -> (250-100)/(200-100) = 1.5
    # temperature: 20 -> (20-10)/(30-10) = 0.5, 40 -> (40-10)/(30-10) = 1.5
    assert np.allclose(result.data["load"].iloc[0], 0.5)
    assert np.allclose(result.data["load"].iloc[1], 1.5)
    assert np.allclose(result.data["temperature"].iloc[0], 0.5)
    assert np.allclose(result.data["temperature"].iloc[1], 1.5)
