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
from openstef_core.exceptions import NotFittedError
from openstef_models.transforms.general import Scaler


@pytest.fixture
def sample_dataset() -> TimeSeriesDataset:
    """Sample dataset for scaler tests."""
    data = pd.DataFrame(
        {"load": [0.0, 50.0, 100.0], "temperature": [10.0, 20.0, 30.0]},
        index=pd.date_range("2023-01-01", periods=3, freq="1h"),
    )
    return TimeSeriesDataset(data, sample_interval=timedelta(hours=1))


def test_scaler_invalid_method():
    """Test Scaler raises ValidationError for invalid scaling method."""
    with pytest.raises(ValidationError, match="Input should be"):
        Scaler(method="invalid_method")  # pyright: ignore[reportArgumentType]


def test_scaler_not_fitted_error(sample_dataset: TimeSeriesDataset):
    """Test that transform raises error when not fitted first."""
    # Arrange
    scaler = Scaler(method="standard")

    # Act & Assert
    with pytest.raises(NotFittedError, match="Scaler"):
        scaler.transform(sample_dataset)


def test_standard_scaler_basic():
    """Test StandardScaler with simple data."""
    # Arrange
    data = pd.DataFrame(
        {"load": [0.0, 10.0, 20.0]},  # mean=10, std=10
        index=pd.date_range("2023-01-01", periods=3, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, sample_interval=timedelta(hours=1))
    scaler = Scaler(method="standard")

    # Act
    scaler.fit(dataset)
    result = scaler.transform(dataset)

    # Assert
    assert result.sample_interval == dataset.sample_interval
    assert list(result.data.columns) == ["load"]
    # StandardScaler: (x - mean) / std where std uses ddof=1 (sample std)
    # mean=10, std=10, so: 0->-1.0, 10->0.0, 20->1.0 scaled by sqrt(3/2)
    expected = [-1.224744871391589, 0.0, 1.224744871391589]
    assert np.allclose(result.data["load"].tolist(), expected)


def test_minmax_scaler_basic():
    """Test MinMaxScaler with simple data."""
    # Arrange
    data = pd.DataFrame(
        {"load": [10.0, 20.0, 30.0]},  # min=10, max=30, range=20
        index=pd.date_range("2023-01-01", periods=3, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, sample_interval=timedelta(hours=1))
    scaler = Scaler(method="min-max")

    # Act
    scaler.fit(dataset)
    result = scaler.transform(dataset)

    # Assert
    # MinMaxScaler: (x - min) / (max - min) = (x - 10) / 20  # noqa: ERA001
    expected = [0.0, 0.5, 1.0]
    assert np.allclose(result.data["load"].tolist(), expected)


def test_fit_on_one_dataset_transform_another():
    """Test critical use case: fit on training data, transform test data."""
    # Arrange
    train_data = pd.DataFrame(
        {"load": [0.0, 100.0]},  # Range: 0-100
        index=pd.date_range("2023-01-01", periods=2, freq="1h"),
    )
    train_dataset = TimeSeriesDataset(train_data, timedelta(hours=1))

    test_data = pd.DataFrame(
        {"load": [50.0, 150.0]},  # Different values, 150 is outside training range
        index=pd.date_range("2023-01-02", periods=2, freq="1h"),
    )
    test_dataset = TimeSeriesDataset(test_data, timedelta(hours=1))

    scaler = Scaler(method="min-max")

    # Act
    scaler.fit(train_dataset)
    result = scaler.transform(test_dataset)

    # Assert
    # Scaling based on training data: (x - 0) / (100 - 0) = x / 100
    expected = [0.5, 1.5]  # 50->0.5, 150->1.5 (can exceed [0,1] for new data)
    assert np.allclose(result.data["load"].tolist(), expected)


def test_multiple_columns(sample_dataset: TimeSeriesDataset):
    """Test scaler works with multiple columns."""
    # Arrange
    scaler = Scaler(method="min-max")

    # Act
    scaler.fit(sample_dataset)
    result = scaler.transform(sample_dataset)

    # Assert
    assert set(result.data.columns) == {"load", "temperature"}

    # Each column scaled independently: [0,50,100] -> [0,0.5,1], [10,20,30] -> [0,0.5,1]
    assert np.allclose(result.data["load"].tolist(), [0.0, 0.5, 1.0])
    assert np.allclose(result.data["temperature"].tolist(), [0.0, 0.5, 1.0])


def test_scaler_transform__state_roundtrip(sample_dataset: TimeSeriesDataset):
    """Test scaler transform state serialization and restoration."""
    # Arrange
    original_transform = Scaler(method="standard")
    original_transform.fit(sample_dataset)

    # Act
    state = original_transform.to_state()
    restored_transform = Scaler(method="standard")
    restored_transform = restored_transform.from_state(state)

    original_result = original_transform.transform(sample_dataset)
    restored_result = restored_transform.transform(sample_dataset)

    # Assert
    pd.testing.assert_frame_equal(original_result.data, restored_result.data)
