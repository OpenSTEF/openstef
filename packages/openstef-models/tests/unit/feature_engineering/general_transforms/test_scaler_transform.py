# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the ScalerTransform transform."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import TransformNotFittedError
from openstef_models.feature_engineering.general_transforms import ScalerTransform


def test_scaler_invalid_method():
    """Test ScalerTransform raises ValidationError for invalid scaling method."""
    with pytest.raises(ValidationError, match="Input should be"):
        ScalerTransform(method="invalid_method")  # pyright: ignore[reportArgumentType]


def test_scaler_not_fitted_error():
    """Test that transform raises error when not fitted first."""
    # Arrange
    data = pd.DataFrame(
        {"load": [1.0, 2.0, 3.0]},
        index=pd.date_range("2023-01-01", periods=3, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, sample_interval=timedelta(hours=1))
    scaler = ScalerTransform(method="standard")

    # Act & Assert
    with pytest.raises(TransformNotFittedError, match="ScalerTransform"):
        scaler.transform(dataset)


def test_standard_scaler_basic():
    """Test StandardScaler with simple data."""
    # Arrange
    data = pd.DataFrame(
        {"load": [0.0, 10.0, 20.0]},  # mean=10, std=10
        index=pd.date_range("2023-01-01", periods=3, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, sample_interval=timedelta(hours=1))
    scaler = ScalerTransform(method="standard")

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
    scaler = ScalerTransform(method="min-max")

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

    scaler = ScalerTransform(method="min-max")

    # Act
    scaler.fit(train_dataset)
    result = scaler.transform(test_dataset)

    # Assert
    # Scaling based on training data: (x - 0) / (100 - 0) = x / 100
    expected = [0.5, 1.5]  # 50->0.5, 150->1.5 (can exceed [0,1] for new data)
    assert np.allclose(result.data["load"].tolist(), expected)


def test_multiple_columns():
    """Test scaler works with multiple columns."""
    # Arrange
    data = pd.DataFrame(
        {
            "load": [10.0, 20.0],
            "temperature": [0.0, 10.0],
        },
        index=pd.date_range("2023-01-01", periods=2, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=1))
    scaler = ScalerTransform(method="min-max")

    # Act
    scaler.fit(dataset)
    result = scaler.transform(dataset)

    # Assert
    assert set(result.data.columns) == {"load", "temperature"}

    # Each column scaled independently
    assert np.allclose(result.data["load"].tolist(), [0.0, 1.0])  # [10,20] -> [0,1]
    assert np.allclose(result.data["temperature"].tolist(), [0.0, 1.0])  # [0,10] -> [0,1]
