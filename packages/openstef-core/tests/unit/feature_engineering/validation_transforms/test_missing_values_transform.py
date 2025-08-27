# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from _pytest.logging import LogCaptureFixture
from pydantic import ValidationError
from sklearn.impute import SimpleImputer

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.feature_engineering.validation_transforms.missing_values_transform import (
    ImputationStrategy,
    MissingValuesTransform,
)


@pytest.fixture
def sample_dataset() -> TimeSeriesDataset:
    """Create a sample dataset with missing values.

    Returns:
        A sample TimeSeriesDataset containing missing values for testing.
    """
    data = pd.DataFrame(
        {
            "radiation": [100.0, np.nan, 110.0, 120.0],
            "temperature": [20.0, 21.0, np.nan, 23.0],
            "wind_speed": [5.0, 6.0, 8.0, np.nan],
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=4, freq="1h"),
    )
    return TimeSeriesDataset(data, timedelta(hours=1))


def test_initialization_with_required_parameters():
    """Test normal initialization with required parameters."""
    # Act
    transform = MissingValuesTransform(imputation_strategy=ImputationStrategy.MEAN)

    # Assert
    assert transform.imputation_strategy == ImputationStrategy.MEAN
    assert np.isnan(transform.missing_value)
    assert transform.fill_value is None
    assert transform.no_fill_future_values_features == []
    assert isinstance(transform._imputer, SimpleImputer)


def test_validation_constant_strategy_without_fill_value():
    """Test that CONSTANT strategy without fill_value raises ValidationError."""

    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        MissingValuesTransform(imputation_strategy=ImputationStrategy.CONSTANT)

    assert "fill_value must be provided when imputation_strategy is CONSTANT" in str(exc_info.value)


@pytest.mark.parametrize(
    ("strategy", "expected_radiation_value", "expected_temperature_value", "expected_wind_speed_value"),
    [
        pytest.param(ImputationStrategy.MEAN, 110.0, 21.33, 6.33, id="mean_imputation"),
        pytest.param(ImputationStrategy.MEDIAN, 110.0, 21.0, 6.0, id="median_imputation"),
    ],
)
def test_basic_imputation_strategies(
    sample_dataset: TimeSeriesDataset,
    strategy: ImputationStrategy,
    expected_radiation_value: float,
    expected_temperature_value: float,
    expected_wind_speed_value: float,
):
    # Arrange
    transform = MissingValuesTransform(imputation_strategy=strategy)

    # Act
    transform.fit(sample_dataset)
    result = transform.transform(sample_dataset)

    # Assert
    assert not result.data.isna().any().any()
    assert result.data.loc[result.data.index[1], "radiation"] == expected_radiation_value
    assert abs(result.data.loc[result.data.index[2], "temperature"] - expected_temperature_value) < 0.01  # type: ignore[arg-type]
    assert abs(result.data.loc[result.data.index[3], "wind_speed"] - expected_wind_speed_value) < 0.01  # type: ignore[arg-type]


def test_constant_imputation(sample_dataset: TimeSeriesDataset):
    # Arrange
    transform = MissingValuesTransform(imputation_strategy=ImputationStrategy.CONSTANT, fill_value=999.0)

    # Act
    transform.fit(sample_dataset)
    result = transform.transform(sample_dataset)

    # Assert
    assert not result.data.isna().any().any()
    assert result.data.loc[result.data.index[1], "radiation"] == 999.0
    assert result.data.loc[result.data.index[2], "temperature"] == 999.0
    assert result.data.loc[result.data.index[3], "wind_speed"] == 999.0


def test_remove_trailing_null_rows():
    # Arrange
    data = pd.DataFrame(
        {
            "radiation": [100.0, 110.0, np.nan, np.nan],
            "temperature": [20.0, 21.0, np.nan, 23.0],
            "wind_speed": [5.0, 6.0, 8.0, np.nan],
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=4, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=1))
    transform = MissingValuesTransform(
        imputation_strategy=ImputationStrategy.MEAN, no_fill_future_values_features=["radiation"]
    )

    # Act
    transform.fit(dataset)
    result = transform.transform(dataset)

    # Assert
    assert len(result.data) == 2
    assert not result.data["radiation"].isna().any()
    assert result.data["radiation"].tolist() == [100.0, 110.0]


def test_remove_trailing_nulls_multiple_features():
    # Arrange
    data = pd.DataFrame(
        {
            "radiation": [100.0, 110.0, np.nan],
            "temperature": [20.0, 21.0, np.nan],
            "wind_speed": [5.0, 6.0, 8.0],
            "price": [10.0, np.nan, np.nan],
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=1))
    transform = MissingValuesTransform(
        imputation_strategy=ImputationStrategy.MEAN, no_fill_future_values_features=["radiation", "price"]
    )

    # Act
    transform.fit(dataset)
    result = transform.transform(dataset)

    # Assert
    assert len(result.data) == 1  # Only first row remains
    assert result.data["radiation"].iloc[0] == 100.0
    assert result.data["price"].iloc[0] == 10.0


def test_no_trailing_nulls_removal_when_feature_not_in_data(
    sample_dataset: TimeSeriesDataset, caplog: LogCaptureFixture
):
    # Arrange
    transform = MissingValuesTransform(
        imputation_strategy=ImputationStrategy.MEAN, no_fill_future_values_features=["nonexistent_feature"]
    )

    # Act
    with caplog.at_level(logging.WARNING):
        transform.fit(sample_dataset)
        result = transform.transform(sample_dataset)

    # Assert
    assert len(result.data) == 4  # No rows removed
    assert not result.data.isna().any().any()  # All values imputed
    assert "Features 'nonexistent_feature' not found in dataset columns." in caplog.text


def test_empty_feature_removal():
    # Arrange
    data = pd.DataFrame(
        {
            "radiation": [100.0, np.nan, 110.0],
            "temperature": [20.0, 21.0, np.nan],
            "wind_speed": [5.0, 6.0, 8.0],
            "empty_feature": [np.nan, np.nan, np.nan],
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=1))
    transform = MissingValuesTransform(imputation_strategy=ImputationStrategy.MEAN)

    # Act
    transform.fit(dataset)
    result = transform.transform(dataset)

    # Assert
    assert "empty_feature" not in result.data.columns
    assert list(result.data.columns) == ["radiation", "temperature", "wind_speed"]
    assert not result.data.isna().any().any()


def test_no_missing_values():
    # Arrange
    data = pd.DataFrame(
        {
            "radiation": [100.0, 110.0, 120.0],
            "temperature": [20.0, 21.0, 23.0],
            "wind_speed": [5.0, 6.0, 8.0],
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=1))
    original_data = dataset.data.copy()
    transform = MissingValuesTransform(imputation_strategy=ImputationStrategy.MEAN)

    # Act
    transform.fit(dataset)
    result = transform.transform(dataset)

    # Assert
    pd.testing.assert_frame_equal(result.data, original_data)


def test_custom_missing_value_placeholder():
    # Arrange
    data = pd.DataFrame(
        {
            "radiation": [100.0, -999.0, 120.0],
            "temperature": [20.0, 21.0, -999.0],
            "wind_speed": [5.0, 6.0, 7.0],
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=1))
    transform = MissingValuesTransform(missing_value=-999.0, imputation_strategy=ImputationStrategy.MEAN)

    # Act
    transform.fit(dataset)
    result = transform.transform(dataset)

    # Assert
    assert not (result.data == -999.0).any().any()
    assert result.data.loc[result.data.index[1], "radiation"] == 110.0
    assert result.data.loc[result.data.index[2], "temperature"] == 20.5


def test_all_null_dataset_with_trailing_removal(caplog: LogCaptureFixture):
    # Arrange
    data = pd.DataFrame(
        {
            "radiation": [np.nan, np.nan, np.nan],
            "temperature": [20.0, 21.0, 22.0],
            "wind_speed": [5.0, 6.0, 8.0],
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=1))
    transform = MissingValuesTransform(
        imputation_strategy=ImputationStrategy.CONSTANT, fill_value=0.0, no_fill_future_values_features=["radiation"]
    )

    # Act
    with caplog.at_level(logging.WARNING):
        transform.fit(dataset)

    result = transform.transform(dataset)

    # Assert
    assert len(result.data) == 3  # One row removed
    assert "Dropped column 'radiation' from dataset because it contains only missing values." in caplog.text
    assert "Features 'radiation' not found in dataset columns." in caplog.text


def test_drop_empty_feature_with_custom_missing_value(caplog: LogCaptureFixture):
    # Arrange
    data = pd.DataFrame(
        {
            "radiation": [-999.0, -999.0, -999.0],  # All missing
            "temperature": [20.0, 21.0, 22.0],  # No missing
            "wind_speed": [5.0, -999.0, 8.0],  # Some missing
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1h"),
    )
    transform = MissingValuesTransform(
        missing_value=-999.0, imputation_strategy=ImputationStrategy.CONSTANT, fill_value=0.0
    )

    # Act
    with caplog.at_level(logging.WARNING):
        result = transform._drop_empty_features(data)

    # Assert
    assert "radiation" not in result.columns
    assert "temperature" in result.columns
    assert "wind_speed" in result.columns
    assert "Dropped column 'radiation' from dataset because it contains only missing values." in caplog.text
