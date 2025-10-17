# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from _pytest.logging import LogCaptureFixture

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import NotFittedError
from openstef_models.transforms.general import EmptyFeatureRemover
from openstef_models.utils.feature_selection import FeatureSelection


@pytest.fixture
def sample_dataset_with_empty_columns() -> TimeSeriesDataset:
    """Create a sample dataset with some empty columns for testing."""
    data = pd.DataFrame(
        {
            "radiation": [100.0, 110.0, 120.0],
            "temperature": [20.0, np.nan, 22.0],  # Has some missing values but not empty
            "wind_speed": [5.0, 6.0, 7.0],  # No missing values
            "empty_col1": [np.nan, np.nan, np.nan],  # Completely empty
            "empty_col2": [np.nan, np.nan, np.nan],  # Completely empty
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1h"),
    )
    return TimeSeriesDataset(data, timedelta(hours=1))


def test_removes_all_empty_columns(sample_dataset_with_empty_columns: TimeSeriesDataset, caplog: LogCaptureFixture):
    """Test that all empty columns are removed when no columns parameter is specified."""
    # Arrange
    transform = EmptyFeatureRemover()

    # Act
    with caplog.at_level(logging.WARNING):
        transform.fit(sample_dataset_with_empty_columns)
        result = transform.transform(sample_dataset_with_empty_columns)

    # Assert
    expected_columns = {"radiation", "temperature", "wind_speed"}
    assert set(result.data.columns) == expected_columns
    assert result.sample_interval == sample_dataset_with_empty_columns.sample_interval
    # Check that warning was logged
    assert "Dropped columns from dataset because they contain only missing values:" in caplog.text
    assert "empty_col1" in caplog.text
    assert "empty_col2" in caplog.text


def test_removes_only_specified_empty_columns(sample_dataset_with_empty_columns: TimeSeriesDataset):
    """Test that only specified empty columns are removed."""
    # Arrange
    transform = EmptyFeatureRemover(selection=FeatureSelection(include={"empty_col1", "radiation"}))

    # Act
    transform.fit(sample_dataset_with_empty_columns)
    result = transform.transform(sample_dataset_with_empty_columns)

    # Assert
    # empty_col1 should be removed, empty_col2 should remain (not checked)
    # radiation should remain (not empty)
    expected_columns = {"radiation", "temperature", "wind_speed", "empty_col2"}
    assert set(result.data.columns) == expected_columns


def test_custom_missing_value_placeholder():
    """Test that custom missing value placeholders are handled correctly."""
    # Arrange
    data = pd.DataFrame(
        {
            "radiation": [100.0, 110.0, 120.0],
            "empty_col": [-999.0, -999.0, -999.0],  # Empty with custom placeholder
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=1))
    transform = EmptyFeatureRemover(missing_value=-999.0)

    # Act
    transform.fit(dataset)
    result = transform.transform(dataset)

    # Assert
    assert list(result.data.columns) == ["radiation"]


def test_no_empty_columns_preserves_data():
    """Test that data is preserved when no empty columns exist."""
    # Arrange
    data = pd.DataFrame(
        {
            "radiation": [100.0, 110.0, 120.0],
            "temperature": [20.0, 21.0, 22.0],
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=1))
    original_data = dataset.data.copy()
    transform = EmptyFeatureRemover()

    # Act
    transform.fit(dataset)
    result = transform.transform(dataset)

    # Assert
    pd.testing.assert_frame_equal(result.data, original_data)
    assert result.sample_interval == dataset.sample_interval


def test_transform_not_fitted_error():
    """Test that NotFittedError is raised when transform is called before fit."""
    # Arrange
    data = pd.DataFrame(
        {"radiation": [100.0, 110.0]},
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=2, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=1))
    transform = EmptyFeatureRemover()

    # Act & Assert
    with pytest.raises(NotFittedError):
        transform.transform(dataset)


def test_nonexistent_columns_are_ignored():
    """Test that specifying nonexistent columns doesn't cause errors."""
    # Arrange
    data = pd.DataFrame(
        {
            "radiation": [100.0, 110.0, 120.0],
            "empty_col": [np.nan, np.nan, np.nan],
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=1))
    transform = EmptyFeatureRemover(selection=FeatureSelection(include={"nonexistent", "empty_col"}))

    # Act
    transform.fit(dataset)
    result = transform.transform(dataset)

    # Assert
    assert list(result.data.columns) == ["radiation"]


def test_remove_empty_columns_transform__state_roundtrip(sample_dataset_with_empty_columns: TimeSeriesDataset):
    """Test remove empty columns transform state serialization and restoration."""
    # Arrange
    original_transform = EmptyFeatureRemover(
        selection=FeatureSelection(include={"empty_col1", "radiation", "temperature"})
    )
    original_transform.fit(sample_dataset_with_empty_columns)

    # Act
    state = original_transform.to_state()
    restored_transform = EmptyFeatureRemover()
    restored_transform = restored_transform.from_state(state)

    original_result = original_transform.transform(sample_dataset_with_empty_columns)
    restored_result = restored_transform.transform(sample_dataset_with_empty_columns)

    # Assert
    pd.testing.assert_frame_equal(original_result.data, restored_result.data)
