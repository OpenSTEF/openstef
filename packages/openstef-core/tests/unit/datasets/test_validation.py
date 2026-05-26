# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import pandas as pd
import pytest

from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.datasets.validation import (
    validate_datetime_column,
    validate_disjoint_columns,
    validate_required_columns,
    validate_same_columns,
    validate_same_sample_intervals,
)
from openstef_core.exceptions import (
    InvalidColumnTypeError,
    MissingColumnsError,
    TimeSeriesValidationError,
)


def test_validate_required_columns__valid_columns():
    """Test validation passes when all required columns are present."""
    # Arrange
    df = pd.DataFrame({"load": [100, 110], "temp": [20, 22]})

    # Act & Assert - should not raise
    validate_required_columns(df, ["load", "temp"])


def test_validate_required_columns__missing_columns_raises_error():
    """Test validation fails when required columns are missing."""
    # Arrange
    df = pd.DataFrame({"load": [100, 110]})

    # Act & Assert
    with pytest.raises(MissingColumnsError) as exc_info:
        validate_required_columns(df, ["load", "temp", "humidity"])

    # Assert error contains missing column names
    assert "temp" in str(exc_info.value)
    assert "humidity" in str(exc_info.value)


def test_validate_disjoint_columns__disjoint_features():
    """Test validation passes when datasets have disjoint feature sets."""
    # Arrange
    data1 = pd.DataFrame({"load": [100]}, index=pd.date_range("2025-01-01", periods=1, freq="1h"))
    data2 = pd.DataFrame({"temp": [20]}, index=pd.date_range("2025-01-01", periods=1, freq="1h"))
    dataset1 = TimeSeriesDataset(data1, timedelta(hours=1))
    dataset2 = TimeSeriesDataset(data2, timedelta(hours=1))

    # Act
    all_features = validate_disjoint_columns([dataset1, dataset2])

    # Assert - returns combined feature list
    assert set(all_features) == {"load", "temp"}


def test_validate_disjoint_columns__overlapping_features_raises_error():
    """Test validation fails when datasets have overlapping features."""
    # Arrange
    data1 = pd.DataFrame({"load": [100], "temp": [20]}, index=pd.date_range("2025-01-01", periods=1, freq="1h"))
    data2 = pd.DataFrame({"temp": [22], "wind": [5]}, index=pd.date_range("2025-01-01", periods=1, freq="1h"))
    dataset1 = TimeSeriesDataset(data1, timedelta(hours=1))
    dataset2 = TimeSeriesDataset(data2, timedelta(hours=1))

    # Act & Assert
    with pytest.raises(TimeSeriesValidationError) as exc_info:
        validate_disjoint_columns([dataset1, dataset2])

    # Assert error mentions the duplicate feature
    assert "temp" in str(exc_info.value)


def test_validate_same_columns__identical_features():
    """Test validation passes when datasets have identical feature sets."""
    # Arrange
    data1 = pd.DataFrame({"load": [100], "temp": [20]}, index=pd.date_range("2025-01-01", periods=1, freq="1h"))
    data2 = pd.DataFrame({"load": [110], "temp": [22]}, index=pd.date_range("2025-01-02", periods=1, freq="1h"))
    dataset1 = TimeSeriesDataset(data1, timedelta(hours=1))
    dataset2 = TimeSeriesDataset(data2, timedelta(hours=1))

    # Act
    common_features = validate_same_columns([dataset1, dataset2])

    # Assert - returns common feature list
    assert set(common_features) == {"load", "temp"}


def test_validate_same_columns__different_features_raises_error():
    """Test validation fails when datasets have different feature sets."""
    # Arrange
    data1 = pd.DataFrame({"load": [100], "temp": [20]}, index=pd.date_range("2025-01-01", periods=1, freq="1h"))
    data2 = pd.DataFrame({"load": [110], "wind": [5]}, index=pd.date_range("2025-01-02", periods=1, freq="1h"))
    dataset1 = TimeSeriesDataset(data1, timedelta(hours=1))
    dataset2 = TimeSeriesDataset(data2, timedelta(hours=1))

    # Act & Assert
    with pytest.raises(TimeSeriesValidationError) as exc_info:
        validate_same_columns([dataset1, dataset2])

    # Assert error message indicates different features
    assert "different feature names" in str(exc_info.value)


def test_validate_same_sample_intervals__identical_intervals():
    """Test validation passes when datasets have identical sample intervals."""
    # Arrange
    data1 = pd.DataFrame({"load": [100]}, index=pd.date_range("2025-01-01", periods=1, freq="1h"))
    data2 = pd.DataFrame({"load": [110]}, index=pd.date_range("2025-01-02", periods=1, freq="1h"))
    dataset1 = TimeSeriesDataset(data1, timedelta(hours=1))
    dataset2 = TimeSeriesDataset(data2, timedelta(hours=1))

    # Act
    common_interval = validate_same_sample_intervals([dataset1, dataset2])

    # Assert - returns common interval
    assert common_interval == timedelta(hours=1)


def test_validate_same_sample_intervals__different_intervals_raises_error():
    """Test validation fails when datasets have different sample intervals."""
    # Arrange
    data1 = pd.DataFrame({"load": [100]}, index=pd.date_range("2025-01-01", periods=1, freq="1h"))
    data2 = pd.DataFrame({"load": [110]}, index=pd.date_range("2025-01-02", periods=1, freq="15min"))
    dataset1 = TimeSeriesDataset(data1, timedelta(hours=1))
    dataset2 = TimeSeriesDataset(data2, timedelta(minutes=15))

    # Act & Assert
    with pytest.raises(TimeSeriesValidationError) as exc_info:
        validate_same_sample_intervals([dataset1, dataset2])

    # Assert error mentions different sample intervals
    assert "different sample intervals" in str(exc_info.value)


def test_validate_datetime_column__valid_datetime():
    """Test validation passes for datetime series."""
    # Arrange
    series = pd.Series(pd.date_range("2025-01-01", periods=3, freq="1h"), name="timestamp")

    # Act & Assert - should not raise
    validate_datetime_column(series)


def test_validate_datetime_column__non_datetime_raises_error():
    """Test validation fails for non-datetime series."""
    # Arrange
    series = pd.Series([1, 2, 3], name="timestamp")

    # Act & Assert
    with pytest.raises(InvalidColumnTypeError) as exc_info:
        validate_datetime_column(series)

    # Assert error indicates expected datetime type
    assert "datetime" in str(exc_info.value)
