# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for core functionality of VersionedTimeSeriesDataset."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from openstef_core.datasets.versioned_timeseries_dataset import VersionedTimeSeriesDataset
from openstef_core.datasets.versioned_timeseries_dataset_part import VersionedTimeSeriesPart
from openstef_core.exceptions import TimeSeriesValidationError


# Initialization tests
def test_initialization_with_multiple_parts(
    dataset_part_a: VersionedTimeSeriesPart, dataset_part_b: VersionedTimeSeriesPart
):
    # Act
    dataset = VersionedTimeSeriesDataset([dataset_part_a, dataset_part_b])

    # Assert
    assert len(dataset.data_parts) == 2
    assert dataset.sample_interval == timedelta(hours=1)
    assert sorted(dataset.feature_names) == ["feature_a", "feature_b"]
    assert isinstance(dataset.index, pd.DatetimeIndex)
    assert len(dataset.index) == 5  # Union of both parts' indices


def test_initialization_with_single_part(dataset_part_a: VersionedTimeSeriesPart):
    # Act
    dataset = VersionedTimeSeriesDataset([dataset_part_a])

    # Assert
    assert len(dataset.data_parts) == 1
    assert dataset.sample_interval == timedelta(hours=1)
    assert dataset.feature_names == ["feature_a"]
    assert len(dataset.index) == 4


def test_initialization_with_custom_index(
    dataset_part_a: VersionedTimeSeriesPart, dataset_part_b: VersionedTimeSeriesPart
):
    # Arrange
    custom_index = pd.date_range("2023-01-01T09:00:00", "2023-01-01T15:00:00", freq="1h")

    # Act
    dataset = VersionedTimeSeriesDataset([dataset_part_a, dataset_part_b], index=custom_index)

    # Assert
    assert len(dataset.index) == len(custom_index)
    pd.testing.assert_index_equal(dataset.index, custom_index)


def test_initialization_with_empty_parts():
    # Act & Assert
    with pytest.raises(TimeSeriesValidationError, match="At least one data part must be provided"):
        VersionedTimeSeriesDataset([])


def test_initialization_with_mismatched_sample_intervals():
    # Arrange
    data1 = pd.DataFrame({
        "timestamp": pd.to_datetime(["2023-01-01T10:00:00"]),
        "available_at": pd.to_datetime(["2023-01-01T10:05:00"]),
        "feature1": [1.0],
    })
    data2 = pd.DataFrame({
        "timestamp": pd.to_datetime(["2023-01-01T10:00:00"]),
        "available_at": pd.to_datetime(["2023-01-01T10:05:00"]),
        "feature2": [2.0],
    })

    part1 = VersionedTimeSeriesPart(data1, timedelta(hours=1))
    part2 = VersionedTimeSeriesPart(data2, timedelta(minutes=30))  # Different interval

    # Act & Assert
    with pytest.raises(TimeSeriesValidationError, match="Datasets have different sample intervals"):
        VersionedTimeSeriesDataset([part1, part2])


def test_initialization_with_overlapping_features():
    # Arrange
    data1 = pd.DataFrame({
        "timestamp": pd.to_datetime(["2023-01-01T10:00:00"]),
        "available_at": pd.to_datetime(["2023-01-01T10:05:00"]),
        "feature1": [1.0],
        "shared": [10.0],  # Overlapping feature
    })
    data2 = pd.DataFrame({
        "timestamp": pd.to_datetime(["2023-01-01T10:00:00"]),
        "available_at": pd.to_datetime(["2023-01-01T10:05:00"]),
        "feature2": [2.0],
        "shared": [20.0],  # Overlapping feature
    })

    part1 = VersionedTimeSeriesPart(data1, timedelta(hours=1))
    part2 = VersionedTimeSeriesPart(data2, timedelta(hours=1))

    # Act & Assert
    with pytest.raises(TimeSeriesValidationError, match="Datasets have overlapping feature names"):
        VersionedTimeSeriesDataset([part1, part2])


# from_dataframe tests


def test_from_dataframe_basic():
    # Arrange
    data = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2023-01-01T10:00:00",
            "2023-01-01T11:00:00",
        ]),
        "available_at": pd.to_datetime([
            "2023-01-01T10:05:00",
            "2023-01-01T11:05:00",
        ]),
        "feature_a": [20.0, 22.0],
        "feature_b": [100.0, 120.0],
    })

    # Act
    dataset = VersionedTimeSeriesDataset.from_dataframe(data, timedelta(hours=1))

    # Assert
    assert len(dataset.data_parts) == 1
    assert dataset.sample_interval == timedelta(hours=1)
    assert sorted(dataset.feature_names) == ["feature_a", "feature_b"]
    assert len(dataset.index) == 2


def test_from_dataframe_custom_columns():
    # Arrange
    data = pd.DataFrame({
        "custom_ts": pd.to_datetime(["2023-01-01T10:00:00"]),
        "custom_avail": pd.to_datetime(["2023-01-01T10:05:00"]),
        "value": [42.0],
    })

    # Act
    dataset = VersionedTimeSeriesDataset.from_dataframe(
        data, timedelta(hours=1), timestamp_column="custom_ts", available_at_column="custom_avail"
    )

    # Assert
    assert dataset.feature_names == ["value"]
    assert dataset.data_parts[0].timestamp_column == "custom_ts"
    assert dataset.data_parts[0].available_at_column == "custom_avail"


# Property tests


def test_feature_names_property(combined_dataset: VersionedTimeSeriesDataset):
    # Act
    features = combined_dataset.feature_names

    # Assert
    assert sorted(features) == ["feature_a", "feature_b"]


def test_index_property(combined_dataset: VersionedTimeSeriesDataset):
    # Arrange
    expected_timestamps = pd.to_datetime([
        "2023-01-01T10:00:00",
        "2023-01-01T11:00:00",
        "2023-01-01T12:00:00",
        "2023-01-01T13:00:00",
        "2023-01-01T14:00:00",
    ])

    # Act
    index = combined_dataset.index

    # Assert
    assert isinstance(index, pd.DatetimeIndex)
    assert len(index) == 5
    assert all(ts in index for ts in expected_timestamps)


def test_sample_interval_property(combined_dataset: VersionedTimeSeriesDataset):
    # Act
    interval = combined_dataset.sample_interval

    # Assert
    assert interval == timedelta(hours=1)


# Version selection tests


def test_select_version_basic(combined_dataset: VersionedTimeSeriesDataset):
    # Act
    result = combined_dataset.select_version(available_before=None)

    # Assert
    assert isinstance(result.data, pd.DataFrame)
    assert len(result.data) == 5  # All timestamps
    assert sorted(result.data.columns) == ["feature_a", "feature_b"]


def test_select_version_with_cutoff(combined_dataset: VersionedTimeSeriesDataset):
    # Arrange
    cutoff = pd.to_datetime("2023-01-01T12:00:00")

    # Act
    result = combined_dataset.select_version(available_before=cutoff)

    # Assert
    assert isinstance(result.data, pd.DataFrame)
    # Should have some data but not all due to availability cutoff
    assert len(result.data) == 5  # All timestamps in index
    # Check that some values are NaN due to availability cutoff
    total_values = result.data.notna().sum().sum()
    assert total_values < len(result.data) * len(result.data.columns)  # Some should be missing

    # Check specific values - early timestamps should have data available
    # feature_a: available at 10:05, 11:05 (both before 12:00 cutoff)
    # feature_b: available at 10:10, 11:10 (both before 12:00 cutoff)
    assert not pd.isna(result.data.loc[pd.to_datetime("2023-01-01T10:00:00"), "feature_a"])
    assert not pd.isna(result.data.loc[pd.to_datetime("2023-01-01T11:00:00"), "feature_a"])
    assert not pd.isna(result.data.loc[pd.to_datetime("2023-01-01T10:00:00"), "feature_b"])
    assert not pd.isna(result.data.loc[pd.to_datetime("2023-01-01T11:00:00"), "feature_b"])


def test_select_version_empty_result():
    # Arrange - create a dataset with late availability
    data = pd.DataFrame({
        "timestamp": pd.to_datetime(["2023-01-01T10:00:00"]),
        "available_at": pd.to_datetime(["2023-01-01T12:00:00"]),  # Very late
        "feature": [42.0],
    })
    dataset = VersionedTimeSeriesDataset.from_dataframe(data, timedelta(hours=1))

    # Act - select with early cutoff
    result = dataset.select_version(available_before=pd.to_datetime("2023-01-01T10:30:00"))

    # Assert
    assert isinstance(result.data, pd.DataFrame)
    assert len(result.data) == 1  # Still has the timestamp
    assert pd.isna(result.data.iloc[0]["feature"])  # But value should be missing


# Complex scenarios


def test_dataset_union_behavior(dataset_part_a: VersionedTimeSeriesPart, dataset_part_b: VersionedTimeSeriesPart):
    # Arrange - dataset_part_b has extra timestamp
    dataset = VersionedTimeSeriesDataset([dataset_part_a, dataset_part_b])

    # Act
    result = dataset.select_version(available_before=None)

    # Assert
    assert len(result.data) == 5  # Union includes all timestamps
    # Check that 14:00 only has feature_b data
    row_14 = result.data[result.data.index == datetime.fromisoformat("2023-01-01T14:00:00")]
    assert len(row_14) == 1
    assert row_14.iloc[0]["feature_b"] == 125.0  # Has value
    assert pd.isna(row_14.iloc[0]["feature_a"])  # Missing for this timestamp


def test_realistic_energy_forecasting_scenario():
    # Arrange - simulate actual energy forecasting data
    forecast_data = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01T10:00:00", periods=6, freq="1h"),
        "available_at": pd.date_range("2023-01-01T09:30:00", periods=6, freq="1h"),  # 30 min ahead
        "forecast": [100, 95, 105, 110, 98, 102],
    })

    actual_data = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01T10:00:00", periods=6, freq="1h"),
        "available_at": pd.date_range("2023-01-01T11:00:00", periods=6, freq="1h"),  # 1 hour delay
        "actual": [102, 97, 108, 107, 96, 104],
    })

    forecast_part = VersionedTimeSeriesPart(forecast_data, timedelta(hours=1))
    actual_part = VersionedTimeSeriesPart(actual_data, timedelta(hours=1))
    dataset = VersionedTimeSeriesDataset([forecast_part, actual_part])

    # Act - what's available at 12:00?
    available_at_12 = dataset.select_version(available_before=datetime.fromisoformat("2023-01-01T12:00:00"))

    # Assert
    assert len(available_at_12.data) == 6
    # Forecasts should be available for early hours
    assert not pd.isna(available_at_12.data.iloc[0]["forecast"])  # 10:00 forecast available
    assert not pd.isna(available_at_12.data.iloc[1]["forecast"])  # 11:00 forecast available

    # Actuals: 10:00 actual available at 11:00, 11:00 actual available at 12:00
    # Both should be available when we check available_before 12:00 (inclusive)
    assert not pd.isna(available_at_12.data.iloc[0]["actual"])  # 10:00 actual available at 11:00
    assert not pd.isna(available_at_12.data.iloc[1]["actual"])  # 11:00 actual available at 12:00
