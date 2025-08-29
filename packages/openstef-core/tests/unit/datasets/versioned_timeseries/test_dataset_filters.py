# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for filtering functionality of VersionedTimeSeriesDataset."""

from datetime import datetime, timedelta

import pandas as pd

from openstef_core.datasets.versioned_timeseries.dataset import VersionedTimeSeriesDataset
from openstef_core.datasets.versioned_timeseries.dataset_part import VersionedTimeSeriesPart
from openstef_core.types import AvailableAt, LeadTime

# Basic filtering tests


def test_filter_by_range_basic(combined_dataset: VersionedTimeSeriesDataset):
    # Arrange
    start = datetime.fromisoformat("2023-01-01T11:00:00")
    end = datetime.fromisoformat("2023-01-01T13:00:00")

    # Act
    filtered = combined_dataset.filter_by_range(start, end)

    # Assert
    assert isinstance(filtered, VersionedTimeSeriesDataset)
    assert len(filtered.data_parts) == 2
    assert len(filtered.index) == 2  # 11:00 and 12:00
    assert filtered.sample_interval == timedelta(hours=1)
    assert sorted(filtered.feature_names) == ["feature_a", "feature_b"]

    # Check that filtering preserved the correct data values
    result = filtered.select_version(available_before=None)
    expected_feature_a_values = [22.0, 24.0]  # Values at 11:00 and 12:00
    expected_feature_b_values = [120.0, 110.0]  # Values at 11:00 and 12:00

    assert result.data["feature_a"].dropna().tolist() == expected_feature_a_values
    assert result.data["feature_b"].dropna().tolist() == expected_feature_b_values


def test_filter_by_range_partial_coverage(combined_dataset: VersionedTimeSeriesDataset):
    # Arrange - range that goes beyond available data
    start = datetime.fromisoformat("2023-01-01T12:00:00")
    end = datetime.fromisoformat("2023-01-01T16:00:00")

    # Act
    filtered = combined_dataset.filter_by_range(start, end)

    # Assert
    assert len(filtered.index) == 3  # 12:00, 13:00, 14:00

    # Check that filtering preserved the correct data values
    result = filtered.select_version(available_before=None)
    # feature_a values: 24.0 at 12:00, 23.0 at 13:00, NaN at 14:00 (no data)
    # feature_b values: 110.0 at 12:00, 105.0 at 13:00, 125.0 at 14:00
    expected_feature_a_values = [24.0, 23.0]  # Only non-NaN values
    expected_feature_b_values = [110.0, 105.0, 125.0]  # All values available

    assert result.data["feature_a"].dropna().tolist() == expected_feature_a_values
    assert result.data["feature_b"].dropna().tolist() == expected_feature_b_values


def test_filter_by_available_at(combined_dataset: VersionedTimeSeriesDataset):
    # Arrange
    available_at = AvailableAt.from_string("D0T10:30")  # Available by 10:30 on same day

    # Act
    filtered = combined_dataset.filter_by_available_at(available_at)

    # Assert
    assert isinstance(filtered, VersionedTimeSeriesDataset)
    assert len(filtered.data_parts) == 2
    assert sorted(filtered.feature_names) == ["feature_a", "feature_b"]
    # Both parts should be filtered independently


def test_filter_by_lead_time(combined_dataset: VersionedTimeSeriesDataset):
    # Arrange
    lead_time = LeadTime.from_string("PT1H")  # 1 hour lead time

    # Act
    filtered = combined_dataset.filter_by_lead_time(lead_time)

    # Assert
    assert isinstance(filtered, VersionedTimeSeriesDataset)
    assert len(filtered.data_parts) == 2
    assert sorted(filtered.feature_names) == ["feature_a", "feature_b"]


# Chaining operations tests


def test_filter_and_select_chain(combined_dataset: VersionedTimeSeriesDataset):
    # Arrange
    start = datetime.fromisoformat("2023-01-01T11:00:00")
    end = datetime.fromisoformat("2023-01-01T13:00:00")
    available_at = AvailableAt.from_string("D0T12:00")

    # Act
    result = (
        combined_dataset.filter_by_range(start, end)
        .filter_by_available_at(available_at)
        .select_version(available_before=None)
    )

    # Assert
    assert isinstance(result.data, pd.DataFrame)
    assert len(result.data) == 2  # Filtered range
    assert sorted(result.data.columns) == ["feature_a", "feature_b"]


def test_multiple_filters_chain(combined_dataset: VersionedTimeSeriesDataset):
    # Arrange
    start = datetime.fromisoformat("2023-01-01T10:00:00")
    end = datetime.fromisoformat("2023-01-01T14:00:00")
    available_at = AvailableAt.from_string("D0T11:00")
    lead_time = LeadTime.from_string("PT30M")

    # Act
    filtered = (
        combined_dataset.filter_by_range(start, end).filter_by_available_at(available_at).filter_by_lead_time(lead_time)
    )

    # Assert
    assert isinstance(filtered, VersionedTimeSeriesDataset)
    assert len(filtered.data_parts) == 2
    assert sorted(filtered.feature_names) == ["feature_a", "feature_b"]


# Edge cases and error handling


def test_empty_filter_result():
    # Arrange - create a dataset and filter to empty result
    data = pd.DataFrame({
        "timestamp": pd.to_datetime(["2023-01-01T10:00:00"]),
        "available_at": pd.to_datetime(["2023-01-01T12:00:00"]),  # Late availability
        "value": [42.0],
    })
    dataset = VersionedTimeSeriesDataset.from_dataframe(data, timedelta(hours=1))

    # Act - filter with early availability requirement
    filtered = dataset.filter_by_available_at(AvailableAt.from_string("D0T10:30"))

    # Assert
    assert isinstance(filtered, VersionedTimeSeriesDataset)
    # Parts should be empty but structure preserved
    assert len(filtered.data_parts) == 1


def test_index_preservation_through_filters(combined_dataset: VersionedTimeSeriesDataset):
    # Arrange
    original_index = combined_dataset.index

    # Act
    filtered = combined_dataset.filter_by_available_at(AvailableAt.from_string("D0T15:00"))

    # Assert
    # Index should be preserved even if data is filtered
    pd.testing.assert_index_equal(filtered.index, original_index)


def test_realistic_backtesting_scenario():
    # Arrange - simulate realistic data with different availability patterns
    dataset_a_data = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01T10:00:00", periods=24, freq="1h"),
        "available_at": pd.date_range("2023-01-01T10:10:00", periods=24, freq="1h"),  # 10 min delay
        "feature_a": range(100, 124),
    })

    dataset_b_data = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01T10:00:00", periods=24, freq="1h"),
        "available_at": pd.date_range("2023-01-01T09:00:00", periods=24, freq="1h"),  # 1 hour ahead
        "feature_b": range(95, 119),
    })

    part_a = VersionedTimeSeriesPart(dataset_a_data, timedelta(hours=1))
    part_b = VersionedTimeSeriesPart(dataset_b_data, timedelta(hours=1))
    dataset = VersionedTimeSeriesDataset([part_a, part_b])

    # Act - simulate backtesting: forecast available early, actuals with delay
    backtest_window = dataset.filter_by_range(
        pd.to_datetime("2023-01-01T12:00:00"), pd.to_datetime("2023-01-01T18:00:00")
    )

    # Available data as of 13:00
    available_data = backtest_window.select_version(available_before=pd.to_datetime("2023-01-01T13:00:00"))

    # Assert
    assert len(available_data.data) == 6  # 6 hour window
    assert "feature_a" in available_data.data.columns
    assert "feature_b" in available_data.data.columns

    # Forecast should be available for early timestamps (those available before 13:00)
    feature_b_available_count = available_data.data["feature_b"].notna().sum()
    assert feature_b_available_count >= 3  # At least the first few hours should be available

    # Actuals should have some missing values due to availability delay
    feature_a_available = available_data.data["feature_a"].notna().sum()
    assert feature_a_available < len(available_data.data)  # Some should be missing


def test_dataset_composition_maintains_part_independence():
    # Arrange - create parts with different characteristics
    fast_data = pd.DataFrame({
        "timestamp": pd.to_datetime(["2023-01-01T10:00:00"]),
        "available_at": pd.to_datetime(["2023-01-01T10:01:00"]),  # Very fast
        "fast_feature": [1.0],
    })

    slow_data = pd.DataFrame({
        "timestamp": pd.to_datetime(["2023-01-01T10:00:00"]),
        "available_at": pd.to_datetime(["2023-01-01T12:00:00"]),  # Very slow
        "slow_feature": [2.0],
    })

    fast_part = VersionedTimeSeriesPart(fast_data, timedelta(hours=1))
    slow_part = VersionedTimeSeriesPart(slow_data, timedelta(hours=1))
    dataset = VersionedTimeSeriesDataset([fast_part, slow_part])

    # Act - filter with intermediate availability
    filtered = dataset.filter_by_available_at(AvailableAt.from_string("D0T11:00"))
    result = filtered.select_version(available_before=None)

    # Assert
    assert len(result.data) == 1
    assert not pd.isna(result.data.iloc[0]["fast_feature"])  # Fast feature available
    assert pd.isna(result.data.iloc[0]["slow_feature"])  # Slow feature not available
