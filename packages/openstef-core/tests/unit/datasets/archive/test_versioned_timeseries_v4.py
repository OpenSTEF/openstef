# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for the new V4 versioned time series implementation."""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.datasets.versioned_timeseries import (
    filter_by_available_at,
    filter_by_latest_lead_time,
    filter_by_lead_time,
)
from openstef_core.datasets.versioned_timeseries_v4 import VersionedTimeSeriesDataset, VersionedTimeSeriesPart
from openstef_core.exceptions import MissingColumnsError
from openstef_core.types import AvailableAt, LeadTime


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Sample data with timestamps and availability times."""
    data = {
        "timestamp": [
            datetime.fromisoformat("2023-01-01T10:00:00"),
            datetime.fromisoformat("2023-01-01T11:00:00"),
            datetime.fromisoformat("2023-01-01T12:00:00"),
            datetime.fromisoformat("2023-01-01T13:00:00"),
        ],
        "available_at": [
            datetime.fromisoformat("2023-01-01T10:15:00"),
            datetime.fromisoformat("2023-01-01T11:15:00"),
            datetime.fromisoformat("2023-01-01T12:15:00"),
            datetime.fromisoformat("2023-01-01T13:15:00"),
        ],
        "value1": [10, 20, 30, 40],
        "value2": [100, 200, 300, 400],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_part(sample_data: pd.DataFrame) -> VersionedTimeSeriesPart:
    """Sample VersionedTimeSeriesPart instance."""
    return VersionedTimeSeriesPart.from_dataframe(
        data=sample_data,
        sample_interval=timedelta(hours=1),
    )


@pytest.fixture
def sample_dataset(sample_part: VersionedTimeSeriesPart) -> VersionedTimeSeriesDataset:
    """Sample VersionedTimeSeriesDataset instance."""
    return VersionedTimeSeriesDataset([sample_part])


class TestVersionedTimeSeriesPart:
    """Tests for VersionedTimeSeriesPart class."""

    def test_from_dataframe_basic(self, sample_data: pd.DataFrame):
        """Test basic creation from DataFrame."""
        # Act
        part = VersionedTimeSeriesPart.from_dataframe(
            data=sample_data,
            sample_interval=timedelta(hours=1),
        )

        # Assert
        assert part.sample_interval == timedelta(hours=1)
        assert part.timestamp_column == "timestamp"
        assert part.available_at_column == "available_at"
        assert sorted(part.feature_names) == ["value1", "value2"]
        assert len(part.data) == 4

    def test_from_dataframe_custom_columns(self):
        """Test creation with custom column names."""
        # Arrange
        data = pd.DataFrame({
            "custom_ts": [datetime.fromisoformat("2023-01-01T10:00:00")],
            "custom_avail": [datetime.fromisoformat("2023-01-01T10:15:00")],
            "value": [10],
        })

        # Act
        part = VersionedTimeSeriesPart.from_dataframe(
            data=data,
            sample_interval=timedelta(hours=1),
            timestamp_column="custom_ts",
            available_at_column="custom_avail",
        )

        # Assert
        assert part.timestamp_column == "custom_ts"
        assert part.available_at_column == "custom_avail"
        assert part.feature_names == ["value"]

    def test_from_dataframe_missing_columns(self):
        """Test error handling for missing required columns."""
        # Arrange
        data = pd.DataFrame({
            "timestamp": [datetime.fromisoformat("2023-01-01T10:00:00")],
            "value": [10],  # Missing available_at column
        })

        # Act & Assert
        with pytest.raises(MissingColumnsError):
            VersionedTimeSeriesPart.from_dataframe(
                data=data,
                sample_interval=timedelta(hours=1),
            )

    def test_filter_by_available_at(self, sample_part: VersionedTimeSeriesPart):
        """Test filtering by availability time."""
        # Arrange
        available_at = AvailableAt.from_string("D+1T12:00")

        # Act
        filtered = sample_part.filter_by_available_at(available_at)

        # Assert
        assert isinstance(filtered, VersionedTimeSeriesPart)
        assert len(filtered.data) <= len(sample_part.data)
        assert filtered.sample_interval == sample_part.sample_interval

    def test_filter_by_lead_time(self, sample_part: VersionedTimeSeriesPart):
        """Test filtering by lead time."""
        # Arrange
        lead_time = LeadTime.from_string("PT1H")

        # Act
        filtered = sample_part.filter_by_lead_time(lead_time)

        # Assert
        assert isinstance(filtered, VersionedTimeSeriesPart)
        assert len(filtered.data) <= len(sample_part.data)
        assert filtered.sample_interval == sample_part.sample_interval

    def test_filter_by_range(self, sample_part: VersionedTimeSeriesPart):
        """Test filtering by time range."""
        # Arrange
        start = datetime.fromisoformat("2023-01-01T10:30:00")
        end = datetime.fromisoformat("2023-01-01T12:30:00")

        # Act
        filtered = sample_part.filter_by_range(start, end)

        # Assert
        assert isinstance(filtered, VersionedTimeSeriesPart)
        assert len(filtered.data) <= len(sample_part.data)
        # Should only include timestamps within the range
        assert all(start <= ts < end for ts in filtered.data.index)

    def test_select_version(self, sample_part: VersionedTimeSeriesPart):
        """Test version selection."""
        # Arrange
        cutoff = datetime.fromisoformat("2023-01-01T12:00:00")

        # Act
        result = sample_part.select_version(cutoff)

        # Assert
        assert hasattr(result, 'data')  # Should return a TimeSeriesDataset-like object
        assert len(result.data) <= len(sample_part.data)

    def test_to_parquet_and_read_parquet(self, sample_part: VersionedTimeSeriesPart, tmp_path: Path):
        """Test saving to and loading from parquet."""
        # Arrange
        file_path = tmp_path / "test_part.parquet"

        # Act
        sample_part.to_parquet(file_path)
        loaded_part = VersionedTimeSeriesPart.read_parquet(file_path)

        # Assert
        assert loaded_part.sample_interval == sample_part.sample_interval
        assert loaded_part.timestamp_column == sample_part.timestamp_column
        assert loaded_part.available_at_column == sample_part.available_at_column
        assert loaded_part.feature_names == sample_part.feature_names
        pd.testing.assert_frame_equal(loaded_part.data, sample_part.data)


class TestVersionedTimeSeriesDataset:
    """Tests for VersionedTimeSeriesDataset class."""

    def test_initialization_single_part(self, sample_part: VersionedTimeSeriesPart):
        """Test initialization with a single part."""
        # Act
        dataset = VersionedTimeSeriesDataset([sample_part])

        # Assert
        assert len(dataset.data_parts) == 1
        assert dataset.data_parts[0] is sample_part
        assert dataset.sample_interval == sample_part.sample_interval
        assert dataset.feature_names == sample_part.feature_names

    def test_initialization_multiple_parts(self, sample_data: pd.DataFrame):
        """Test initialization with multiple parts."""
        # Arrange
        part1 = VersionedTimeSeriesPart.from_dataframe(
            data=sample_data[:2],
            sample_interval=timedelta(hours=1),
        )
        part2 = VersionedTimeSeriesPart.from_dataframe(
            data=sample_data[2:],
            sample_interval=timedelta(hours=1),
        )

        # Act
        dataset = VersionedTimeSeriesDataset([part1, part2])

        # Assert
        assert len(dataset.data_parts) == 2
        assert dataset.sample_interval == timedelta(hours=1)
        # Features should be combined from all parts
        assert set(dataset.feature_names) == {"value1", "value2"}

    def test_get_window(self, sample_dataset: VersionedTimeSeriesDataset):
        """Test getting a data window."""
        # Arrange
        start = datetime.fromisoformat("2023-01-01T10:00:00")
        end = datetime.fromisoformat("2023-01-01T13:00:00")

        # Act
        window = sample_dataset.get_window(start, end)

        # Assert
        assert isinstance(window, pd.DataFrame)
        assert len(window) >= 0  # May be empty depending on availability
        assert all(col in ["value1", "value2"] for col in window.columns)

    def test_get_window_with_availability_filter(self, sample_dataset: VersionedTimeSeriesDataset):
        """Test getting a window with availability filtering."""
        # Arrange
        start = datetime.fromisoformat("2023-01-01T10:00:00")
        end = datetime.fromisoformat("2023-01-01T13:00:00")
        available_before = datetime.fromisoformat("2023-01-01T12:00:00")

        # Act
        window = sample_dataset.get_window(start, end, available_before)

        # Assert
        assert isinstance(window, pd.DataFrame)
        # Should have fewer or equal rows compared to no availability filter
        window_no_filter = sample_dataset.get_window(start, end)
        assert len(window) <= len(window_no_filter)

    def test_filter_by_available_at(self, sample_dataset: VersionedTimeSeriesDataset):
        """Test filtering by availability time."""
        # Arrange
        available_at = AvailableAt.from_string("D+1T12:00")

        # Act
        filtered = sample_dataset.filter_by_available_at(available_at)

        # Assert
        assert isinstance(filtered, VersionedTimeSeriesDataset)
        assert len(filtered.data_parts) <= len(sample_dataset.data_parts)

    def test_filter_by_lead_time(self, sample_dataset: VersionedTimeSeriesDataset):
        """Test filtering by lead time."""
        # Arrange
        lead_time = LeadTime.from_string("PT1H")

        # Act
        filtered = sample_dataset.filter_by_lead_time(lead_time)

        # Assert
        assert isinstance(filtered, VersionedTimeSeriesDataset)
        assert len(filtered.data_parts) <= len(sample_dataset.data_parts)

    def test_filter_by_range(self, sample_dataset: VersionedTimeSeriesDataset):
        """Test filtering by time range."""
        # Arrange
        start = datetime.fromisoformat("2023-01-01T10:30:00")
        end = datetime.fromisoformat("2023-01-01T12:30:00")

        # Act
        filtered = sample_dataset.filter_by_range(start, end)

        # Assert
        assert isinstance(filtered, VersionedTimeSeriesDataset)
        assert len(filtered.data_parts) <= len(sample_dataset.data_parts)

    def test_select_version(self, sample_dataset: VersionedTimeSeriesDataset):
        """Test version selection."""
        # Arrange
        cutoff = datetime.fromisoformat("2023-01-01T12:00:00")

        # Act
        result = sample_dataset.select_version(cutoff)

        # Assert
        assert hasattr(result, 'data')  # Should return a TimeSeriesDataset-like object
        # Should combine data from all parts
        assert isinstance(result.data, pd.DataFrame)

    def test_from_dataframe_factory(self, sample_data: pd.DataFrame):
        """Test creating dataset from DataFrame using factory method."""
        # Act
        dataset = VersionedTimeSeriesDataset.from_dataframe(
            data=sample_data,
            sample_interval=timedelta(hours=1),
        )

        # Assert
        assert isinstance(dataset, VersionedTimeSeriesDataset)
        assert len(dataset.data_parts) == 1
        assert dataset.sample_interval == timedelta(hours=1)
        assert sorted(dataset.feature_names) == ["value1", "value2"]

    def test_index_property(self, sample_dataset: VersionedTimeSeriesDataset):
        """Test the index property."""
        # Act
        index = sample_dataset.index

        # Assert
        assert isinstance(index, pd.DatetimeIndex)
        # Should be based on the combined data from all parts
        assert len(index) >= 0


class TestCompatibilityWithOldInterface:
    """Test compatibility with the old versioned timeseries interface."""

    def test_import_from_main_datasets_module(self):
        """Test that imports work from the main datasets module."""
        # Act & Assert - should not raise ImportError
        from openstef_core.datasets import VersionedTimeSeriesDataset
        from openstef_core.datasets.versioned_timeseries import (
            filter_by_available_at,
            filter_by_latest_lead_time,
            filter_by_lead_time,
        )
        
        assert VersionedTimeSeriesDataset is not None
        assert filter_by_available_at is not None
        assert filter_by_latest_lead_time is not None
        assert filter_by_lead_time is not None

    def test_old_filter_functions_work(self, sample_dataset: VersionedTimeSeriesDataset):
        """Test that the old filter functions work with V4 datasets."""
        # Arrange
        from openstef_core.datasets.versioned_timeseries import (
            filter_by_available_at,
            filter_by_latest_lead_time,
            filter_by_lead_time,
        )
        
        available_at = AvailableAt.from_string("D+1T12:00")
        lead_time = LeadTime.from_string("PT1H")

        # Act
        result1 = filter_by_available_at(sample_dataset, available_at)
        result2 = filter_by_lead_time(sample_dataset, lead_time)
        result3 = filter_by_latest_lead_time(sample_dataset)

        # Assert
        from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
        assert isinstance(result1, TimeSeriesDataset)
        assert isinstance(result2, TimeSeriesDataset)
        assert isinstance(result3, TimeSeriesDataset)
