# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import VersionedTimeSeriesDataset
from openstef_core.exceptions import MissingColumnsError


@pytest.fixture
def sample_data() -> pd.DataFrame:
    data = {
        "timestamp": [
            datetime.fromisoformat("2023-01-01T10:00:00"),
            datetime.fromisoformat("2023-01-01T11:00:00"),
            datetime.fromisoformat("2023-01-01T12:00:00"),
            datetime.fromisoformat("2023-01-01T13:00:00"),
            # Duplicate timestamp with different availability
            datetime.fromisoformat("2023-01-01T14:00:00"),
            datetime.fromisoformat("2023-01-01T14:00:00"),
        ],
        "available_at": [
            datetime.fromisoformat("2023-01-01T10:15:00"),
            datetime.fromisoformat("2023-01-01T11:15:00"),
            datetime.fromisoformat("2023-01-01T12:15:00"),
            datetime.fromisoformat("2023-01-01T13:15:00"),
            datetime.fromisoformat("2023-01-01T14:15:00"),
            datetime.fromisoformat("2023-01-01T14:30:00"),  # More recent version
        ],
        "value1": [10, 20, 30, 40, 50, 55],  # 55 should override 50 for 14:00
        "value2": [100, 200, 300, 400, 500, 550],
    }
    return pd.DataFrame(data)


@pytest.fixture
def dataset(sample_data: pd.DataFrame) -> VersionedTimeSeriesDataset:
    return VersionedTimeSeriesDataset(
        data=sample_data,
        sample_interval=timedelta(hours=1),
        timestamp_column="timestamp",
        available_at_column="available_at",
    )


def test_initialization_with_valid_data(sample_data: pd.DataFrame):
    # Arrange
    sample_interval = timedelta(hours=1)

    # Act
    dataset = VersionedTimeSeriesDataset(
        data=sample_data,
        sample_interval=sample_interval,
        timestamp_column="timestamp",
        available_at_column="available_at",
    )

    # Assert
    assert dataset.sample_interval == sample_interval
    assert dataset._timestamp_column == "timestamp"
    assert dataset._available_at_column == "available_at"
    assert isinstance(dataset.index, pd.DatetimeIndex)


@pytest.mark.parametrize(
    "missing_column",
    [
        pytest.param("timestamp", id="missing_timestamp"),
        pytest.param("available_at", id="missing_available_at"),
    ],
)
def test_initialization_with_missing_columns(sample_data: pd.DataFrame, missing_column: str):
    # Arrange
    data = sample_data.drop(columns=[missing_column])

    # Act & Assert
    with pytest.raises(MissingColumnsError):
        VersionedTimeSeriesDataset(
            data=data,
            sample_interval=timedelta(hours=1),
            timestamp_column="timestamp",
            available_at_column="available_at",
        )


def test_custom_column_names():
    # Arrange
    data = pd.DataFrame({
        "custom_ts": [datetime.fromisoformat("2023-01-01T10:00:00"), datetime.fromisoformat("2023-01-01T11:00:00")],
        "custom_avail": [datetime.fromisoformat("2023-01-01T10:15:00"), datetime.fromisoformat("2023-01-01T11:15:00")],
        "value": [10, 20],
    })

    # Act
    dataset = VersionedTimeSeriesDataset(
        data=data,
        sample_interval=timedelta(hours=1),
        timestamp_column="custom_ts",
        available_at_column="custom_avail",
    )

    # Assert
    assert dataset._timestamp_column == "custom_ts"
    assert dataset._available_at_column == "custom_avail"
    assert dataset.feature_names == ["value"]


def test_feature_names_property(dataset: VersionedTimeSeriesDataset):
    # Arrange
    expected_features = ["value1", "value2"]

    # Act
    features = dataset.feature_names

    # Assert
    assert sorted(features) == sorted(expected_features)


def test_index_property(dataset: VersionedTimeSeriesDataset, sample_data: pd.DataFrame):
    # Arrange
    expected_index = pd.DatetimeIndex(sample_data["timestamp"])

    # Act
    index = dataset.index

    # Assert
    pd.testing.assert_index_equal(index, expected_index)


def test_data_property(dataset: VersionedTimeSeriesDataset, sample_data: pd.DataFrame):
    # Arrange
    expected_data = sample_data

    # Act
    data = dataset.data

    # Assert
    pd.testing.assert_frame_equal(data, expected_data)


def test_get_window_basic_filtering(dataset: VersionedTimeSeriesDataset):
    # Arrange
    start = datetime.fromisoformat("2023-01-01T10:00:00")
    end = datetime.fromisoformat("2023-01-01T13:00:00")

    # Act
    window = dataset.get_window(start, end)

    # Assert
    assert len(window) == 3  # Should include timestamps at 10:00, 11:00, 12:00
    assert window.index[0] == start
    assert window.index[-1] == datetime.fromisoformat("2023-01-01T12:00:00")
    assert all(col in window.columns for col in ["value1", "value2"])
    assert "available_at" not in window.columns


def test_get_window_with_availability_filtering(dataset: VersionedTimeSeriesDataset):
    # Arrange
    start = datetime.fromisoformat("2023-01-01T10:00:00")
    end = datetime.fromisoformat("2023-01-01T15:00:00")
    available_before = datetime.fromisoformat("2023-01-01T13:00:00")

    expected = pd.DataFrame(
        {
            "value1": [10, 20, 30, np.nan, np.nan],
        },
        index=pd.DatetimeIndex([
            "2023-01-01T10:00:00",
            "2023-01-01T11:00:00",
            "2023-01-01T12:00:00",
            "2023-01-01T13:00:00",  # Data point is after cutoff
            "2023-01-01T14:00:00",
        ]),
    )

    # Act
    window = dataset.get_window(start, end, available_before)

    # Assert
    pd.testing.assert_frame_equal(window[["value1"]], expected, check_freq=False)


def test_get_window_duplicates_keep_latest(dataset: VersionedTimeSeriesDataset):
    # Arrange
    start = datetime.fromisoformat("2023-01-01T13:00:00")
    end = datetime.fromisoformat("2023-01-01T15:00:00")

    # Act
    window = dataset.get_window(start, end)

    # Assert
    assert len(window) == 2  # 13:00 and 14:00
    # For the duplicate 14:00 timestamp, should take the latest available (value1=55)
    assert window.loc[pd.Timestamp.fromisoformat("2023-01-01T14:00:00"), "value1"] == 55


def test_get_window_reindex_with_missing_timestamps():
    # Arrange
    data = pd.DataFrame({
        "timestamp": [
            datetime.fromisoformat("2023-01-01T10:00:00"),
            # Gap at 11:00
            datetime.fromisoformat("2023-01-01T12:00:00"),
        ],
        "available_at": [
            datetime.fromisoformat("2023-01-01T10:15:00"),
            datetime.fromisoformat("2023-01-01T12:15:00"),
        ],
        "value": [10, 30],
    })

    gapped_dataset = VersionedTimeSeriesDataset(
        data=data,
        sample_interval=timedelta(hours=1),
        timestamp_column="timestamp",
        available_at_column="available_at",
    )

    start = datetime.fromisoformat("2023-01-01T10:00:00")
    end = datetime.fromisoformat("2023-01-01T13:00:00")

    # Act
    window = gapped_dataset.get_window(start, end)

    # Assert
    assert len(window) == 3  # Should include 10:00, 11:00, 12:00
    assert pd.isna(window.loc[pd.Timestamp.fromisoformat("2023-01-01T11:00:00"), "value"])  # Missing timestamp
    assert window.loc[pd.Timestamp.fromisoformat("2023-01-01T10:00:00"), "value"] == 10
    assert window.loc[pd.Timestamp.fromisoformat("2023-01-01T12:00:00"), "value"] == 30


def test_from_parquet(tmp_path: Path):
    # Arrange
    data = pd.DataFrame({
        "timestamp": [datetime.fromisoformat("2023-01-01T10:00:00"), datetime.fromisoformat("2023-01-01T11:00:00")],
        "available_at": [datetime.fromisoformat("2023-01-01T10:15:00"), datetime.fromisoformat("2023-01-01T11:15:00")],
        "value": [10, 20],
    })

    # Add required attributes
    data.attrs["sample_interval"] = "PT1H"
    data.attrs["timestamp_column"] = "timestamp"
    data.attrs["available_at_column"] = "available_at"

    parquet_path = tmp_path / "test_data.parquet"
    data.to_parquet(parquet_path)

    # Act
    dataset = VersionedTimeSeriesDataset.from_parquet(path=parquet_path)

    # Assert
    assert dataset.sample_interval == timedelta(hours=1)
    assert dataset._timestamp_column == "timestamp"
    assert dataset._available_at_column == "available_at"
    pd.testing.assert_frame_equal(dataset.data, data)
