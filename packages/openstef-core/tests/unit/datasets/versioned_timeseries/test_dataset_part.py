# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from openstef_core.datasets.versioned_timeseries.dataset_part import VersionedTimeSeriesPart
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
def dataset_part(sample_data: pd.DataFrame) -> VersionedTimeSeriesPart:
    return VersionedTimeSeriesPart(
        data=sample_data,
        sample_interval=timedelta(hours=1),
        timestamp_column="timestamp",
        available_at_column="available_at",
    )


def test_initialization_with_valid_data(sample_data: pd.DataFrame):
    # Arrange
    sample_interval = timedelta(hours=1)

    # Act
    dataset_part = VersionedTimeSeriesPart(
        data=sample_data,
        sample_interval=sample_interval,
        timestamp_column="timestamp",
        available_at_column="available_at",
    )

    # Assert
    assert dataset_part.sample_interval == sample_interval
    assert dataset_part.timestamp_column == "timestamp"
    assert dataset_part.available_at_column == "available_at"
    assert isinstance(dataset_part.index, pd.DatetimeIndex)


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
        VersionedTimeSeriesPart(
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
    dataset_part = VersionedTimeSeriesPart(
        data=data,
        sample_interval=timedelta(hours=1),
        timestamp_column="custom_ts",
        available_at_column="custom_avail",
    )

    # Assert
    assert dataset_part.timestamp_column == "custom_ts"
    assert dataset_part.available_at_column == "custom_avail"
    assert dataset_part.feature_names == ["value"]


def test_feature_names_property(dataset_part: VersionedTimeSeriesPart):
    # Arrange
    expected_features = ["value1", "value2"]

    # Act
    features = dataset_part.feature_names

    # Assert
    assert sorted(features) == sorted(expected_features)


def test_index_property(dataset_part: VersionedTimeSeriesPart, sample_data: pd.DataFrame):
    # Arrange
    expected_index = pd.DatetimeIndex(sample_data["timestamp"])

    # Act
    index = dataset_part.index

    # Assert
    pd.testing.assert_index_equal(index, expected_index)


def test_to_parquet_and_read_parquet(tmp_path: Path):
    # Arrange
    data = pd.DataFrame({
        "timestamp": [datetime.fromisoformat("2023-01-01T10:00:00"), datetime.fromisoformat("2023-01-01T11:00:00")],
        "available_at": [datetime.fromisoformat("2023-01-01T10:15:00"), datetime.fromisoformat("2023-01-01T11:15:00")],
        "value": [10, 20],
    })

    dataset_part = VersionedTimeSeriesPart(
        data=data,
        sample_interval=timedelta(hours=1),
        timestamp_column="timestamp",
        available_at_column="available_at",
    )

    parquet_path = tmp_path / "test_data.parquet"

    # Act
    dataset_part.to_parquet(parquet_path)
    loaded_part = VersionedTimeSeriesPart.read_parquet(parquet_path)

    # Assert
    assert loaded_part.sample_interval == timedelta(hours=1)
    assert loaded_part.timestamp_column == "timestamp"
    assert loaded_part.available_at_column == "available_at"
    pd.testing.assert_frame_equal(loaded_part.data, dataset_part.data)
