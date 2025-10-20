# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta

import pandas as pd
import pytest

from openstef_core.datasets.versioned_timeseries_dataset_part import VersionedTimeSeriesPart
from openstef_core.types import AvailableAt, LeadTime


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


# Range filtering tests


def test_filter_by_range_basic_filtering(dataset_part: VersionedTimeSeriesPart):
    # Arrange
    start = datetime.fromisoformat("2023-01-01T10:00:00")
    end = datetime.fromisoformat("2023-01-01T13:00:00")

    # Act
    filtered_part = dataset_part.filter_by_range(start, end)
    window = filtered_part.select_version(available_before=None)

    # Assert
    assert len(window.data) == 3  # Should include timestamps at 10:00, 11:00, 12:00
    assert window.data.index[0] == start
    assert window.data.index[-1] == datetime.fromisoformat("2023-01-01T12:00:00")
    assert all(col in window.data.columns for col in ["value1", "value2"])
    assert "available_at" not in window.data.columns


def test_filter_by_range_with_availability_filtering(dataset_part: VersionedTimeSeriesPart):
    # Arrange
    start = datetime.fromisoformat("2023-01-01T10:00:00")
    end = datetime.fromisoformat("2023-01-01T15:00:00")
    available_before = datetime.fromisoformat("2023-01-01T13:00:00")

    # Act
    filtered_part = dataset_part.filter_by_range(start, end)
    window = filtered_part.select_version(available_before=available_before)

    # Assert
    # Data should be filtered by availability time
    assert len(window.data) == 3  # Only data available before 13:00
    assert window.data.index[0] == start
    assert window.data.index[-1] == datetime.fromisoformat("2023-01-01T12:00:00")


def test_filter_by_range_duplicates_keep_latest(dataset_part: VersionedTimeSeriesPart):
    # Arrange
    start = datetime.fromisoformat("2023-01-01T13:00:00")
    end = datetime.fromisoformat("2023-01-01T15:00:00")

    # Act
    filtered_part = dataset_part.filter_by_range(start, end)
    window = filtered_part.select_version(available_before=None)

    # Assert
    assert len(window.data) == 2  # 13:00 and 14:00
    # For the duplicate 14:00 timestamp, should take the latest available (value1=55)
    assert window.data.loc[pd.Timestamp.fromisoformat("2023-01-01T14:00:00"), "value1"] == 55


def test_filter_by_range_reindex_with_missing_timestamps():
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

    gapped_dataset_part = VersionedTimeSeriesPart(
        data=data,
        sample_interval=timedelta(hours=1),
        timestamp_column="timestamp",
        available_at_column="available_at",
    )

    start = datetime.fromisoformat("2023-01-01T10:00:00")
    end = datetime.fromisoformat("2023-01-01T13:00:00")

    # Act
    filtered_part = gapped_dataset_part.filter_by_range(start, end)
    window = filtered_part.select_version(available_before=None)

    # Assert
    assert len(window.data) == 2  # Should include only existing 10:00, 12:00 (no reindexing in select_version)
    assert window.data.loc[pd.Timestamp.fromisoformat("2023-01-01T10:00:00"), "value"] == 10
    assert window.data.loc[pd.Timestamp.fromisoformat("2023-01-01T12:00:00"), "value"] == 30


# Available at filtering tests


@pytest.mark.parametrize(
    ("data", "available_at", "expected_values"),
    [
        pytest.param(
            pd.DataFrame({
                "timestamp": pd.to_datetime(["2023-01-09T00:00", "2023-01-10T00:00"]),
                "available_at": pd.to_datetime(["2023-01-08T06:00", "2023-01-09T06:00"]),
                "value": [1, 2],
            }),
            AvailableAt.from_string("D-1T06:00"),
            [1, 2],
            id="all_available",
        ),
        pytest.param(
            pd.DataFrame({
                "timestamp": pd.to_datetime(["2023-01-09T00:00", "2023-01-10T00:00"]),
                "available_at": pd.to_datetime(["2023-01-08T05:00", "2023-01-09T07:00"]),  # Second one is too late
                "value": [1, 2],
            }),
            AvailableAt.from_string("D-1T06:00"),
            [1],
            id="partial_available",
        ),
        pytest.param(
            pd.DataFrame({
                "timestamp": pd.to_datetime([
                    "2023-01-10T00:00",
                    "2023-01-10T00:00",
                    "2023-01-10T00:00",
                    "2023-01-11T00:00",
                ]),
                "available_at": pd.to_datetime([
                    "2023-01-08T06:00",
                    "2023-01-08T12:00",
                    "2023-01-09T05:00",
                    "2023-01-10T05:00",
                ]),
                "value": [1, 2, 3, 4],
            }),
            AvailableAt.from_string("D-1T06:00"),
            [3, 4],
            id="duplicate_timestamps_latest_available",
        ),
        pytest.param(
            pd.DataFrame({
                "timestamp": pd.to_datetime(["2023-01-09T00:00", "2023-01-10T00:00"]),
                "available_at": pd.to_datetime(["2023-01-07T12:00", "2023-01-08T12:00"]),
                "value": [1, 2],
            }),
            AvailableAt.from_string("D-2T12:00"),
            [1, 2],
            id="different_lag_all_available",
        ),
    ],
)
def test_filter_by_available_at(data: pd.DataFrame, available_at: AvailableAt, expected_values: list[int]):
    # Arrange
    dataset_part = VersionedTimeSeriesPart(
        data=data,
        sample_interval=timedelta(days=1),
    )

    # Act
    filtered_part = dataset_part.filter_by_available_at(available_at)
    result = filtered_part.select_version(available_before=None)

    # Assert
    assert result.data["value"].tolist() == expected_values
    assert "available_at" not in result.data.columns


# Lead time filtering tests


@pytest.mark.parametrize(
    ("data", "lead_time", "expected_values"),
    [
        pytest.param(
            pd.DataFrame({
                "timestamp": pd.to_datetime(["2023-01-09T00:00", "2023-01-10T00:00"]),
                "available_at": pd.to_datetime(["2023-01-08T00:00", "2023-01-09T00:00"]),
                "value": [1, 2],
            }),
            LeadTime.from_string("PT24H"),
            [1, 2],
            id="all_available",
        ),
        pytest.param(
            pd.DataFrame({
                "timestamp": pd.to_datetime(["2023-01-10T00:00", "2023-01-11T00:00"]),
                "available_at": pd.to_datetime(["2023-01-08T00:00", "2023-01-10T00:00"]),
                "value": [1, 2],
            }),
            LeadTime.from_string("PT48H"),
            [1],
            id="partial_available",
        ),
        pytest.param(
            pd.DataFrame({
                "timestamp": pd.to_datetime([
                    "2023-01-10T00:00",
                    "2023-01-10T00:00",
                    "2023-01-10T00:00",
                    "2023-01-11T00:00",
                ]),
                "available_at": pd.to_datetime([
                    "2023-01-08T00:00",
                    "2023-01-09T00:00",
                    "2023-01-10T00:00",
                    "2023-01-10T00:00",
                ]),
                "value": [1, 2, 3, 4],
            }),
            LeadTime.from_string("PT24H"),
            [2, 4],
            id="duplicate_timestamps_with_varying_availability",
        ),
    ],
)
def test_filter_by_lead_time(data: pd.DataFrame, lead_time: LeadTime, expected_values: list[int]):
    # Arrange
    dataset_part = VersionedTimeSeriesPart(
        data=data,
        sample_interval=timedelta(days=1),
    )

    # Act
    filtered_part = dataset_part.filter_by_lead_time(lead_time)
    result = filtered_part.select_version(available_before=None)

    # Assert
    assert result.data["value"].tolist() == expected_values
    assert "available_at" not in result.feature_names


# Version selection tests


def test_select_version_basic():
    # Arrange
    dataset_part = VersionedTimeSeriesPart(
        data=pd.DataFrame({
            "timestamp": pd.to_datetime(["2023-01-01T00:00", "2023-01-01T00:00", "2023-01-02T00:00"]),
            "available_at": pd.to_datetime(["2022-12-30T00:00", "2022-12-31T00:00", "2023-01-01T00:00"]),
            "value": [1, 2, 3],
        }),
        sample_interval=timedelta(days=1),
    )

    # Act
    result = dataset_part.select_version(available_before=None)

    # Assert
    # select_version uses keep="last" for duplicates, so latest value (2) should be kept for 2023-01-01
    assert result.data["value"].tolist() == [2, 3]
    assert "available_at" not in result.feature_names


def test_select_version_with_availability_cutoff():
    # Arrange
    dataset_part = VersionedTimeSeriesPart(
        data=pd.DataFrame({
            "timestamp": pd.to_datetime(["2023-01-01T00:00", "2023-01-01T00:00", "2023-01-02T00:00"]),
            "available_at": pd.to_datetime(["2022-12-30T00:00", "2022-12-31T00:00", "2023-01-01T00:00"]),
            "value": [1, 2, 3],
        }),
        sample_interval=timedelta(days=1),
    )

    # Act
    result = dataset_part.select_version(available_before=pd.Timestamp("2022-12-30T12:00"))

    # Assert
    # Only the first record should be available before the cutoff
    assert result.data["value"].tolist() == [1]
    assert "available_at" not in result.feature_names
