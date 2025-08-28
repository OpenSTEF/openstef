# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import pandas as pd
import pytest

from openstef_core.datasets import VersionedTimeSeriesDataset
from openstef_core.datasets.versioned_timeseries import (
    filter_by_available_at,
    filter_by_latest_lead_time,
    filter_by_lead_time,
)
from openstef_core.types import AvailableAt, LeadTime


@pytest.mark.parametrize(
    ("data", "available_at", "expected_values"),
    [
        pytest.param(
            pd.DataFrame({
                "timestamp": pd.to_datetime(["2023-01-09", "2023-01-10"]),
                "available_at": pd.to_datetime(["2023-01-08T06:00", "2023-01-09T06:00"]),
                "value": [1, 2],
            }),
            AvailableAt.from_string("D-1T06:00"),
            [1, 2],
            id="all_available",
        ),
        pytest.param(
            pd.DataFrame({
                "timestamp": pd.to_datetime(["2023-01-09", "2023-01-10"]),
                "available_at": pd.to_datetime(["2023-01-08T05:00", "2023-01-09T07:00"]),  # Second one is too late
                "value": [1, 2],
            }),
            AvailableAt.from_string("D-1T06:00"),
            [1],
            id="partial_available",
        ),
        pytest.param(
            pd.DataFrame({
                "timestamp": pd.to_datetime(["2023-01-10", "2023-01-10", "2023-01-10", "2023-01-11"]),
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
                "timestamp": pd.to_datetime(["2023-01-09", "2023-01-10"]),
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
    dataset = VersionedTimeSeriesDataset(
        data=data,
        sample_interval=timedelta(days=1),
    )

    # Act
    result = filter_by_available_at(dataset, available_at)

    # Assert
    assert result.data.value.tolist() == expected_values
    assert "available_at" not in result.data.columns


@pytest.mark.parametrize(
    ("data", "lead_time", "expected_values"),
    [
        pytest.param(
            pd.DataFrame({
                "timestamp": pd.to_datetime(["2023-01-09", "2023-01-10"]),
                "available_at": pd.to_datetime(["2023-01-08", "2023-01-09"]),
                "value": [1, 2],
            }),
            LeadTime.from_string("PT24H"),
            [1, 2],
            id="all_available",
        ),
        pytest.param(
            pd.DataFrame({
                "timestamp": pd.to_datetime(["2023-01-10", "2023-01-11"]),
                "available_at": pd.to_datetime(["2023-01-08", "2023-01-10"]),
                "value": [1, 2],
            }),
            LeadTime.from_string("PT48H"),
            [1],
            id="partial_available",
        ),
        pytest.param(
            pd.DataFrame({
                "timestamp": pd.to_datetime(["2023-01-10", "2023-01-10", "2023-01-10", "2023-01-11"]),
                "available_at": pd.to_datetime(["2023-01-08", "2023-01-09", "2023-01-10", "2023-01-10"]),
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
    dataset = VersionedTimeSeriesDataset(
        data=data,
        sample_interval=timedelta(days=1),
    )

    # Act
    result = filter_by_lead_time(dataset, lead_time)

    # Assert
    assert result.data.value.tolist() == expected_values
    assert "available_at" not in result.feature_names


def test_filter_by_latest_lead_time():
    # Arrange
    dataset = VersionedTimeSeriesDataset(
        data=pd.DataFrame({
            "timestamp": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02"]),
            "available_at": pd.to_datetime(["2022-12-30", "2022-12-31", "2023-01-01"]),
            "value": [1, 2, 3],
        }),
        sample_interval=timedelta(days=1),
    )

    # Act
    result = filter_by_latest_lead_time(dataset)

    # Assert
    assert result.data.value.tolist() == [2, 3]
    assert "available_at" not in result.feature_names
