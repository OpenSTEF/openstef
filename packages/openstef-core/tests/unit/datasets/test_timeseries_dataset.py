# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for TimeSeriesDataset parquet serialization."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.testing import create_timeseries_dataset
from openstef_core.types import AvailableAt, LeadTime


@pytest.fixture
def simple_dataset() -> TimeSeriesDataset:
    return TimeSeriesDataset(
        data=pd.DataFrame(
            data={
                "available_at": pd.to_datetime([
                    "2023-01-01T09:50:00",  # lead time = 10:00 - 09:50 = +10min
                    "2023-01-01T10:55:00",  # lead time = 11:00 - 10:55 = +5min
                    "2023-01-01T12:10:00",  # lead time = 12:00 - 12:10 = -10min
                    "2023-01-01T13:20:00",  # lead time = 13:00 - 13:20 = -20min
                    "2023-01-01T14:15:00",  # lead time = 14:00 - 14:15 = -15min
                    "2023-01-01T14:30:00",  # lead time = 14:00 - 14:30 = -30min
                ]),
                "value1": [10, 20, 30, 40, 50, 55],  # 55 should override 50 for 14:00
            },
            index=pd.to_datetime([
                "2023-01-01T10:00:00",
                "2023-01-01T11:00:00",
                "2023-01-01T12:00:00",
                "2023-01-01T13:00:00",
                # Duplicate timestamp with different availability
                "2023-01-01T14:00:00",
                "2023-01-01T14:00:00",
            ]),
        ),
        sample_interval=timedelta(hours=1),
    )


@pytest.mark.parametrize(
    ("horizons_input", "expected_horizons"),
    [
        pytest.param(
            [timedelta(hours=1), timedelta(hours=1), timedelta(hours=2), timedelta(hours=2)],
            [LeadTime(timedelta(hours=1)), LeadTime(timedelta(hours=2))],
            id="multiple_unique_horizons",
        ),
        pytest.param(
            [timedelta(hours=3)] * 4,
            [LeadTime(timedelta(hours=3))],
            id="single_unique_horizon",
        ),
    ],
)
def test_horizons_property_parses_correctly(horizons_input: list[timedelta], expected_horizons: list[LeadTime]):
    """Horizons property returns unique LeadTime objects parsed from horizon column."""
    # Arrange
    index = pd.to_datetime(["2023-01-01T10:00:00", "2023-01-01T11:00:00", "2023-01-01T12:00:00", "2023-01-01T13:00:00"])
    horizons_series = pd.Series(horizons_input, index=index)
    dataset = create_timeseries_dataset(
        values=[10, 20, 30, 40],
        index=index,
        horizons=horizons_series,
    )

    # Act
    parsed_horizons = dataset.horizons

    # Assert - horizons property contains unique LeadTime objects in sorted order
    assert parsed_horizons is not None
    assert len(parsed_horizons) == len(expected_horizons)
    assert set(parsed_horizons) == set(expected_horizons)


@pytest.mark.parametrize(
    ("start", "end", "expected_values"),
    [
        (datetime.fromisoformat("2023-01-01T10:00:00"), datetime.fromisoformat("2023-01-01T13:00:00"), [10, 20, 30]),
        (None, datetime.fromisoformat("2023-01-01T13:00:00"), [10, 20, 30]),
        (datetime.fromisoformat("2023-01-01T11:00:00"), None, [20, 30, 40, 55, 50]),
    ],
)
def test_filter_by_range(simple_dataset: TimeSeriesDataset, start: datetime, end: datetime, expected_values: list[int]):
    # Act
    filtered = simple_dataset.filter_by_range(start, end)

    # Assert
    assert list(filtered.data["value1"]) == expected_values


@pytest.mark.parametrize(
    ("available_before", "expected_values"),
    [
        (datetime.fromisoformat("2023-01-01T12:15:00"), [10, 20, 30]),
        (datetime.fromisoformat("2023-01-01T14:15:00"), [10, 20, 30, 40, 50]),
        (datetime.fromisoformat("2023-01-01T14:30:00"), [10, 20, 30, 40, 55, 50]),
    ],
)
def test_filter_by_available_before(
    simple_dataset: TimeSeriesDataset, available_before: datetime, expected_values: list[int]
):
    # Act
    filtered = simple_dataset.filter_by_available_before(available_before)

    # Assert
    assert list(filtered.data["value1"]) == expected_values


@pytest.mark.parametrize(
    ("available_at", "expected_values"),
    [
        (AvailableAt(timedelta(hours=-13)), [10, 20, 30]),
        (AvailableAt(timedelta(hours=-15)), [10, 20, 30, 40, 55, 50]),
    ],
)
def test_filter_by_available_at(
    simple_dataset: TimeSeriesDataset, available_at: AvailableAt, expected_values: list[int]
):
    # Act
    filtered = simple_dataset.filter_by_available_at(available_at)

    # Assert
    assert list(filtered.data["value1"]) == expected_values


@pytest.mark.parametrize(
    ("lead_time", "expected_values"),
    [
        # Keep rows where lead_time >= -15min: keeps +10min, +5min, -10min, -15min but NOT -20min, -30min
        (LeadTime(timedelta(minutes=-15)), [10, 20, 30, 50]),
        # Keep rows where lead_time >= 0min: keeps only +10min, +5min (positive lead times)
        (LeadTime(timedelta(minutes=0)), [10, 20]),
        # Keep rows where lead_time >= +6min: keeps only +10min
        (LeadTime(timedelta(minutes=6)), [10]),
    ],
)
def test_filter_by_lead_time(simple_dataset: TimeSeriesDataset, lead_time: LeadTime, expected_values: list[int]):
    # Act
    filtered = simple_dataset.filter_by_lead_time(lead_time)

    # Assert
    assert list(filtered.data["value1"]) == expected_values


def test_select_version(simple_dataset: TimeSeriesDataset):
    # Act
    selected = simple_dataset.select_version()

    # Assert
    assert list(selected.data["value1"]) == [10, 20, 30, 40, 55]
    assert "available_at" not in selected.data.columns
    assert not selected.is_versioned


@pytest.fixture
def empty_dataset() -> TimeSeriesDataset:
    return TimeSeriesDataset(
        data=pd.DataFrame(
            {
                "value1": pd.Series([], dtype="float64"),
                "available_at": pd.Series([], dtype="datetime64[ns, UTC]"),
            },
            index=pd.DatetimeIndex([], tz="UTC"),
        ),
        sample_interval=timedelta(hours=1),
    )


def test_select_version_empty(empty_dataset: TimeSeriesDataset):
    # Act
    selected = empty_dataset.select_version()

    # Assert
    assert len(selected.data) == 0
    assert not selected.is_versioned


def test_available_at_and_lead_time_series_equivalence():
    # Arrange
    index = pd.to_datetime(["2023-01-01T10:00:00", "2023-01-01T11:00:00"])
    available_at_dataset = create_timeseries_dataset(
        values=[10, 20],
        index=index,
        available_ats=cast(pd.Series, index) - timedelta(hours=5),
    )

    horizon_dataset = create_timeseries_dataset(
        values=[10, 20],
        index=index,
        horizons=pd.to_timedelta([5, 5], unit="h"),  # type: ignore
    )

    # Assert
    assert available_at_dataset.available_at_series is not None
    assert horizon_dataset.available_at_series is not None
    assert available_at_dataset.lead_time_series is not None
    assert horizon_dataset.lead_time_series is not None

    pd.testing.assert_series_equal(
        available_at_dataset.available_at_series,
        horizon_dataset.available_at_series,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        available_at_dataset.lead_time_series,
        horizon_dataset.lead_time_series,
        check_names=False,
    )


@pytest.mark.parametrize(
    ("horizon", "expected_values"),
    [
        pytest.param(
            LeadTime(timedelta(hours=1)),
            [10, 20, 30, 40, 50, 55],
            id="select_1h_horizon",
        ),
        pytest.param(
            LeadTime(timedelta(hours=2)),
            [],
            id="select_2h_horizon_no_match",
        ),
    ],
)
def test_select_horizon(horizon: LeadTime, expected_values: list[int]):
    """Filter dataset to include only data for a specific forecast horizon."""
    # Arrange
    index = pd.to_datetime([
        "2023-01-01T10:00:00",
        "2023-01-01T11:00:00",
        "2023-01-01T12:00:00",
        "2023-01-01T13:00:00",
        "2023-01-01T14:00:00",
        "2023-01-01T14:00:00",
    ])
    horizons = pd.Series([timedelta(hours=1)] * 6, index=index)
    dataset = create_timeseries_dataset(
        value1=[10, 20, 30, 40, 50, 55],
        index=index,
        horizons=horizons,
    )

    # Act
    selected = dataset.select_horizon(horizon)

    # Assert - selected dataset contains only rows matching the horizon
    assert list(selected.data["value1"]) == expected_values


def test_select_horizon__no_horizons():
    """Return original dataset when it has no horizons column."""
    # Arrange
    data = pd.DataFrame(
        {
            "value1": [10, 20, 30],
            "available_at": pd.to_datetime(["2023-01-01T10:15:00", "2023-01-01T11:15:00", "2023-01-01T12:15:00"]),
        },
        index=pd.to_datetime(["2023-01-01T10:00:00", "2023-01-01T11:00:00", "2023-01-01T12:00:00"]),
    )
    dataset = TimeSeriesDataset(data=data, sample_interval=timedelta(hours=1))

    # Act
    selected = dataset.select_horizon(LeadTime(timedelta(hours=1)))

    # Assert - returns the same dataset when no horizons exist
    assert list(selected.data["value1"]) == [10, 20, 30]


def test_parquet_roundtrip(tmp_path: Path):
    """Test that parquet roundtrip preserves data, metadata, and versioning information."""
    # Arrange - create dataset with available_at versioning
    data = pd.DataFrame(
        {
            "temperature": [20.0, 21.5, 19.0],
            "wind_speed": [5.0, 6.2, 4.8],
            "load": [100.0, 110.0, 105.0],
            "available_at": pd.date_range("2025-01-01T10:05", periods=3, freq="1h"),
        },
        index=pd.date_range("2025-01-01T10:00", periods=3, freq="1h"),
    )
    original = TimeSeriesDataset(data, sample_interval=timedelta(hours=1))

    # Act
    parquet_path = tmp_path / "timeseries_dataset.parquet"
    original.to_parquet(parquet_path)
    loaded = TimeSeriesDataset.read_parquet(parquet_path)

    # Assert - metadata preserved
    assert loaded.sample_interval == original.sample_interval
    assert loaded.is_versioned == original.is_versioned
    assert loaded.horizon_column == original.horizon_column
    assert loaded.available_at_column == original.available_at_column
    assert loaded.feature_names == original.feature_names

    # Assert - data preserved
    pd.testing.assert_frame_equal(
        loaded.data.sort_index(),
        original.data.sort_index(),
        check_freq=False,
    )
