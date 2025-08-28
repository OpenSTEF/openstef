# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for datetime utilities."""

from datetime import UTC, datetime, time, timedelta, timezone
from typing import Literal

import pytest

from openstef_core.utils.datetime import align_datetime, align_datetime_to_time


@pytest.mark.parametrize(
    ("timestamp", "interval", "mode", "expected"),
    [
        pytest.param(
            datetime.fromisoformat("2020-01-01T10:15:00"),
            timedelta(hours=1),
            "ceil",
            datetime.fromisoformat("2020-01-01T11:00:00"),
            id="ceil_to_next_hour",
        ),
        pytest.param(
            datetime.fromisoformat("2020-01-01T10:00:00"),
            timedelta(hours=1),
            "ceil",
            datetime.fromisoformat("2020-01-01T10:00:00"),
            id="ceil_already_aligned",
        ),
        pytest.param(
            datetime.fromisoformat("2020-01-01T10:15:00"),
            timedelta(hours=1),
            "floor",
            datetime.fromisoformat("2020-01-01T10:00:00"),
            id="floor_to_previous_hour",
        ),
        pytest.param(
            datetime.fromisoformat("2020-01-01T10:00:00"),
            timedelta(hours=1),
            "floor",
            datetime.fromisoformat("2020-01-01T10:00:00"),
            id="floor_already_aligned",
        ),
        pytest.param(
            datetime.fromisoformat("2020-01-01T10:07:30"),
            timedelta(minutes=15),
            "ceil",
            datetime.fromisoformat("2020-01-01T10:15:00"),
            id="ceil_15_minute_intervals",
        ),
        pytest.param(
            datetime.fromisoformat("2020-01-01T10:07:30"),
            timedelta(minutes=15),
            "floor",
            datetime.fromisoformat("2020-01-01T10:00:00"),
            id="floor_15_minute_intervals",
        ),
    ],
)
def test_align_datetime_naive(
    timestamp: datetime, interval: timedelta, mode: Literal["ceil", "floor"], expected: datetime
):
    # Arrange / Act
    result = align_datetime(timestamp, interval, mode)

    # Assert
    assert result == expected
    assert result.tzinfo is None


def test_align_datetime_with_timezone():
    # Arrange
    tz = UTC
    timestamp = datetime.fromisoformat("2020-01-01T10:15:00+00:00")
    interval = timedelta(hours=1)
    expected = datetime.fromisoformat("2020-01-01T11:00:00+00:00")

    # Act
    result = align_datetime(timestamp, interval, "ceil")

    # Assert
    assert result == expected
    assert result.tzinfo == tz


def test_align_datetime_preserves_timezone():
    # Arrange
    tz = timezone(timedelta(hours=2))
    timestamp = datetime(2020, 1, 1, 10, 15, 0, tzinfo=tz)
    interval = timedelta(hours=1)

    # Act
    result = align_datetime(timestamp, interval, "ceil")

    # Assert
    assert result.tzinfo == tz


def test_align_datetime_invalid_mode():
    # Arrange
    timestamp = datetime.fromisoformat("2020-01-01T10:15:00")
    interval = timedelta(hours=1)

    # Act / Assert
    with pytest.raises(ValueError, match="Unknown alignment mode: invalid"):
        align_datetime(timestamp, interval, "invalid")  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("timestamp_str", "align_time_str", "mode", "expected_str"),
    [
        pytest.param("2020-01-01T10:15:00", "09:00:00", "ceil", "2020-01-02T09:00:00", id="ceil_naive_next_day"),
        pytest.param("2020-01-01T10:15:00", "09:00:00", "floor", "2020-01-01T09:00:00", id="floor_naive_same_day"),
        pytest.param("2020-01-01T08:30:00", "09:00:00", "ceil", "2020-01-01T09:00:00", id="ceil_naive_same_day"),
        pytest.param("2020-01-01T10:15:00", "11:30:00", "floor", "2019-12-31T11:30:00", id="floor_naive_previous_day"),
        pytest.param("2020-01-01T09:00:00", "09:00:00", "ceil", "2020-01-01T09:00:00", id="ceil_naive_exact_match"),
        pytest.param(
            "2020-01-01T10:15:00+00:00", "09:00:00", "ceil", "2020-01-02T09:00:00+00:00", id="ceil_utc_naive_time"
        ),
        pytest.param(
            "2020-01-01T08:30:00+02:00",
            "09:00:00+01:00",
            "ceil",
            "2020-01-01T10:00:00+02:00",
            id="ceil_different_timezones",
        ),
        pytest.param(
            "2020-01-01T10:15:00+02:00",
            "09:00:00+01:00",
            "floor",
            "2020-01-01T10:00:00+02:00",
            id="floor_different_timezones",
        ),
        pytest.param(
            "2020-01-01T10:15:00", "09:00:00+00:00", "ceil", "2020-01-02T09:00:00", id="ceil_naive_timestamp_tz_time"
        ),
        pytest.param(
            "2020-01-01T10:15:00+00:00", "09:00:00+00:00", "ceil", "2020-01-02T09:00:00+00:00", id="ceil_same_timezone"
        ),
    ],
)
def test_align_datetime_to_time_returns_correct_alignment(
    timestamp_str: str, align_time_str: str, mode: Literal["ceil", "floor"], expected_str: str
):
    # Arrange
    timestamp = datetime.fromisoformat(timestamp_str)
    align_time = time.fromisoformat(align_time_str)
    expected = datetime.fromisoformat(expected_str)

    # Act
    result = align_datetime_to_time(timestamp, align_time, mode)

    # Assert
    # Check that the result matches expected datetime
    assert result == expected
    # Check that timezone is preserved from original timestamp
    assert result.tzinfo == timestamp.tzinfo
