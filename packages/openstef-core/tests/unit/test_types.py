# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import pytest

from openstef_core.types import AvailableAt, LeadTime


@pytest.mark.parametrize(
    ("lead_time", "expected_string"),
    [
        pytest.param(timedelta(hours=18), "PT18H", id="PT18H"),
        pytest.param(timedelta(days=1, hours=12), "P1DT12H", id="P1DT12H"),
    ],
)
def test_lead_time_str(lead_time: timedelta, expected_string: str):
    # Act
    result = str(LeadTime(lead_time))

    # Assert
    assert result == expected_string


@pytest.mark.parametrize(
    "input_delta",
    [
        pytest.param(timedelta(days=1), id="days_only"),
        pytest.param(timedelta(hours=3), id="hours_only"),
        pytest.param(timedelta(minutes=30), id="minutes_only"),
        pytest.param(timedelta(days=5, hours=5, minutes=5), id="mixed"),
    ],
)
def test_lead_time_from_string_roundtrip(input_delta: timedelta):
    # Arrange
    original = LeadTime(input_delta)

    # Act
    str_repr = str(original)
    reconstructed = LeadTime.from_string(str_repr)

    # Assert
    assert reconstructed.value == original.value


@pytest.mark.parametrize(
    ("lag_from_day", "expected_string"),
    [
        pytest.param(timedelta(hours=18), "D-1T06:00", id="D-1T06:00"),
        pytest.param(timedelta(hours=12 + 24), "D-2T12:00", id="D-2T12:00"),
    ],
)
def test_available_at_str(lag_from_day: timedelta, expected_string: str):
    # Arrange
    available_at = AvailableAt(lag_from_day=lag_from_day)

    # Act
    result = str(available_at)

    # Assert
    assert result == expected_string


@pytest.mark.parametrize(
    "available_at",
    [
        pytest.param(AvailableAt(lag_from_day=timedelta(hours=18)), id="D-1T06:00"),
        pytest.param(AvailableAt(lag_from_day=timedelta(hours=12 + 24)), id="D-2T12:00"),
    ],
)
def test_available_at_from_string_roundtrip(available_at: AvailableAt):
    # Arrange
    original = available_at

    # Act
    str_repr = str(original)
    reconstructed = AvailableAt.from_string(str_repr)

    # Assert
    assert reconstructed.lag_from_day == original.lag_from_day
