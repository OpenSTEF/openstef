# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, time, timedelta

import pytest
import pytz

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
        pytest.param(timedelta(hours=18), "D-1T0600", id="D-1T0600"),
        pytest.param(timedelta(hours=12 + 24), "D-2T1200", id="D-2T1200"),
    ],
)
def test_available_at_str(lag_from_day: timedelta, expected_string: str):
    # Act
    result = str(AvailableAt(lag_from_day=lag_from_day))

    # Assert
    assert result == expected_string


@pytest.mark.parametrize(
    "available_at",
    [
        pytest.param(AvailableAt(lag_from_day=timedelta(hours=18)), id="D-1T0600"),
        pytest.param(AvailableAt(lag_from_day=timedelta(hours=12 + 24)), id="D-2T1200"),
    ],
)
def test_available_at_from_string_roundtrip(available_at: AvailableAt):
    # Act
    reconstructed = AvailableAt.from_string(str(available_at))

    # Assert
    assert reconstructed.lag_from_day == available_at.lag_from_day


def test_available_at_from_string_rejects_positive_days_part():
    with pytest.raises(ValueError, match="Day offset must be negative or zero"):
        AvailableAt.from_string("D1T0600")


_AMS = pytz.timezone("Europe/Amsterdam")


@pytest.mark.parametrize(
    ("available_at", "reference_date", "output_tz", "expected", "expected_tz"),
    [
        pytest.param(
            AvailableAt(lag_from_day=timedelta(hours=18)),
            datetime(2026, 3, 6),  # noqa: DTZ001
            None,
            datetime(2026, 3, 5, 6, 0),  # noqa: DTZ001
            None,
            id="naive",
        ),
        pytest.param(
            AvailableAt(lag_from_day=timedelta(hours=36)),
            datetime(2026, 3, 6),  # noqa: DTZ001
            None,
            datetime(2026, 3, 4, 12, 0),  # noqa: DTZ001
            None,
            id="naive_D-2",
        ),
        pytest.param(
            AvailableAt(lag_from_day=timedelta(hours=18), tzinfo=pytz.UTC),
            datetime(2026, 3, 6, tzinfo=pytz.UTC),
            None,
            datetime(2026, 3, 5, 6, 0, tzinfo=pytz.UTC),
            "UTC",
            id="utc",
        ),
        pytest.param(
            AvailableAt(lag_from_day=timedelta(hours=18), tzinfo=_AMS),
            datetime(2026, 3, 6),  # noqa: DTZ001
            None,
            _AMS.localize(datetime(2026, 3, 5, 6, 0)),  # noqa: DTZ001
            "Europe/Amsterdam",
            id="named_tz",
        ),
        pytest.param(
            AvailableAt(lag_from_day=timedelta(hours=18), tzinfo=_AMS),
            datetime(2026, 3, 6, tzinfo=pytz.UTC),
            None,
            _AMS.localize(datetime(2026, 3, 5, 6, 0)),  # noqa: DTZ001
            "Europe/Amsterdam",
            id="own_tz_over_date_tz",
        ),
        pytest.param(
            AvailableAt(lag_from_day=timedelta(hours=18)),
            datetime(2026, 3, 6, tzinfo=pytz.UTC),
            None,
            datetime(2026, 3, 5, 6, 0, tzinfo=pytz.UTC),
            "UTC",
            id="fallback_to_date_tz",
        ),
        pytest.param(
            AvailableAt(lag_from_day=timedelta(hours=18), tzinfo=pytz.UTC),
            datetime(2026, 3, 6, tzinfo=pytz.UTC),
            _AMS,
            _AMS.localize(datetime(2026, 3, 5, 6, 0)),  # noqa: DTZ001
            "Europe/Amsterdam",
            id="output_tz_overrides",
        ),
    ],
)
def test_available_at_apply(
    available_at: AvailableAt,
    reference_date: datetime,
    output_tz: pytz.BaseTzInfo | None,
    expected: datetime,
    expected_tz: str | None,
):
    # Act
    result = available_at.apply(reference_date, output_tz=output_tz)

    # Assert
    assert result == expected
    if expected_tz is None:
        assert result.tzinfo is None
    else:
        assert str(result.tzinfo) == expected_tz


def test_available_at_day_offset():
    assert AvailableAt(lag_from_day=timedelta(hours=18)).day_offset == -1
    assert AvailableAt(lag_from_day=timedelta(hours=36)).day_offset == -2


def test_available_at_time_of_day():
    assert AvailableAt(lag_from_day=timedelta(hours=18)).time_of_day == time(6, 0)
    assert AvailableAt(lag_from_day=timedelta(hours=36)).time_of_day == time(12, 0)
    assert AvailableAt(lag_from_day=timedelta(hours=18, minutes=30)).time_of_day == time(5, 30)
