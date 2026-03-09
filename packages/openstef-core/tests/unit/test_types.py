# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import UTC, datetime, time, timedelta, timezone

import pandas as pd
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
    ("available_at", "expected_string"),
    [
        pytest.param(AvailableAt(day_offset=-1, time_of_day=time(6, 0)), "D-1T0600", id="no_tz"),
        pytest.param(AvailableAt(day_offset=-2, time_of_day=time(12, 0)), "D-2T1200", id="no_tz_D-2"),
        pytest.param(
            AvailableAt(day_offset=-1, time_of_day=time(6, 0), tzinfo=pytz.UTC),
            "D-1T0600[UTC]",
            id="pytz_utc",
        ),
        pytest.param(
            AvailableAt(day_offset=-1, time_of_day=time(6, 0), tzinfo=pytz.timezone("Europe/Amsterdam")),
            "D-1T0600[Europe/Amsterdam]",
            id="named_tz",
        ),
        pytest.param(
            AvailableAt(day_offset=-1, time_of_day=time(6, 0), tzinfo=UTC),
            "D-1T0600[UTC]",
            id="stdlib_utc",
        ),
    ],
)
def test_available_at_str(available_at: AvailableAt, expected_string: str):
    # Act
    result = str(available_at)

    # Assert
    assert result == expected_string


@pytest.mark.parametrize(
    "available_at",
    [
        pytest.param(AvailableAt(day_offset=-1, time_of_day=time(6, 0)), id="no_tz"),
        pytest.param(AvailableAt(day_offset=-2, time_of_day=time(12, 0)), id="no_tz_D-2"),
        pytest.param(
            AvailableAt(day_offset=-1, time_of_day=time(6, 0), tzinfo=pytz.UTC),
            id="pytz_utc",
        ),
        pytest.param(
            AvailableAt(day_offset=-1, time_of_day=time(6, 0), tzinfo=pytz.timezone("Europe/Amsterdam")),
            id="named_tz",
        ),
    ],
)
def test_available_at_from_string_roundtrip(available_at: AvailableAt):
    # Act
    reconstructed = AvailableAt.from_string(str(available_at))

    # Assert
    assert reconstructed.day_offset == available_at.day_offset
    assert reconstructed.time_of_day == available_at.time_of_day
    if available_at.tzinfo is None:
        assert reconstructed.tzinfo is None
    else:
        assert reconstructed.tzinfo is not None
        # Both should resolve to the same zone
        assert str(reconstructed.tzinfo) == str(available_at.tzinfo)


def test_available_at_from_string_legacy_colon_format():
    """The legacy DnTHH:MM format (with colon) should still be accepted."""
    at = AvailableAt.from_string("D-1T06:00")
    assert at.day_offset == -1
    assert at.time_of_day == time(6, 0)
    assert at.tzinfo is None


@pytest.mark.parametrize(
    "tz_str",
    [
        pytest.param("UTC", id="pytz_utc"),
        pytest.param("Europe/Amsterdam", id="named_tz"),
    ],
)
def test_available_at_from_string_legacy_colon_with_tz(tz_str: str):
    """Legacy colon format combined with timezone suffix."""
    at = AvailableAt.from_string(f"D-1T06:00[{tz_str}]")
    assert at.day_offset == -1
    assert at.time_of_day == time(6, 0)
    assert str(at.tzinfo) == tz_str


def test_available_at_from_string_rejects_positive_days_part():
    with pytest.raises(ValueError, match="Day offset must be negative or zero"):
        AvailableAt.from_string("D1T0600")


def test_available_at_rejects_positive_day_offset():
    with pytest.raises(ValueError, match="Day offset must be negative or zero"):
        AvailableAt(day_offset=1, time_of_day=time(6, 0))


def test_available_at_from_string_rejects_invalid():
    with pytest.raises(ValueError, match="Cannot convert"):
        AvailableAt.from_string("INVALID")


def test_available_at_from_string_rejects_trailing_garbage():
    with pytest.raises(ValueError, match="Cannot convert"):
        AvailableAt.from_string("D-1T0600INVALID")


def test_available_at_from_string_z_suffix():
    """'Z' suffix should parse as UTC."""
    at = AvailableAt.from_string("D-1T0600Z")
    assert at.day_offset == -1
    assert at.time_of_day == time(6, 0)
    assert at.tzinfo == pytz.UTC


def test_available_at_from_string_rejects_invalid_tz():
    with pytest.raises(pytz.UnknownTimeZoneError):
        AvailableAt.from_string("D-1T0600[INVALID]")


_AMS = pytz.timezone("Europe/Amsterdam")


@pytest.mark.parametrize(
    ("available_at", "reference_date", "expected"),
    [
        pytest.param(
            AvailableAt(day_offset=-1, time_of_day=time(6, 0)),
            datetime(2026, 3, 6),  # noqa: DTZ001
            datetime(2026, 3, 5, 6, 0),  # noqa: DTZ001
            id="naive",
        ),
        pytest.param(
            AvailableAt(day_offset=-2, time_of_day=time(12, 0)),
            datetime(2026, 3, 6, 13),  # noqa: DTZ001
            datetime(2026, 3, 4, 12, 0),  # noqa: DTZ001
            id="naive_D-2",
        ),
        pytest.param(
            AvailableAt(day_offset=-1, time_of_day=time(6, 0), tzinfo=pytz.UTC),
            datetime(2026, 3, 6, tzinfo=UTC),
            datetime(2026, 3, 5, 6, 0, tzinfo=UTC),
            id="utc_to_utc",
        ),
        pytest.param(
            AvailableAt(day_offset=-1, time_of_day=time(6, 0), tzinfo=_AMS),
            datetime(2026, 3, 6),  # noqa: DTZ001
            datetime(2026, 3, 5, 6, 0),  # noqa: DTZ001
            id="ams_tz_naive_ref_returns_naive",
        ),
        pytest.param(
            AvailableAt(day_offset=-1, time_of_day=time(6, 0), tzinfo=UTC),
            _AMS.localize(datetime(2026, 3, 6)),  # noqa: DTZ001
            # 06:00 UTC = 07:00 CET (March 5 is still winter time, UTC+1)
            _AMS.localize(datetime(2026, 3, 5, 7, 0)),  # noqa: DTZ001
            id="utc_stdlib_tz_ams_ref_returns_ams",
        ),
        pytest.param(
            AvailableAt(day_offset=-1, time_of_day=time(6, 0)),
            datetime(2026, 3, 6, tzinfo=UTC),
            datetime(2026, 3, 5, 6, 0, tzinfo=UTC),
            id="fallback_to_date_tz",
        ),
        pytest.param(
            AvailableAt(day_offset=-1, time_of_day=time(6, 0)),
            datetime(2026, 3, 6, tzinfo=timezone(timedelta(hours=1))),
            datetime(2026, 3, 5, 6, 0, tzinfo=timezone(timedelta(hours=1))),
            id="fallback_to_stdlib_fixed_offset_tz",
        ),
        # DST transition: clocks spring forward on 2026-03-29 in Europe/Amsterdam
        # 06:00 CEST (UTC+2) on Mar 29 = 04:00 UTC
        pytest.param(
            AvailableAt(day_offset=-1, time_of_day=time(6, 0), tzinfo=_AMS),
            datetime(2026, 3, 30, tzinfo=UTC),
            datetime(2026, 3, 29, 4, 0, tzinfo=UTC),
            id="ams_to_utc_after_dst_switch",
        ),
        # Day before DST: 06:00 CET (UTC+1) on Mar 28 = 05:00 UTC
        pytest.param(
            AvailableAt(day_offset=-1, time_of_day=time(6, 0), tzinfo=_AMS),
            datetime(2026, 3, 29, tzinfo=UTC),
            datetime(2026, 3, 28, 5, 0, tzinfo=UTC),
            id="ams_to_utc_before_dst_switch",
        ),
    ],
)
def test_available_at_apply(
    available_at: AvailableAt,
    reference_date: datetime,
    expected: datetime,
):
    # Act
    result = available_at.apply(reference_date)

    # Assert
    assert result == expected
    if reference_date.tzinfo is None:
        assert result.tzinfo is None
    else:
        assert result.tzinfo == reference_date.tzinfo


def test_available_at_day_offset():
    assert AvailableAt(day_offset=-1, time_of_day=time(6, 0)).day_offset == -1
    assert AvailableAt(day_offset=-2, time_of_day=time(12, 0)).day_offset == -2


def test_available_at_time_of_day():
    assert AvailableAt(day_offset=-1, time_of_day=time(6, 0)).time_of_day == time(6, 0)
    assert AvailableAt(day_offset=-2, time_of_day=time(12, 0)).time_of_day == time(12, 0)
    assert AvailableAt(day_offset=-1, time_of_day=time(5, 30)).time_of_day == time(5, 30)


def test_available_at_apply_index_naive():
    """apply_index on a naive DatetimeIndex returns correct naive cutoffs."""

    index = pd.DatetimeIndex([
        datetime(2026, 3, 6),  # noqa: DTZ001
        datetime(2026, 3, 7),  # noqa: DTZ001
    ])
    at = AvailableAt(day_offset=-1, time_of_day=time(6, 0))

    result = at.apply_index(index)

    expected = pd.DatetimeIndex([
        datetime(2026, 3, 5, 6, 0),  # noqa: DTZ001
        datetime(2026, 3, 6, 6, 0),  # noqa: DTZ001
    ])
    pd.testing.assert_index_equal(result, expected)


def test_available_at_apply_index_utc():
    """apply_index on a UTC index with no self.tzinfo falls back to index tz."""

    index = pd.to_datetime(["2026-03-06T00:00:00+00:00", "2026-03-07T00:00:00+00:00"])
    at = AvailableAt(day_offset=-1, time_of_day=time(6, 0))

    result = at.apply_index(index)

    expected = pd.to_datetime(["2026-03-05T06:00:00+00:00", "2026-03-06T06:00:00+00:00"])
    pd.testing.assert_index_equal(result, expected)


def test_available_at_apply_index_cross_tz_dst():
    """apply_index with AMS tzinfo on UTC index shifts correctly across DST."""
    # Index in UTC, AvailableAt in Europe/Amsterdam
    index = pd.to_datetime([
        "2026-03-29T12:00:00+00:00",  # cutoff = Mar 28 06:00 CET = 05:00 UTC
        "2026-03-30T12:00:00+00:00",  # cutoff = Mar 29 06:00 CEST = 04:00 UTC
    ])
    at = AvailableAt(day_offset=-1, time_of_day=time(6, 0), tzinfo=_AMS)

    result = at.apply_index(index)

    expected = pd.to_datetime([
        "2026-03-28T05:00:00+00:00",
        "2026-03-29T04:00:00+00:00",
    ])
    pd.testing.assert_index_equal(result, expected)
    assert result.tz == index.tz


def test_available_at_apply_index_matches_apply():
    """apply_index results should match element-wise apply() calls."""
    index = pd.to_datetime([
        "2026-03-28T12:00:00+00:00",
        "2026-03-29T12:00:00+00:00",
        "2026-03-30T12:00:00+00:00",
    ])
    at = AvailableAt(day_offset=-1, time_of_day=time(6, 0), tzinfo=_AMS)

    vectorized = at.apply_index(index)
    scalar = pd.DatetimeIndex([at.apply(ts.to_pydatetime()) for ts in index])

    pd.testing.assert_index_equal(vectorized, scalar)
