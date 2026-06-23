# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Core type definitions for OpenSTEF time series analysis.

Provides typed wrappers for common temporal and quantile concepts used throughout
the forecasting pipeline. Ensures consistent serialization and validation of
key domain types like lead times, availability timestamps, and quantile values.
"""

from __future__ import annotations

import re
from datetime import datetime, time, timedelta
from datetime import timezone as dt_timezone
from decimal import Decimal
from enum import StrEnum
from functools import total_ordering
from typing import Any, Literal, Self, override
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import GetCoreSchemaHandler, TypeAdapter, ValidationInfo
from pydantic_core import CoreSchema, core_schema

from openstef_core.base_model import PydanticStringPrimitive


@total_ordering
class LeadTime(PydanticStringPrimitive):
    """Represents a lead time as a timedelta.

    Used for serialization and validation of lead time values. Maintains a consistent
    string representation for timedeltas in ISO 8601 format.

    Example:
        Creating and using lead times

        >>> from datetime import timedelta
        >>> lt = LeadTime(timedelta(hours=6))
        >>> str(lt)
        'PT6H'
        >>> LeadTime.from_string('PT6H').value
        datetime.timedelta(seconds=21600)
        >>> LeadTime.validate(timedelta(days=1)).value
        datetime.timedelta(days=1)
    """

    def __init__(self, value: timedelta):
        """Initialize a LeadTime with the given timedelta value.

        Args:
            value: The timedelta representing the lead time duration.
        """
        self.value = value

    @override
    def __str__(self) -> str:
        """Converts to ISO 8601 duration string.

        Returns:
            ISO 8601 formatted duration string.
        """
        return TypeAdapter(timedelta).dump_python(self.value, mode="json")

    @override
    def __repr__(self) -> str:
        """Returns a detailed string representation for debugging.

        Returns:
            String representation showing the class name and ISO 8601 duration string.
        """
        return f"LeadTime('{self}')"

    @classmethod
    @override
    def from_string(cls, s: str) -> Self:
        """Creates an instance from an ISO 8601 duration string.

        Args:
            s: ISO 8601 duration string to parse.

        Returns:
            LeadTime instance parsed from the string.
        """
        return cls(TypeAdapter(timedelta).validate_python(s))

    @classmethod
    @override
    def validate(cls, v: Self | str | timedelta, _info: ValidationInfo | None = None) -> Self:
        """Validates and converts various input types to LeadTime.

        Accepts LeadTime objects, ISO 8601 duration strings, or timedelta objects.

        Args:
            v: Value to validate (LeadTime, string, or timedelta).
            _info: Additional validation info (unused).

        Returns:
            Validated LeadTime instance.
        """
        if isinstance(v, timedelta):
            return cls(v)

        return super().validate(v, _info)

    def __lt__(self, other: object) -> bool:
        """Less-than comparison based on timedelta value.

        Returns:
            True if this LeadTime is less than the other, False otherwise.
        """
        if not isinstance(other, LeadTime):
            return NotImplemented

        return self.value < other.value

    def to_hours(self) -> float:
        """Converts the lead time to total hours.

        Returns:
            Total hours represented by the lead time.
        """
        return self.value.total_seconds() / 3600.0


class AvailableAt(PydanticStringPrimitive):
    """Represents a time point available relative to a reference day.

    Uses a specialized string format ``DnTHHMM`` where:

    - *n* is the day offset (negative or zero)
    - *HHMM* is the time of day

    An optional timezone suffix ``[Region/City]`` (RFC 9557 bracket
    notation) makes the availability time timezone-aware.  ``zoneinfo``
    and stdlib ``datetime.timezone`` objects are accepted; they
    round-trip through the IANA name via ``str(tz)`` /
    ``zoneinfo.ZoneInfo(name)``.

    For example, ``D-1T0600[Europe/Amsterdam]`` means "6:00
    Europe/Amsterdam on the previous day".
    The legacy ``DnTHH:MM`` format (with colon) is also accepted by
    ``from_string()``.

    Example:
        >>> from datetime import time
        >>> from zoneinfo import ZoneInfo
        >>> tz_at = AvailableAt(day_offset=-1, time_of_day=time(6, 0), tzinfo=ZoneInfo('Europe/Amsterdam'))
        >>> str(tz_at)
        'D-1T0600[Europe/Amsterdam]'
        >>> at = AvailableAt.from_string("D-1T0600")
        >>> at.day_offset, at.time_of_day
        (-1, datetime.time(6, 0))
    """

    def __init__(self, day_offset: int, time_of_day: time, *, tzinfo: ZoneInfo | dt_timezone | None = None):
        """Initialise with a day offset, time of day, and optional timezone.

        Args:
            day_offset: Day offset from the reference day (must be ≤ 0).
                ``-1`` means "the previous day", ``0`` means "the same day".
            time_of_day: Clock time when data becomes available.
            tzinfo: Optional timezone for the availability time
                (e.g. ``zoneinfo.ZoneInfo("Europe/Amsterdam")``, ``zoneinfo.ZoneInfo("UTC")``,
                or ``datetime.timezone.utc``).

        Raises:
            ValueError: If day_offset is positive.
        """
        if day_offset > 0:
            msg = f"Day offset must be negative or zero, got {day_offset}"
            raise ValueError(msg)
        self.day_offset = day_offset
        self.time_of_day = time_of_day
        self.tzinfo = tzinfo

    @override
    def __str__(self) -> str:
        """Converts to string in ``DnTHHMM`` or ``DnTHHMM[tz]`` format.

        Returns:
            String representation, with optional ``[timezone]`` suffix.
        """
        base = f"D{self.day_offset}T{self.time_of_day.hour:02}{self.time_of_day.minute:02}"
        if self.tzinfo is not None:
            return f"{base}[{self.tzinfo}]"
        return base

    @classmethod
    @override
    def from_string(cls, s: str) -> Self:
        """Creates an instance from a string in ``DnTHHMM[tz]`` format.

        Accepts an optional ``[Region/City]`` timezone suffix.  The
        legacy colon format ``DnTHH:MM`` is also accepted.

        Args:
            s: String to parse.

        Returns:
            AvailableAt instance parsed from the string.

        Raises:
            ValueError: If the string format is invalid or day offset is positive.
        """
        match = re.match(r"D(-?\d+)T(\d{2}):?(\d{2})(?:(Z)|\[([^\]]+)\])?$", s)
        if not match:
            error_message = f"Cannot convert {s} to {cls.__name__}"
            raise ValueError(error_message)

        days_part, hours_part, minutes_part, z_part, tz_part = match.groups()

        if int(days_part) > 0:
            msg = f"Day offset must be negative or zero, got {days_part}"
            raise ValueError(msg)

        if z_part:
            resolved_tz = ZoneInfo("UTC")
        elif tz_part:
            resolved_tz = ZoneInfo(tz_part)
        else:
            resolved_tz = None

        return cls(
            day_offset=int(days_part),
            time_of_day=time(hour=int(hours_part), minute=int(minutes_part)),
            tzinfo=resolved_tz,
        )

    def apply(self, date: datetime) -> datetime:
        """Apply this availability offset to a reference date.

        The day offset is interpreted in ``self.tzinfo`` (falls back to
        ``date.tzinfo``), so "D-1" means "the previous calendar day in
        that timezone".  The time-of-day is also placed in that timezone.
        The result is returned in the reference date's timezone, or naive
        when the reference date is naive.

        This matches the vectorised :meth:`apply_index` semantics, which
        converts to ``self.tzinfo`` before extracting the calendar day.

        Args:
            date: The reference date to apply the availability offset to.

        Returns:
            The datetime when data is available, in the reference date's
            timezone (or naive when the reference date is naive).
        """
        source_tz = self.tzinfo or date.tzinfo

        # Convert reference date to source timezone before extracting the
        # calendar day.  This ensures "D-1" means "previous day in the
        # AvailableAt's timezone", consistent with apply_index().
        working_date = date.astimezone(source_tz) if source_tz is not None and date.tzinfo is not None else date

        result_date = (working_date + timedelta(days=self.day_offset)).date()
        naive_result = datetime.combine(result_date, self.time_of_day)

        if source_tz is None:
            return naive_result

        aware = naive_result.replace(tzinfo=source_tz)

        if date.tzinfo is not None:
            return aware.astimezone(date.tzinfo)
        return naive_result

    def apply_index(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Vectorized version of :meth:`apply` for a pandas DatetimeIndex.

        Same timezone logic as :meth:`apply`: the time-of-day is
        interpreted in ``self.tzinfo`` (falls back to ``index.tz``),
        then converted back to the index's timezone.

        Args:
            index: DatetimeIndex of reference dates.

        Returns:
            DatetimeIndex of cutoff timestamps, in the same timezone as *index*.
        """
        source_tz = self.tzinfo
        data_tz = index.tz

        work_index = index.tz_convert(source_tz) if source_tz is not None and data_tz is not None else index

        cutoff = work_index.floor("D") + pd.Timedelta(
            days=self.day_offset,
            hours=self.time_of_day.hour,
            minutes=self.time_of_day.minute,
        )

        if source_tz is not None and data_tz is not None:
            cutoff = cutoff.tz_convert(data_tz)

        return cutoff


class Quantile(float):
    """A float subclass representing a quantile value between 0 and 1.

    Uses ``Decimal`` arithmetic internally for all conversions
    so that values like ``Quantile(0.999)`` round-trip exactly through
    ``format()`` / ``parse()`` and ``to_percentile()`` / ``from_percentile()``.

    Example:
        Creating and using quantiles

        >>> q50 = Quantile(0.5)  # Median
        >>> q50.format()
        'quantile_P50'
        >>> q95 = Quantile(0.95)  # 95th percentile
        >>> q95.format()
        'quantile_P95'
        >>> # Parse from string
        >>> Quantile.parse('quantile_P25')
        0.25
        >>> # Non-integer quantiles
        >>> Quantile(0.025).format()
        'quantile_P2.5'
        >>> # Percentile conversion
        >>> Quantile.from_percentile(99.9).format()
        'quantile_P99.9'
        >>> # Complementary quantiles
        >>> Quantile(0.1).complementary()
        0.9
    """

    def __new__(cls, value: float) -> Self:
        """Create a new Quantile instance with validation.

        Args:
            value: Float value between 0 and 1.

        Returns:
            New Quantile instance.

        Raises:
            ValueError: If value is not between 0 and 1.
        """
        if not 0 <= value <= 1:
            msg = f"Quantile must be between 0 and 1, got {value}"
            raise ValueError(msg)
        return super().__new__(cls, value)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:  # noqa: ANN401
        """Define Pydantic validation and serialization behavior.

        Returns:
            Core schema for Pydantic validation.
        """
        return core_schema.no_info_after_validator_function(
            cls,
            handler(float),
            serialization=core_schema.plain_serializer_function_ser_schema(float),
        )

    def _percentile_decimal(self) -> Decimal:
        """Return the percentile as an exact ``Decimal``.

        Uses ``Decimal(str(...))`` to avoid IEEE-754 rounding artefacts
        that occur with plain ``float * 100``.
        """
        return Decimal(str(float(self))) * 100

    def format(self) -> str:
        """Instance method to format the quantile as a string.

        Returns:
            Formatted quantile string in 'quantile_PXX' format.
        """
        value = self._percentile_decimal()
        normalized = value.normalize()

        # Check if the value is a whole number
        if normalized == int(normalized):
            return f"quantile_P{int(normalized):02d}"
        return f"quantile_P{normalized}"

    @classmethod
    def parse(cls, quantile_str: str) -> Self:
        """Parse a quantile string back to a Quantile object.

        Args:
            quantile_str: String in 'quantile_PXX' format.

        Returns:
            Parsed Quantile object.

        Raises:
            ValueError: If the string format is invalid.
        """
        if not quantile_str.startswith("quantile_P"):
            msg = f"Invalid quantile string: {quantile_str}"
            raise ValueError(msg)
        value = Decimal(quantile_str.split("_P")[1]) / 100
        return cls(float(value))

    @staticmethod
    def is_valid_quantile_string(quantile_str: str) -> bool:
        """Check if a string is a valid quantile representation.

        Args:
            quantile_str: String to check.

        Returns:
            True if the string is a valid quantile representation, False otherwise.
        """
        pattern = r"^quantile_P(\d{1,3}(\.\d+)?|100)$"
        return re.match(pattern, quantile_str) is not None

    @classmethod
    def from_percentile(cls, percentile: float) -> Self:
        """Create a Quantile from a percentile value.

        Args:
            percentile: Percentile value between 0 and 100.

        Returns:
            Quantile corresponding to the given percentile.

        Raises:
            ValueError: If the percentile is not between 0 and 100.
        """
        min_percentile, max_percentile = 0, 100
        if not (min_percentile <= percentile <= max_percentile):
            msg = f"Percentile must be between 0 and 100, got {percentile}"
            raise ValueError(msg)
        return cls(float(Decimal(str(percentile)) / 100))

    def to_percentile(self) -> float:
        """Convert this Quantile to a percentile value.

        Returns:
            Percentile value corresponding to this Quantile.
        """
        return float(self._percentile_decimal())

    def complementary(self) -> Self:
        """Return the complementary quantile (``1 - self``)."""
        return self.__class__(float(Decimal(1) - Decimal(str(self))))


Q = Quantile  # Alias for easier imports
QuantileOrGlobal = Quantile | Literal["global"]


class EnergyComponentType(StrEnum):
    """Enumeration of energy component types."""

    WIND = "wind"
    SOLAR = "solar"
    OTHER = "other"
