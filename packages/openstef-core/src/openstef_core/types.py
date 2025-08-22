# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import re
from datetime import timedelta
from typing import Any, NewType, Self, override

from pydantic import TypeAdapter

from openstef_core.base_model import PydanticStringPrimitive


class LeadTime(PydanticStringPrimitive):
    """Represents a lead time as a timedelta.

    Used for serialization and validation of lead time values. Maintains a consistent
    string representation for timedeltas in ISO 8601 format.
    """

    def __init__(self, value: timedelta):
        self.value = value

    def __str__(self) -> str:
        """Converts to ISO 8601 duration string."""
        return TypeAdapter(timedelta).dump_python(self.value, mode="json")

    @classmethod
    def from_string(cls, s: str) -> Self:
        """Creates an instance from an ISO 8601 duration string."""
        return cls(TypeAdapter(timedelta).validate_python(s))

    @classmethod
    @override
    def validate(cls, v: Self | str | timedelta, _info: Any = None) -> Self:
        """Validates and converts various input types to LeadTime.

        Accepts LeadTime objects, ISO 8601 duration strings, or timedelta objects.
        Raises ValueError if the input cannot be converted.
        """
        if isinstance(v, timedelta):
            return cls(v)

        return super().validate(v, _info)


class AvailableAt(PydanticStringPrimitive):
    """Represents a time point available relative to a reference day.

    Uses a specialized string format 'DnTHH:MM' where:
    - n is the day offset (negative indicates prior days)
    - HH:MM is the time of day

    For example, 'D-1T06:00' means "6:00 AM on the previous day".
    """

    def __init__(self, lag_from_day: timedelta):
        """Initializes with a lag from the reference day start."""
        self.lag_from_day = lag_from_day

    def __str__(self) -> str:
        """Converts to string in 'DnTHH:MM' format."""
        lag_days = -int(self.lag_from_day / timedelta(days=1)) - 1
        time = timedelta(hours=24) - (self.lag_from_day % timedelta(days=1))
        return f"D{lag_days}T{time.seconds // 3600:02}:{(time.seconds // 60) % 60:02}"

    @classmethod
    def from_string(cls, s: str) -> Self:
        """Creates an instance from a string in 'DnTHH:MM' format.

        Raises ValueError if the string format is invalid.
        """
        match = re.match(r"D(-?\d+)T(\d{2}):(\d{2})", s)
        if not match:
            error_message = f"Cannot convert {s} to {cls.__name__}"
            raise ValueError(error_message)

        days_part, hours_part, minutes_part = match.groups()

        # Calculate lag_from_day
        lag_days = -int(days_part) - 1
        time = timedelta(hours=int(hours_part), minutes=int(minutes_part))
        lag_from_day = timedelta(days=lag_days) + (timedelta(hours=24) - time)

        return cls(lag_from_day=lag_from_day)

    @classmethod
    @override
    def validate(cls, v: Self | str | timedelta, _info: Any = None) -> Self:
        """Validates and converts various input types to AvailableAt."""
        if isinstance(v, timedelta):
            return cls(lag_from_day=v)

        return super().validate(v, _info)


Quantile = NewType("Quantile", float)


def format_quantile(quantile: float) -> str:
    value = quantile * 100

    # Check if the value is a whole number
    if value.is_integer():
        return f"quantile_P{int(value):02d}"
    # Format with one decimal place for non-integer values
    return f"quantile_P{value:.1f}"


def parse_quantile(quantile_str: str) -> float:
    if not quantile_str.startswith("quantile_P"):
        msg = f"Invalid quantile string: {quantile_str}"
        raise ValueError(msg)
    return float(quantile_str.split("_P")[1]) / 100
