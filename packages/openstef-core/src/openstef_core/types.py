# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Core type definitions for OpenSTEF time series analysis.

Provides typed wrappers for common temporal and quantile concepts used throughout
the forecasting pipeline. Ensures consistent serialization and validation of
key domain types like lead times, availability timestamps, and quantile values.
"""

import re
from datetime import timedelta
from enum import StrEnum
from functools import total_ordering
from typing import Any, Literal, Self, override

from pydantic import GetCoreSchemaHandler, TypeAdapter
from pydantic_core import CoreSchema, core_schema

from openstef_core.base_model import PydanticStringPrimitive


@total_ordering
class LeadTime(PydanticStringPrimitive):
    """Represents a lead time as a timedelta.

    Used for serialization and validation of lead time values. Maintains a consistent
    string representation for timedeltas in ISO 8601 format.

    Example:
        Creating and using lead times:

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

    def __str__(self) -> str:
        """Converts to ISO 8601 duration string.

        Returns:
            ISO 8601 formatted duration string.
        """
        return TypeAdapter(timedelta).dump_python(self.value, mode="json")

    def __repr__(self) -> str:
        """Returns a detailed string representation for debugging.

        Returns:
            String representation showing the class name and ISO 8601 duration string.
        """
        return f"LeadTime('{self}')"

    @classmethod
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
    def validate(cls, v: Self | str | timedelta, _info: Any = None) -> Self:
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

    Uses a specialized string format 'DnTHH:MM' where:
    - n is the day offset (negative indicates prior days)
    - HH:MM is the time of day

    For example, 'D-1T06:00' means "6:00 AM on the previous day".

    Example:
        Creating and using availability times:

        >>> from datetime import timedelta
        >>> # Available at 6 AM on the previous day
        >>> at = AvailableAt(timedelta(hours=18))  # 18 hours before day end
        >>> str(at)
        'D-1T06:00'
        >>> # Available at midnight of the current day
        >>> AvailableAt.from_string('D0T00:00').lag_from_day
        datetime.timedelta(0)
    """

    def __init__(self, lag_from_day: timedelta):
        """Initializes with a lag from the reference day start."""
        self.lag_from_day = lag_from_day

    def __str__(self) -> str:
        """Converts to string in 'DnTHH:MM' format.

        Returns:
            String representation in 'DnTHH:MM' format.
        """
        lag_days = -int(self.lag_from_day / timedelta(days=1)) - 1
        time = timedelta(hours=24) - (self.lag_from_day % timedelta(days=1))
        return f"D{lag_days}T{time.seconds // 3600:02}:{(time.seconds // 60) % 60:02}"

    @classmethod
    def from_string(cls, s: str) -> Self:
        """Creates an instance from a string in 'DnTHH:MM' format.

        Args:
            s: String in 'DnTHH:MM' format to parse.

        Returns:
            AvailableAt instance parsed from the string.

        Raises:
            ValueError: If the string format is invalid.
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
        """Validates and converts various input types to AvailableAt.

        Args:
            v: Value to validate (AvailableAt, string, or timedelta).
            _info: Additional validation info (unused).

        Returns:
            Validated AvailableAt instance.
        """
        if isinstance(v, timedelta):
            return cls(lag_from_day=v)

        return super().validate(v, _info)


class Quantile(float):
    """A float subclass representing a quantile value between 0 and 1.

    Example:
        Creating and using quantiles:

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
        return core_schema.no_info_after_validator_function(cls, handler(float))

    def format(self) -> str:
        """Instance method to format the quantile as a string.

        Returns:
            Formatted quantile string in 'quantile_PXX' format.
        """
        value = self * 100

        # Check if the value is a whole number
        if value.is_integer():
            return f"quantile_P{int(value):02d}"
        # Format with one decimal place for non-integer values
        return f"quantile_P{value:.1f}"

    @staticmethod
    def parse(quantile_str: str) -> "Quantile":
        """Static method to parse a quantile string back to a Quantile object.

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
        value = float(quantile_str.split("_P")[1]) / 100
        return Quantile(value)

    @staticmethod
    def is_valid_quantile_string(quantile_str: str) -> bool:
        """Check if a string is a valid quantile representation.

        Args:
            quantile_str: String to check.

        Returns:
            True if the string is a valid quantile representation, False otherwise.
        """
        pattern = r"^quantile_P(\d{1,2}(\.\d)?|100)$"
        return re.match(pattern, quantile_str) is not None


Q = Quantile  # Alias for easier imports
QuantileOrGlobal = Quantile | Literal["global"]


class EnergyComponentType(StrEnum):
    """Enumeration of energy component types."""

    WIND = "wind"
    SOLAR = "solar"
    OTHER = "other"
