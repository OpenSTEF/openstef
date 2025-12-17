# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Time window definition for sliding window analysis.

Provides structured representation of temporal analysis windows with configurable
lag, size, and stride parameters. Supports serialization for consistent window
specification across evaluation pipelines.
"""

import re
from datetime import timedelta
from typing import Any, Self, override

from pydantic import TypeAdapter

from openstef_core.base_model import PydanticStringPrimitive
from openstef_core.types import AvailableAt, LeadTime

type Filtering = AvailableAt | LeadTime


class Window(PydanticStringPrimitive):
    """Represents a time window with lag, size, and stride parameters.

    Used for defining sliding windows for time series analysis with:
    - lag: How far back from the reference point to start the window
    - size: The duration of the window
    - stride: How much to advance for the next window
    - minimum_coverage: Minimum required data coverage (0.0-1.0)
    """

    def __init__(
        self, lag: timedelta, size: timedelta, stride: timedelta = timedelta(days=1), minimum_coverage: float = 0.5
    ):
        """Initialize a time window with the specified parameters.

        Args:
            lag: How far back from the reference point to start the window.
            size: The duration of the window.
            stride: How much to advance for the next window.
            minimum_coverage: Minimum required data coverage (0.0-1.0).
        """
        self.lag = lag
        self.size = size
        self.stride = stride
        self.minimum_coverage = minimum_coverage

    def __str__(self) -> str:
        """Converts to string in '(lag=X,size=Y,stride=Z)' format.

        Timedeltas are serialized in ISO 8601 format.

        Returns:
            String representation of the window parameters.
        """
        lag_str, size_str, stride_str = TypeAdapter(tuple[timedelta, timedelta, timedelta]).dump_python(
            (self.lag, self.size, self.stride), mode="json"
        )
        return f"(lag={lag_str},size={size_str},stride={stride_str})"

    @classmethod
    def from_string(cls, s: str) -> Self:
        """Creates an instance from a string in '(lag=X,size=Y,stride=Z)' format.

        Args:
            s: String representation to parse.

        Returns:
            Window instance parsed from the string.

        Raises:
            ValueError: If the string format is invalid.
        """
        match = re.findall(r"\(lag=(.*),size=(.*),stride=(.*)\)", s)
        if not match:
            error_message = f"Invalid format: {s}"
            raise ValueError(error_message)

        lag, size, stride = TypeAdapter(tuple[timedelta, timedelta, timedelta]).validate_python(match[0])
        return cls(lag=lag, size=size, stride=stride)

    @classmethod
    @override
    def validate(cls, v: Self | str | dict[str, Any], _info: Any = None) -> Self:
        """Validates and converts various input types to Window.

        Args:
            v: Input value to validate (Window, string, or dict).
            _info: Additional validation info (unused).

        Returns:
            Validated Window instance.
        """
        if isinstance(v, dict):
            minimum_coverage = v.get("minimum_coverage", 0.5)
            lag, size, stride = TypeAdapter(tuple[timedelta, timedelta, timedelta]).validate_python((
                v.get("lag"),
                v.get("size"),
                v.get("stride", timedelta(days=1)),
            ))

            return cls(lag=lag, size=size, stride=stride, minimum_coverage=minimum_coverage)

        return super().validate(v, _info)
