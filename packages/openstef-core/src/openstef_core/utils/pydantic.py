# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Pydantic-based utility functions for data serialization.

This module provides functions for converting Python timedelta objects to and from
ISO 8601 duration string format using Pydantic's serialization capabilities.
These utilities are used for persisting temporal metadata in dataset files.
"""

from datetime import timedelta

from pydantic import TypeAdapter


def timedelta_to_isoformat(td: timedelta) -> str:
    """Convert timedelta to ISO 8601 string format.

    Args:
        td: The timedelta object to convert.

    Returns:
        ISO 8601 duration string representation (e.g., 'PT15M' for 15 minutes).

    Example:
        Convert a 15-minute interval:

        >>> from datetime import timedelta
        >>> td = timedelta(minutes=15)
        >>> timedelta_to_isoformat(td)
        'PT15M'
    """
    return TypeAdapter(timedelta).dump_python(td, mode="json")


def timedelta_from_isoformat(s: str) -> timedelta:
    """Convert ISO 8601 string format to timedelta.

    Args:
        s: ISO 8601 duration string to parse.

    Returns:
        Python timedelta object representing the duration.

    Example:
        Parse a 15-minute interval:

        >>> td = timedelta_from_isoformat('PT15M')
        >>> td.total_seconds()
        900.0
    """
    return TypeAdapter(timedelta).validate_python(s)


__all__ = [
    "timedelta_from_isoformat",
    "timedelta_to_isoformat",
]
