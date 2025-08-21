# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

from pydantic import TypeAdapter


def timedelta_to_isoformat(td: timedelta) -> str:
    """Convert timedelta to ISO 8601 string format."""
    return TypeAdapter(timedelta).dump_python(td, mode="json")


def timedelta_from_isoformat(s: str) -> timedelta:
    """Convert ISO 8601 string format to timedelta."""
    return TypeAdapter(timedelta).validate_python(s)


__all__ = [
    "timedelta_from_isoformat",
    "timedelta_to_isoformat",
]
