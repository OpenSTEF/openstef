# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Datetime manipulation utilities for time series alignment and processing.

Provides functions for aligning timestamps to specific intervals or times of day,
commonly used in energy forecasting workflows where data must be synchronized
to regular time grids or specific analysis windows.
"""

import math
from datetime import datetime, time, timedelta
from typing import Literal


def align_datetime(timestamp: datetime, interval: timedelta, mode: Literal["ceil", "floor"] = "ceil") -> datetime:
    """Align timestamp using modulo approach.

    Args:
        timestamp: The datetime to align.
        interval: Time interval to align to.
        mode: Alignment direction - "ceil" for next interval, "floor" for previous.

    Returns:
        Aligned datetime matching the specified interval boundary.

    Raises:
        ValueError: If mode is not "ceil" or "floor".
    """
    timestamp_epoch_seconds = timestamp.timestamp()
    interval_seconds = interval.total_seconds()

    if mode == "floor":
        aligned_secs = math.floor(timestamp_epoch_seconds / interval_seconds) * interval_seconds
    elif mode == "ceil":
        aligned_secs = math.ceil(timestamp_epoch_seconds / interval_seconds) * interval_seconds
    else:
        msg = f"Unknown alignment mode: {mode}"
        raise ValueError(msg)

    if timestamp.tzinfo is None:
        return datetime.fromtimestamp(aligned_secs, tz=None)  # noqa: DTZ006
    return datetime.fromtimestamp(aligned_secs, tz=timestamp.tzinfo)


def align_datetime_to_time(timestamp: datetime, align_time: time, mode: Literal["ceil", "floor"] = "ceil") -> datetime:
    """Align timestamp to the nearest occurrence of a specific time of day.

    Aligns a timestamp to either the next (ceil) or previous (floor) occurrence
    of the specified time. Properly handles timezone conversions when both
    timestamp and align_time have timezone information.

    Args:
        timestamp: The datetime to align.
        align_time: Target time of day to align to. If timezone-aware and timestamp
                   has timezone info, converts align_time to timestamp's timezone.
        mode: Alignment direction - "ceil" for next occurrence, "floor" for previous.

    Returns:
        Aligned datetime with the same timezone as the original timestamp.

    Example:
        >>> from datetime import datetime, time
        >>> dt = datetime.fromisoformat("2023-01-15T14:30:00")
        >>> target = time.fromisoformat("09:00:00")
        >>> align_datetime_to_time(dt, target, "ceil")
        datetime.datetime(2023, 1, 16, 9, 0)
        >>> align_datetime_to_time(dt, target, "floor")
        datetime.datetime(2023, 1, 15, 9, 0)
    """
    # Convert align_time to timestamp's timezone if both have timezone info
    if timestamp.tzinfo is not None and align_time.tzinfo is not None:
        # Convert align_time to timestamp's timezone for comparison
        temp_dt = datetime.combine(timestamp.date(), align_time)
        temp_dt = temp_dt.replace(tzinfo=align_time.tzinfo)
        temp_dt = temp_dt.astimezone(timestamp.tzinfo)
        target_time = temp_dt.time()
    elif timestamp.tzinfo is not None and align_time.tzinfo is None:
        # align_time is naive, treat as if it's in timestamp's timezone
        target_time = align_time
    elif timestamp.tzinfo is None and align_time.tzinfo is not None:
        # timestamp is naive, convert align_time to naive
        target_time = align_time.replace(tzinfo=None)
    else:
        # Both are naive
        target_time = align_time

    # Create aligned datetime in the same timezone as original timestamp
    aligned_date = timestamp.replace(
        hour=target_time.hour,
        minute=target_time.minute,
        second=target_time.second if hasattr(target_time, "second") else 0,
        microsecond=0,
    )

    # Adjust by one day if needed based on mode
    if mode == "ceil" and aligned_date < timestamp:
        aligned_date += timedelta(days=1)
    elif mode == "floor" and aligned_date > timestamp:
        aligned_date -= timedelta(days=1)

    return aligned_date


__all__ = [
    "align_datetime",
    "align_datetime_to_time",
]
