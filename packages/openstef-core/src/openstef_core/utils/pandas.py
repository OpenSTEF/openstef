# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Pandas utility functions for time series data processing.

This module provides utility functions for working with pandas time series data.
"""

import functools
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import cast

import pandas as pd


def unsafe_sorted_range_slice_idxs(
    data: pd.Series, start: datetime | pd.Timestamp | None, end: datetime | pd.Timestamp | None
) -> tuple[int, int]:
    """Get sorted slice indices for a datetime range.

    Efficiently finds the start and end indices for slicing a sorted datetime series
    within a specified time range. Uses binary search for optimal performance on
    large datasets.

    This function is particularly useful when working with large time series datasets
    where you need to quickly extract data for specific time windows without
    iterating through all data points.

    Args:
        data: Sorted pandas Series with datetime index or values to search.
        start: Start of the time range (inclusive). If None, starts from beginning.
        end: End of the time range (exclusive). If None, goes to the end.

    Returns:
        Tuple of (start_index, end_index) for slicing the data series.

    Note:
        This function assumes that the input series is sorted in ascending order.
        It does not perform any checks to verify this, so it's up to the caller to
        ensure that this invariant holds true.
    """
    start_idx = data.searchsorted(start, side="left") if start else 0
    end_idx = data.searchsorted(end, side="left") if end else len(data)
    return int(start_idx), int(end_idx)


def combine_timeseries_indexes(indexes: Sequence[pd.DatetimeIndex]) -> pd.DatetimeIndex:
    """Combine multiple datetime indexes into a single sorted index.

    Merges several pandas DatetimeIndex objects into one, ensuring that the
    resulting index is sorted and contains no duplicate timestamps.

    Args:
        indexes: Sequence of pandas DatetimeIndex objects to combine.

    Returns:
        A single pandas DatetimeIndex containing all unique timestamps from the input indexes, sorted in ascending
        order.
    """
    if not indexes:
        return pd.DatetimeIndex([])

    union_fn = cast(
        Callable[[pd.DatetimeIndex, pd.DatetimeIndex], pd.DatetimeIndex],
        functools.partial(pd.DatetimeIndex.union, sort=False),
    )
    index_raw = functools.reduce(union_fn, indexes)
    return index_raw.unique().sort_values(ascending=True)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
