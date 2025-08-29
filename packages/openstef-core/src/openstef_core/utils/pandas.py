# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Pandas utility functions for time series data processing.

This module provides utility functions for working with pandas time series data.
"""

from datetime import datetime

import pandas as pd


def sorted_range_slice_idxs(data: pd.Series, start: datetime | None, end: datetime | None) -> tuple[int, int]:
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
    """
    start_idx = data.searchsorted(start, side="left") if start else 0
    end_idx = data.searchsorted(end, side="left") if end else len(data)
    return start_idx, end_idx
