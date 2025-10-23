# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Window-based iteration utilities for evaluation data segmentation.

Provides functions to create sliding time windows and filter evaluation subsets
across different time dimensions. Enables systematic analysis of model performance
over rolling time periods.
"""

from collections.abc import Iterator
from datetime import datetime, timedelta

import pandas as pd

from openstef_beam.evaluation.models import Window
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.utils import align_datetime


def iterate_by_window(
    index: pd.DatetimeIndex,
    window: Window,
    sample_interval: timedelta,
) -> Iterator[tuple[datetime, pd.DatetimeIndex]]:
    """Yields fixed-size sliding windows over a time series index.

    Guarantees:
    - Windows are aligned to the window stride
    - Windows cover the entire range from index.min() + window.size to index.max()
    - Each window has exactly the specified window size
    - Window indices include timestamps from (end - window.size) to end (exclusive)

    Args:
        index: DatetimeIndex to create windows from
        window: Window specification with size, stride parameters
        sample_interval: Time interval between consecutive samples

    Yields:
        Iterator of (window end time, window DatetimeIndex) tuples
    """
    for end in pd.date_range(
        start=align_datetime(index.min() + window.size, window.stride, mode="ceil"),  # type: ignore[reportUnknownMemberType]
        end=align_datetime(index.max(), window.stride, mode="floor"),  # type: ignore[reportUnknownMemberType]
        freq=window.stride,
    ):
        yield (
            end.to_pydatetime(),
            pd.date_range(
                start=end - window.size,
                end=end,
                freq=sample_interval,
                inclusive="left",
            ),
        )


def iterate_subsets_by_window[T: TimeSeriesDataset](
    subset: T,
    window: Window,
) -> Iterator[tuple[datetime, T]]:
    """Yields evaluation subsets for each window with sufficient data coverage.

    Guarantees:
    - Windows with coverage less than window.minimum_coverage are skipped
    - Each yielded subset contains only timestamps present in window, ground truth and predictions
    - All yielded subsets maintain the original sample interval

    Args:
        subset: The evaluation subset to iterate over
        window: Window specification with size, stride, and minimum coverage parameters

    Yields:
        Iterator of (window end time, windowed evaluation subset) tuples
    """
    for window_timestamp, window_index in iterate_by_window(
        index=subset.index,
        window=window,
        sample_interval=subset.sample_interval,
    ):
        window_data = subset.filter_index(window_index)

        # If there is not enough data in the window, then skip it
        window_coverage = window_data.calculate_time_coverage() / window.size
        if window_coverage < window.minimum_coverage:
            continue

        yield (window_timestamp, window_data)


__all__ = [
    "iterate_by_window",
    "iterate_subsets_by_window",
]
