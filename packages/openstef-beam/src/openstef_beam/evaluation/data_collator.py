# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import cast

import pandas as pd

from openstef_beam.evaluation.models import EvaluationSubset, Window
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

    Returns:
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


def iterate_subsets_by_window(
    subset: EvaluationSubset,
    window: Window,
) -> Iterator[tuple[datetime, EvaluationSubset]]:
    """Yields evaluation subsets for each window with sufficient data coverage.

    Guarantees:
    - Windows with coverage less than window.minimum_coverage are skipped
    - Each yielded subset contains only timestamps present in window, ground truth and predictions
    - All yielded subsets maintain the original sample interval

    Args:
        subset: The evaluation subset to iterate over
        window: Window specification with size, stride, and minimum coverage parameters

    Returns:
        Iterator of (window end time, windowed evaluation subset) tuples
    """
    for window_timestamp, window_index in iterate_by_window(
        index=subset.index,
        window=window,
        sample_interval=subset.sample_interval,
    ):
        mutual_index = window_index.intersection(cast(pd.DatetimeIndex, subset.ground_truth.index)).intersection(
            cast(pd.DatetimeIndex, subset.predictions.index)
        )
        window_ground_truth = subset.ground_truth.loc[mutual_index]
        window_predictions = subset.predictions.loc[mutual_index]

        # If there is not enough data in the window, then skip it
        window_coverage = (
            get_timeseries_coverage(data=window_ground_truth, sample_interval=subset.sample_interval) / window.size
        )
        if window_coverage < window.minimum_coverage:
            continue

        yield (
            window_timestamp,
            EvaluationSubset.create(
                ground_truth=window_ground_truth,
                predictions=window_predictions,
                sample_interval=subset.sample_interval,
            ),
        )


def get_timeseries_coverage(data: pd.DataFrame | pd.Series, sample_interval: timedelta) -> timedelta:
    """Calculates the total time span covered by a time series dataset.

    Guarantees:
    - Returns the product of unique timestamps count and sample interval
    - Operates only on data with DatetimeIndex

    Args:
        data: Time series data with DatetimeIndex
        sample_interval: Time interval between samples

    Returns:
        Total time coverage as a timedelta

    Raises:
        TypeError: If data.index is not a DatetimeIndex
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        msg = "DataFrame index must be a DatetimeIndex"
        raise TypeError(msg)

    return len(data.index.unique()) * sample_interval


__all__ = [
    "get_timeseries_coverage",
    "iterate_by_window",
    "iterate_subsets_by_window",
]
