# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Testing utilities for comparing pandas objects.

Provides matcher classes for use in test assertions when comparing pandas
DataFrames and Series with equality semantics.
"""

from datetime import datetime, timedelta
from typing import Any, override

import pandas as pd

from openstef_core.datasets import TimeSeriesDataset


class IsSamePandas:
    """Utility class to allow comparison of pandas DataFrames in assertion / calls."""

    def __init__(self, pandas_obj: pd.DataFrame | pd.Series):
        """Matcher to check if two DataFrames are equal."""
        self.pandas_obj = pandas_obj

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self.pandas_obj)) and self.pandas_obj.equals(other)  # type: ignore

    @override
    def __hash__(self) -> int:
        return hash(self.pandas_obj)


def assert_timeseries_equal(actual: TimeSeriesDataset, expected: TimeSeriesDataset):
    """Assert that two TimeSeriesDataset objects are equal."""
    pd.testing.assert_frame_equal(actual.data, expected.data)
    assert actual.sample_interval == expected.sample_interval, (  # noqa: S101 - exception - testing utility
        f"Sample intervals differ: {actual.sample_interval} != {expected.sample_interval}"
    )


def create_timeseries_dataset(
    index: pd.DatetimeIndex,
    available_ats: pd.Series | list[datetime] | pd.DatetimeIndex | None = None,
    horizons: pd.Series | list[timedelta] | None = None,
    sample_interval: timedelta = timedelta(hours=1),
    **kwargs: pd.Series | list[Any] | pd.DatetimeIndex,
) -> TimeSeriesDataset:
    """Create a TimeSeriesDataset for testing purposes.

    Args:
        index: Datetime index for the dataset.
        available_ats: Optional available_at timestamps for each data point.
        horizons: Optional forecast horizons for each data point.
        sample_interval: Time interval between consecutive samples.
        **kwargs: Additional columns to include in the dataset.

    Returns:
        TimeSeriesDataset with the specified structure.
    """
    data = kwargs
    if available_ats is not None:
        data["available_at"] = available_ats
    elif horizons is not None:
        data["horizon"] = horizons

    return TimeSeriesDataset(data=pd.DataFrame(data=data, index=index), sample_interval=sample_interval)
