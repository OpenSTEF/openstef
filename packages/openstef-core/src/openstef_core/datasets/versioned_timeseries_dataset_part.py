# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Versioned time series dataset parts for realistic data availability modeling.

This module provides the VersionedTimeSeriesPart class, which represents a single
segment of versioned time series data. Each data point includes both a timestamp
(when the event occurred) and an availability timestamp (when the data became
available for use).

The class enables realistic backtesting and forecasting scenarios by modeling
data publication delays, revisions, and availability constraints that occur
in real-world data pipelines.
"""

import logging
from datetime import timedelta
from typing import Self, override

import pandas as pd

from openstef_core.datasets.mixins import DatasetMixin, VersionedTimeSeriesMixin
from openstef_core.datasets.utils.validation import validate_datetime_column, validate_required_columns
from openstef_core.utils import timedelta_from_isoformat, timedelta_to_isoformat

_logger = logging.getLogger(__name__)


class VersionedTimeSeriesPart(VersionedTimeSeriesMixin, DatasetMixin):
    """A single part of a versioned time series dataset with enhanced filtering capabilities.

    This class represents a contiguous segment of versioned time series data,
    where each data point has both a timestamp (when it occurred) and an
    availability time (when it became available for use). It provides advanced
    filtering methods to simulate realistic data availability constraints for
    backtesting and forecasting.

    Key features include:
    - Filtering by time ranges for windowed analysis
    - AvailableAt constraints to simulate daily data publication schedules
    - Lead time filtering for realistic prediction scenarios
    - Point-in-time data reconstruction for backtesting

    This class is designed to be composed into larger datasets via
    VersionedTimeSeriesDataset, enabling flexible dataset construction
    from multiple data sources.

    Example:
        Create a versioned dataset part for energy load data:

        >>> import pandas as pd
        >>> from datetime import datetime, timedelta
        >>> # Create data with delayed availability
        >>> data = pd.DataFrame({
        ...     'timestamp': pd.to_datetime(['2025-01-01T10:00:00',
        ...                                 '2025-01-01T10:15:00']),
        ...     'available_at': pd.to_datetime(['2024-12-31T16:00:00',
        ...                                     '2024-12-31T17:00:00']),
        ...     'load': [100.0, 120.0]
        ... })
        >>> part = VersionedTimeSeriesPart(data, sample_interval=timedelta(minutes=15))
        >>> part.feature_names
        ['load']

        Get point-in-time snapshot:

        >>> snapshot = part.select_version(available_before=datetime(2025, 1, 1, 10, 30))
        >>> snapshot.data.shape[0] >= 1  # At least one data point
        True

    Note:
        Data is automatically sorted by (timestamp, available_at) to ensure
        efficient filtering operations. When multiple versions of the same
        timestamp exist, the latest available version is used by select_version.
    """

    data: pd.DataFrame
    _sample_interval: timedelta
    _feature_names: list[str]
    _index: pd.DatetimeIndex

    def __init__(self, data: pd.DataFrame, sample_interval: timedelta) -> None:
        # Validate timestamp and available_at columns types
        validate_required_columns(self.data, [self.timestamp_column, self.available_at_column])
        validate_datetime_column(self.data[self.timestamp_column])
        validate_datetime_column(self.data[self.available_at_column])

        # Ensure invariant: data is at all times sorted by (timestamp, available_at) asc.
        if not data.attrs.get("is_sorted", False):
            data = data.sort_values(by=[self.timestamp_column, self.available_at_column], ascending=[True, True])
        data.attrs["is_sorted"] = True

        self.data = data
        self._sample_interval = sample_interval
        self._feature_names = [
            col for col in data.columns if col not in {self.timestamp_column, self.available_at_column}
        ]
        self._index = pd.DatetimeIndex(self.data[self.timestamp_column])

    @property
    @override
    def index(self) -> pd.DatetimeIndex:
        return self._index

    @property
    @override
    def feature_names(self) -> list[str]:
        return self._feature_names

    @property
    @override
    def sample_interval(self) -> timedelta:
        return self._sample_interval

    @override
    def _to_versioned_dataframe(self) -> pd.DataFrame:
        return self.data

    @override
    def _from_versioned_dataframe(self, data: pd.DataFrame) -> Self:
        return self.__class__(data=data, sample_interval=self.sample_interval)

    @override
    def _to_pandas(self) -> pd.DataFrame:
        self.data.attrs["sample_interval"] = timedelta_to_isoformat(self.sample_interval)
        self.data.attrs["timestamp_column"] = self.timestamp_column
        self.data.attrs["available_at_column"] = self.available_at_column
        self.data.attrs["is_sorted"] = True
        return self.data

    @override
    @classmethod
    def _from_pandas(cls, df: pd.DataFrame) -> Self:
        sample_interval = timedelta_from_isoformat(df.attrs.get("sample_interval", "PT15M"))

        return cls(data=df, sample_interval=sample_interval)


__all__ = ["VersionedTimeSeriesPart"]
