# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Self, cast

import numpy as np
import pandas as pd

from openstef_core.datasets.mixins import VersionedTimeseriesMixin
from openstef_core.exceptions import MissingColumnsError
from openstef_core.utils import (
    timedelta_from_isoformat,
    timedelta_to_isoformat,
)

_logger = logging.getLogger(__name__)


class VersionedTimeseriesDataset(VersionedTimeseriesMixin):
    data: pd.DataFrame
    _sample_interval: timedelta
    _index: pd.DatetimeIndex
    _timestamp_column: str
    _available_at_column: str
    _feature_names: list[str]

    def __init__(
        self,
        data: pd.DataFrame,
        sample_interval: timedelta,
        timestamp_column: str = "timestamp",
        available_at_column: str = "available_at",
    ) -> None:
        """Initialize a VersionedTimeseriesDataset with the given data and configuration.

        Args:
            data: DataFrame containing the time series data.
            sample_interval: The regular interval between consecutive data points.
            timestamp_column: Name of the column containing timestamps. Default is 'timestamp'.
            available_at_column: Name of the column indicating when data became available.
                Default is 'available_at'.

        Raises:
            MissingColumnsError: If the required timestamp_column or available_at_column are missing.
        """
        missing_columns = {timestamp_column, available_at_column} - set(data.columns)
        if missing_columns:
            raise MissingColumnsError(missing_columns=list(missing_columns))

        if not data.attrs.get("is_sorted", False):
            data = data.sort_values(by=[timestamp_column, available_at_column], ascending=[True, True])
            data.attrs["is_sorted"] = True

        self.data = data
        self._sample_interval = sample_interval
        self._index = cast(pd.DatetimeIndex, pd.DatetimeIndex(self.data[timestamp_column]))
        self._timestamp_column = timestamp_column
        self._available_at_column = available_at_column
        self._feature_names = list(set(self.data.columns) - {self._timestamp_column, self._available_at_column})

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    @property
    def sample_interval(self) -> timedelta:
        return self._sample_interval

    @property
    def index(self) -> pd.DatetimeIndex:
        return self._index

    def get_window(self, start: datetime, end: datetime, available_before: datetime | None = None) -> pd.DataFrame:
        start_idx = self.data[self._timestamp_column].searchsorted(start, side="left")
        end_idx = self.data[self._timestamp_column].searchsorted(end, side="left")
        subset = self.data.iloc[start_idx:end_idx]

        if available_before is not None:
            subset = subset[subset[self._available_at_column] <= available_before]

        window_range = pd.date_range(start=start, end=end, freq=self._sample_interval, inclusive="left")
        return (
            subset.drop_duplicates(subset=[self._timestamp_column], keep="last")
            .drop(columns=[self._available_at_column])
            .set_index(self._timestamp_column)
            .reindex(window_range, fill_value=np.nan)
        )

    def to_parquet(
        self,
        path: Path,
    ) -> None:
        self.data.attrs["sample_interval"] = timedelta_to_isoformat(self._sample_interval)
        self.data.attrs["timestamp_column"] = self._timestamp_column
        self.data.attrs["available_at_column"] = self._available_at_column
        self.data.attrs["is_sorted"] = True
        self.data.to_parquet(path)

    @classmethod
    def from_parquet(
        cls,
        path: Path,
    ) -> Self:
        """Create a VersionedTimeseriesDataset from a parquet file.

        This factory method loads data from a parquet file and initializes a VersionedTimeseriesDataset.

        Returns:
           VersionedTimeseriesDataset: A new VersionedTimeseriesDataset initialized with the data from the parquet file.
        """
        data = pd.read_parquet(path)
        if "sample_interval" not in data.attrs:
            _logger.warning(
                "Parquet file does not contain 'sample_interval' attribute. Using default value of 15 minutes."
            )

        sample_interval = timedelta_from_isoformat(data.attrs.get("sample_interval", "PT15M"))
        timestamp_column = data.attrs.get("timestamp_column", "timestamp")
        available_at_column = data.attrs.get("available_at_column", "available_at")

        return cls(
            data=data,
            sample_interval=sample_interval,
            timestamp_column=timestamp_column,
            available_at_column=available_at_column,
        )


__all__ = ["VersionedTimeseriesDataset"]
