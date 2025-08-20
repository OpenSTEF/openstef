# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Self, cast

import numpy as np
import pandas as pd

from openstef_beam.core.utils.pydantic import timedelta_from_isoformat, timedelta_to_isoformat
from openstef_beam.datasets.base import TimeseriesDataset

logger = logging.getLogger(__name__)


class PandasTimeseriesDataset(TimeseriesDataset):
    """Implementation of BaseDataset using pandas DataFrame as the underlying data structure.

    This class provides efficient access to time series data with temporal availability tracking.
    It maintains the invariant that data points are uniquely identified by their timestamp
    and can be filtered by when they became available.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        sample_interval: timedelta,
        timestamp_column: str = "timestamp",
        available_at_column: str = "available_at",
    ) -> None:
        """Initialize a PandasDataset with the given data and configuration.

        Args:
            data: DataFrame containing the time series data.
            sample_interval: The regular interval between consecutive data points.
            timestamp_column: Name of the column containing timestamps. Default is 'timestamp'.
            available_at_column: Name of the column indicating when data became available.
                Default is 'available_at'.

        Raises:
            ValueError: If the required timestamp_column or available_at_column are missing.
        """
        if {timestamp_column, available_at_column} - set(data.columns):
            msg = f"Columns {timestamp_column} and {available_at_column} must be present in the data."
            raise ValueError(msg)

        if not data.attrs.get("is_sorted", False):
            data = data.sort_values(by=[timestamp_column, available_at_column], ascending=[True, True])

        self._sample_interval = sample_interval
        self._data = data
        self._index = cast(pd.DatetimeIndex, pd.DatetimeIndex(self._data[timestamp_column]))
        self._timestamp_column = timestamp_column
        self._available_at_column = available_at_column
        self._feature_names = list(set(self._data.columns) - {self._timestamp_column, self._available_at_column})

    @property
    def sample_interval(self) -> timedelta:
        return self._sample_interval

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    @property
    def index(self) -> pd.DatetimeIndex:
        return self._index

    @property
    def data(self) -> pd.DataFrame:
        """Returns the underlying DataFrame containing all time series data.

        Returns:
           pd.DataFrame: The complete dataset.
        """
        return self._data

    @property
    def timestamp_column(self) -> str:
        """Returns the name of the column containing timestamps.

        Returns:
            str: Name of the timestamp column.
        """
        return self._timestamp_column

    @property
    def available_at_column(self) -> str:
        """Returns the name of the column indicating when data became available.

        Returns:
           str: Name of the availability timestamp column.
        """
        return self._available_at_column

    def get_window(self, start: datetime, end: datetime, available_before: datetime | None = None) -> pd.DataFrame:
        start_idx = self._data[self.timestamp_column].searchsorted(start, side="left")
        end_idx = self._data[self.timestamp_column].searchsorted(end, side="left")
        subset = self._data.iloc[start_idx:end_idx]

        if available_before is not None:
            subset = subset[subset[self.available_at_column] <= available_before]

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
        self.data.attrs["sample_interval"] = timedelta_to_isoformat(self.sample_interval)
        self.data.attrs["timestamp_column"] = self.timestamp_column
        self.data.attrs["available_at_column"] = self.available_at_column
        self.data.attrs["is_sorted"] = True
        self.data.to_parquet(path)

    @classmethod
    def from_parquet(
        cls,
        path: Path,
    ) -> Self:
        """Create a PandasDataset from a parquet file.

        This factory method loads data from a parquet file and initializes a PandasDataset.

        Returns:
           PandasDataset: A new PandasDataset initialized with the data from the parquet file.
        """
        data = pd.read_parquet(path)
        if "sample_interval" not in data.attrs:
            logger.warning(
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
