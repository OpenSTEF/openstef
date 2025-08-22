# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Versioned time series dataset for tracking data availability over time.

This module provides the VersionedTimeSeriesDataset class, which extends basic
time series functionality to track when each data point became available. This
is crucial for realistic dataset construction for both backtesting and
forecasting, allowing for accurate simulation of real-time data availability
and revisions.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Self, cast

import numpy as np
import pandas as pd

from openstef_core.datasets.mixins import VersionedTimeSeriesMixin
from openstef_core.exceptions import MissingColumnsError
from openstef_core.utils import (
    timedelta_from_isoformat,
    timedelta_to_isoformat,
)

_logger = logging.getLogger(__name__)


class VersionedTimeSeriesDataset(VersionedTimeSeriesMixin):
    """A time series dataset that tracks data availability over time.

    This dataset extends the basic time series concept by maintaining version
    information for each data point, recording when each piece of data became
    available. This enables realistic backtesting by ensuring that only data
    that was actually available at a given time is used for predictions.

    Each row in the dataset represents a data point at a specific timestamp
    along with the time when that data became available. This allows for:

    - Accurate simulation of real-time forecasting scenarios
    - Handling of data revisions and late-arriving measurements
    - Point-in-time data reconstruction for backtesting

    The dataset maintains temporal ordering by both timestamp and availability time.

    Example:
        Create a versioned dataset with energy load data:

        >>> import pandas as pd
        >>> from datetime import datetime, timedelta
        >>> data = pd.DataFrame({
        ...     'timestamp': [datetime.fromisoformat('2025-01-01T10:00:00'),
        ...                   datetime.fromisoformat('2025-01-01T10:15:00'),
        ...                   datetime.fromisoformat('2025-01-01T10:00:00')],  # Revised data
        ...     'available_at': [datetime.fromisoformat('2025-01-01T10:05:00'),
        ...                      datetime.fromisoformat('2025-01-01T10:20:00'),
        ...                      datetime.fromisoformat('2025-01-01T10:30:00')],  # Later revision
        ...     'load': [100.0, 120.0, 105.0]  # 105.0 is revised value for 10:00
        ... })
        >>> dataset = VersionedTimeSeriesDataset(data, timedelta(minutes=15))
        >>> dataset.feature_names
        ['load']

    Note:
        When multiple versions of the same timestamp exist, `get_window` will
        return the latest version available before the specified time.
    """

    data: pd.DataFrame
    timestamp_column: str
    available_at_column: str
    _sample_interval: timedelta
    _index: pd.DatetimeIndex
    _feature_names: list[str]

    def __init__(
        self,
        data: pd.DataFrame,
        sample_interval: timedelta,
        timestamp_column: str = "timestamp",
        available_at_column: str = "available_at",
    ) -> None:
        """Initialize a VersionedTimeSeriesDataset with the given data and configuration.

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
        self.timestamp_column = timestamp_column
        self.available_at_column = available_at_column
        self._sample_interval = sample_interval
        self._index = cast(pd.DatetimeIndex, pd.DatetimeIndex(self.data[timestamp_column]))
        self._feature_names = list(set(self.data.columns) - {self.timestamp_column, self.available_at_column})

    @property
    def feature_names(self) -> list[str]:
        """Names of feature columns excluding timestamp and availability columns.

        Returns:
            List of column names that contain actual feature data.
        """
        return self._feature_names

    @property
    def sample_interval(self) -> timedelta:
        """Fixed time interval between consecutive data points.

        Returns:
            The sampling interval for this time series.
        """
        return self._sample_interval

    @property
    def index(self) -> pd.DatetimeIndex:
        """Datetime index based on timestamp column.

        Returns:
            DatetimeIndex derived from the timestamp column values.
        """
        return self._index

    def get_window(self, start: datetime, end: datetime, available_before: datetime | None = None) -> pd.DataFrame:
        """Retrieve a time window of data that was available before a specified time.

        Returns data points within the specified time range, considering only data
        that was available before the given availability cutoff. When multiple
        versions of the same timestamp exist, returns the latest version available
        before the cutoff.

        Args:
            start: Inclusive start time of the desired window.
            end: Exclusive end time of the desired window.
            available_before: Optional cutoff time for data availability.
                Only data available at or before this time is included.
                If None, all data is considered available.

        Returns:
            DataFrame with timestamp index containing feature data for the
            requested window. Missing timestamps are filled with NaN values
            to maintain regular intervals.

        Example:
            Get data window considering availability:

            >>> from datetime import datetime, timedelta
            >>> import pandas as pd
            >>> # Create test data
            >>> data = pd.DataFrame({
            ...     'timestamp': [datetime.fromisoformat('2025-01-01T10:00:00'),
            ...                   datetime.fromisoformat('2025-01-01T10:15:00'),
            ...                   datetime.fromisoformat('2025-01-01T10:30:00')],
            ...     'available_at': [datetime.fromisoformat('2025-01-01T10:05:00'),
            ...                      datetime.fromisoformat('2025-01-01T10:20:00'),
            ...                      datetime.fromisoformat('2025-01-01T10:35:00')],
            ...     'load': [100.0, 120.0, 110.0]
            ... })
            >>> dataset = VersionedTimeSeriesDataset(data, timedelta(minutes=15))
            >>> window = dataset.get_window(
            ...     start=datetime.fromisoformat('2025-01-01T10:00:00'),
            ...     end=datetime.fromisoformat('2025-01-01T10:30:00'),
            ...     available_before=datetime.fromisoformat('2025-01-01T10:25:00')
            ... )
            >>> len(window)  # Should have 2 rows since 10:30 data not available yet
            2

        Note:
            The returned DataFrame excludes the availability timestamp column
            and uses the timestamp column as the index.
        """
        start_idx = self.data[self.timestamp_column].searchsorted(start, side="left")
        end_idx = self.data[self.timestamp_column].searchsorted(end, side="left")
        subset = self.data.iloc[start_idx:end_idx]

        if available_before is not None:
            subset = subset[subset[self.available_at_column] <= available_before]

        window_range = pd.date_range(start=start, end=end, freq=self._sample_interval, inclusive="left")
        return (
            subset.drop_duplicates(subset=[self.timestamp_column], keep="last")
            .drop(columns=[self.available_at_column])
            .set_index(self.timestamp_column)
            .reindex(window_range, fill_value=np.nan)
        )

    def to_parquet(
        self,
        path: Path,
    ) -> None:
        """Save the versioned dataset to a parquet file.

        Stores the data and metadata (sample interval, and column configuration)
        in parquet for complete reconstruction.

        Args:
            path: File path where the dataset should be saved.

        Note:
            Metadata includes sample interval (as ISO 8601 duration), timestamp
            column name, availability column name, and sort status.
        """
        self.data.attrs["sample_interval"] = timedelta_to_isoformat(self._sample_interval)
        self.data.attrs["timestamp_column"] = self.timestamp_column
        self.data.attrs["available_at_column"] = self.available_at_column
        self.data.attrs["is_sorted"] = True
        self.data.to_parquet(path)

    @classmethod
    def from_parquet(
        cls,
        path: Path,
    ) -> Self:
        """Create a VersionedTimeSeriesDataset from a parquet file.

        Loads a complete versioned dataset from a parquet file created with
        the `to_parquet` method. Handles missing metadata gracefully with
        reasonable defaults.

        Args:
            path: Path to the parquet file to load.

        Returns:
            New VersionedTimeSeriesDataset instance reconstructed from the file.

        Example:
            Save and reload a versioned dataset:

            >>> from pathlib import Path
            >>> import tempfile
            >>> import pandas as pd
            >>> from datetime import datetime, timedelta
            >>> # Create test versioned dataset
            >>> data = pd.DataFrame({
            ...     'timestamp': [datetime.fromisoformat('2025-01-01T10:00:00'),
            ...                   datetime.fromisoformat('2025-01-01T10:15:00')],
            ...     'available_at': [datetime.fromisoformat('2025-01-01T10:05:00'),
            ...                      datetime.fromisoformat('2025-01-01T10:20:00')],
            ...     'load': [100.0, 120.0]
            ... })
            >>> dataset = VersionedTimeSeriesDataset(data, timedelta(minutes=15))
            >>> # Test save/load cycle
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     file_path = Path(tmpdir) / "versioned_data.parquet"
            ...     dataset.to_parquet(file_path)
            ...     loaded = VersionedTimeSeriesDataset.from_parquet(file_path)
            ...     loaded.feature_names == dataset.feature_names
            True

        Note:
            Missing metadata attributes default to: sample_interval='PT15M',
            timestamp_column='timestamp', available_at_column='available_at'.
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


__all__ = ["VersionedTimeSeriesDataset"]
