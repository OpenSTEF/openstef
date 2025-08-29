# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import logging
from datetime import datetime, timedelta
from typing import Self, cast, override

import pandas as pd
from pydantic import FilePath

from openstef_core.datasets.mixins import VersionedTimeSeriesMixin
from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.datasets.validation import validate_datetime_column
from openstef_core.exceptions import MissingColumnsError
from openstef_core.types import AvailableAt, LeadTime
from openstef_core.utils import timedelta_from_isoformat, timedelta_to_isoformat
from openstef_core.utils.pandas import sorted_range_slice_idxs

_logger = logging.getLogger(__name__)


class VersionedTimeSeriesPart(VersionedTimeSeriesMixin):
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

    Attributes:
        data: DataFrame containing the time series data with timestamp and availability columns.
        sample_interval: Fixed time interval between consecutive data points.
        timestamp_column: Name of the column containing timestamps.
        available_at_column: Name of the column indicating when data became available.
        index: Datetime index representing all timestamps in the dataset.
        feature_names: Names of all available features, excluding metadata columns.

    Example:
        Create a versioned dataset part for energy load data:

        >>> import pandas as pd
        >>> from datetime import datetime, timedelta
        >>> # Create data with delayed availability
        >>> data = pd.DataFrame({
        ...     'timestamp': [datetime.fromisoformat('2025-01-01T10:00:00'),
        ...                   datetime.fromisoformat('2025-01-01T10:15:00'),
        ...                   datetime.fromisoformat('2025-01-01T10:00:00')],  # Revised data
        ...     'available_at': [datetime.fromisoformat('2025-01-01T10:05:00'),
        ...                      datetime.fromisoformat('2025-01-01T10:20:00'),
        ...                      datetime.fromisoformat('2025-01-01T10:30:00')],  # Later revision
        ...     'load': [100.0, 120.0, 105.0]  # 105.0 is revised value for 10:00
        ... })
        >>> part = VersionedTimeSeriesPart(data, timedelta(minutes=15))
        >>> part.feature_names
        ['load']

        Filter by availability for backtesting:

        >>> from openstef_core.types import AvailableAt
        >>> # Only data available by end of day
        >>> filtered = part.filter_by_available_at(AvailableAt(lag_from_day=timedelta(hours=0)))
        >>> len(filtered.data)  # Will vary based on availability constraints
        3

        Get point-in-time snapshot:

        >>> snapshot = part.select_version(available_before=datetime.fromisoformat('2025-01-01T10:25:00'))
        >>> snapshot.feature_names
        ['load']

    Note:
        Data is automatically sorted by (timestamp, available_at) to ensure
        efficient filtering operations. When multiple versions of the same
        timestamp exist, the latest available version is used by select_version.
    """

    data: pd.DataFrame
    sample_interval: timedelta
    timestamp_column: str
    available_at_column: str
    index: pd.DatetimeIndex
    feature_names: list[str]

    def __init__(
        self,
        data: pd.DataFrame,
        sample_interval: timedelta,
        index: pd.DatetimeIndex | None = None,
        timestamp_column: str = "timestamp",
        available_at_column: str = "available_at",
    ) -> None:
        """Initialize a VersionedTimeSeriesPart with the given data and configuration.

        Args:
            data: DataFrame containing versioned time series data. Must include
                timestamp and available_at columns.
            sample_interval: The regular interval between consecutive data points.
            index: Optional predefined datetime index. If None, created from timestamp column.
            timestamp_column: Name of the column containing timestamps. Default is 'timestamp'.
            available_at_column: Name of the column indicating when data became available.
                Default is 'available_at'.

        Raises:
            MissingColumnsError: If required timestamp_column or available_at_column are missing.

        Note:
            Data is automatically sorted by (timestamp, available_at) ascending to ensure
            efficient filtering operations. The 'is_sorted' attribute tracks this state.
        """
        # Validate required columns
        missing_columns = {timestamp_column, available_at_column} - set(data.columns)
        if missing_columns:
            raise MissingColumnsError(missing_columns=list(missing_columns))

        # Validate timestamp and available_at columns types
        validate_datetime_column(data[timestamp_column], timestamp_column)
        validate_datetime_column(data[available_at_column], available_at_column)

        # Ensure invariant: data is at all times sorted by (timestamp, available_at) asc.
        if not data.attrs.get("is_sorted", False):
            data = data.sort_values(by=[timestamp_column, available_at_column], ascending=[True, True])
            data.attrs["is_sorted"] = True

        self.data = data
        self.sample_interval = sample_interval
        self.timestamp_column = timestamp_column
        self.available_at_column = available_at_column
        self.index = (
            index if index is not None else cast(pd.DatetimeIndex, pd.DatetimeIndex(self.data[timestamp_column]))
        )
        self.feature_names = list(set(self.data.columns) - {self.timestamp_column, self.available_at_column})

    @override
    def filter_by_range(self, start: datetime | None = None, end: datetime | None = None) -> Self:
        start_idx, end_idx = sorted_range_slice_idxs(data=self.data[self.timestamp_column], start=start, end=end)
        data_filtered = self.data.iloc[start_idx:end_idx]
        return self._copy_with_data(data_filtered)

    @override
    def filter_by_available_at(self, available_at: AvailableAt) -> Self:
        cutoff = self.data[self.timestamp_column].dt.floor("D") - pd.Timedelta(available_at.lag_from_day)
        data_filtered = self.data[self.data[self.available_at_column] <= cutoff]
        return self._copy_with_data(data_filtered, reindex=False)

    @override
    def filter_by_lead_time(self, lead_time: LeadTime) -> Self:
        cutoff = self.data[self.timestamp_column] - pd.Timedelta(lead_time.value)
        data_filtered = self.data[self.data[self.available_at_column] <= cutoff]
        return self._copy_with_data(data_filtered, reindex=False)

    @override
    def select_version(self, available_before: datetime | None = None) -> TimeSeriesDataset:
        if available_before is not None:
            data = self.data[self.data[self.available_at_column] <= available_before]
        else:
            data = self.data

        return TimeSeriesDataset(
            data=(
                data.drop_duplicates(subset=self.timestamp_column, keep="last")
                .drop(columns=[self.available_at_column])
                .set_index(self.timestamp_column)
            ),
            sample_interval=self.sample_interval,
        )

    def _copy_with_data(self, new_data: pd.DataFrame, *, reindex: bool = True) -> Self:
        return self.__class__(
            data=new_data,
            sample_interval=self.sample_interval,
            index=self.index if not reindex else None,
            timestamp_column=self.timestamp_column,
            available_at_column=self.available_at_column,
        )

    def to_parquet(self, path: FilePath) -> None:
        """Save the versioned dataset part to a parquet file.

        Stores the data and metadata (sample interval, column configuration, and
        sort status) in parquet format for complete reconstruction. This enables
        efficient persistence of versioned data with all necessary metadata.

        Args:
            path: File path where the dataset part should be saved.

        Note:
            Metadata includes sample interval (as ISO 8601 duration), timestamp
            column name, availability column name, and sort status to ensure
            proper reconstruction.

        Example:
            Save and reload a dataset part:

            >>> import tempfile
            >>> from pathlib import Path
            >>> # Assuming 'part' is a VersionedTimeSeriesPart instance
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     file_path = Path(tmpdir) / "data_part.parquet"
            ...     part.to_parquet(file_path)
            ...     loaded = VersionedTimeSeriesPart.read_parquet(file_path)
            ...     loaded.feature_names == part.feature_names
            True
        """
        self.data.attrs["sample_interval"] = timedelta_to_isoformat(self.sample_interval)
        self.data.attrs["timestamp_column"] = self.timestamp_column
        self.data.attrs["available_at_column"] = self.available_at_column
        self.data.attrs["is_sorted"] = True
        self.data.to_parquet(path)

    @classmethod
    def read_parquet(cls, path: FilePath) -> Self:
        """Create a VersionedTimeSeriesPart from a parquet file.

        Loads a complete versioned dataset part from a parquet file created with
        the `to_parquet` method. Handles missing metadata gracefully with
        reasonable defaults to ensure robust data loading.

        Args:
            path: Path to the parquet file to load.

        Returns:
            New VersionedTimeSeriesPart instance reconstructed from the file.

        Example:
            Load a previously saved dataset part:

            >>> from pathlib import Path
            >>> # Load from a saved parquet file
            >>> loaded_part = VersionedTimeSeriesPart.read_parquet("data_part.parquet")
            >>> loaded_part.feature_names
            ['load', 'temperature']

        Note:
            Missing metadata attributes default to: sample_interval='PT15M',
            timestamp_column='timestamp', available_at_column='available_at'.
            A warning is logged if metadata is missing.
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


__all__ = ["VersionedTimeSeriesPart"]
