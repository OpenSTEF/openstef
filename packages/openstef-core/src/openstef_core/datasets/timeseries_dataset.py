# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Regular time series dataset implementation.

This module provides the TimeSeriesDataset class, which handles time series data
with consistent sampling intervals. It offers basic operations for data access,
persistence, and temporal metadata management.

The implementation ensures data integrity through automatic sorting and provides
convenient methods for saving/loading datasets while preserving all metadata.
"""

import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import ClassVar, Concatenate, Self, cast, override

import pandas as pd
from pydantic import FilePath

from openstef_core.datasets.mixins import DatasetMixin, TimeSeriesMixin
from openstef_core.datasets.validation import (
    TimeSeriesValidationError,
    validate_datetime_column,
    validate_timedelta_column,
)
from openstef_core.types import AvailableAt, LeadTime
from openstef_core.utils import timedelta_from_isoformat, timedelta_to_isoformat
from openstef_core.utils.pandas import unsafe_sorted_range_slice_idxs

_logger = logging.getLogger(__name__)


class TimeSeriesDataset(TimeSeriesMixin, DatasetMixin):  # noqa: PLR0904 - important utility class, allow too many public methods
    """A time series dataset with regular sampling intervals and optional versioning.

    This class represents time series data with a consistent sampling interval
    and provides operations for data access, persistence, and filtering. It supports
    both regular time series and versioned datasets where data availability is tracked
    over time through either a horizon column or an available_at column.

    The dataset automatically detects versioning:
    - If a horizon column exists, data is versioned by forecast horizon
    - If an available_at column exists, data is versioned by availability time
    - Otherwise, data is treated as a regular time series

    The dataset guarantees:
        - Data is sorted by timestamp in ascending order
        - Consistent sampling interval across all data points
        - DateTime index for temporal operations

    Attributes:
        data: DataFrame containing the time series data indexed by timestamp.
        horizon_column: Name of the column storing forecast horizons (if versioned by horizon).
        available_at_column: Name of the column storing availability times (if versioned).

    Example:
        Create a simple time series dataset:

        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> data = pd.DataFrame({
        ...     'temperature': [20.1, 22.3, 21.5],
        ...     'load': [100, 120, 110]
        ... }, index=pd.date_range('2025-01-01', periods=3, freq='15min'))
        >>> dataset = TimeSeriesDataset(data, sample_interval=timedelta(minutes=15))
        >>> dataset.feature_names
        ['temperature', 'load']
        >>> dataset.is_versioned
        False

        Create a versioned dataset with horizons:

        >>> data_with_horizon = pd.DataFrame({
        ...     'load': [100, 120],
        ...     'horizon': pd.to_timedelta(['1h', '2h'])
        ... }, index=pd.date_range('2025-01-01', periods=2, freq='1h'))
        >>> dataset = TimeSeriesDataset(data_with_horizon, sample_interval=timedelta(hours=1))
        >>> dataset.is_versioned
        True
        >>> dataset.horizons is not None
        True
    """

    index_name: ClassVar[str] = "timestamp"

    data: pd.DataFrame
    horizon_column: str
    available_at_column: str

    _sample_interval: timedelta
    _version_column: str | None
    _feature_names: list[str]
    _horizons: list[LeadTime] | None
    _internal_columns: set[str]

    def __init__(
        self,
        data: pd.DataFrame,
        sample_interval: timedelta = timedelta(minutes=15),
        *,
        horizon_column: str = "horizon",
        available_at_column: str = "available_at",
        is_sorted: bool = False,
    ) -> None:
        """Initialize a time series dataset.

        The dataset automatically detects whether it's versioned based on column presence:
        - If horizon_column exists: versioned by forecast horizon
        - If available_at_column exists: versioned by availability time
        - Otherwise: regular time series

        Args:
            data: DataFrame with DatetimeIndex containing the time series data.
            sample_interval: Fixed interval between consecutive data points.
            horizon_column: Name of the column storing forecast horizons.
            available_at_column: Name of the column storing availability times.
            is_sorted: Whether the data is sorted by timestamp.

        Raises:
            TypeError: If data index is not a pandas DatetimeIndex or if versioning
                columns have incorrect types.
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("Data index must be a pandas DatetimeIndex.")

        self.data = data
        self.horizon_column = horizon_column
        self.available_at_column = available_at_column
        self._sample_interval = sample_interval
        self._internal_columns = set()
        data.index.name = self.index_name

        if self.horizon_column in data.columns:
            validate_timedelta_column(data[self.horizon_column])
            self._version_column = self.horizon_column
            self._internal_columns.add(self.horizon_column)
            self._feature_names = [col for col in data.columns if col not in self._internal_columns]
            self._horizons = list({LeadTime(value=td) for td in data[self.horizon_column].unique()})
        elif self.available_at_column in data.columns:
            validate_datetime_column(data[self.available_at_column])
            self._version_column = self.available_at_column
            self._internal_columns.add(self.available_at_column)
            self._feature_names = [col for col in data.columns if col not in self._internal_columns]
            self._horizons = None
        else:
            self._version_column = None
            self._feature_names = data.columns.to_list()
            self._horizons = None

        # Ensure invariants: data is sorted by timestamp
        if not is_sorted:
            if self._version_column == self.available_at_column:
                self.data = self.data.sort_values(
                    by=[self.index_name, self.available_at_column],
                    ascending=[True, False],  # timestamp ascending, available_at descending
                )
            elif self._version_column == self.horizon_column:
                self.data = self.data.sort_values(
                    by=[self.index_name, self.horizon_column],
                    ascending=[True, True],  # timestamp ascending, horizon ascending
                )
            else:
                self.data = self.data.sort_index(ascending=True)

    @property
    @override
    def index(self) -> pd.DatetimeIndex:
        return cast(pd.DatetimeIndex, self.data.index)

    @property
    @override
    def sample_interval(self) -> timedelta:
        return self._sample_interval

    @property
    @override
    def feature_names(self) -> list[str]:
        return self._feature_names

    @property
    @override
    def is_versioned(self) -> bool:
        return self._version_column is not None

    @property
    def horizons(self) -> list[LeadTime] | None:
        """Get the list of forecast horizons present in the dataset.

        Returns:
            List of unique forecast horizons if the dataset is versioned by horizons,
            None otherwise.
        """
        return self._horizons

    @property
    def available_at_series(self) -> pd.Series | None:
        """Get the availability times as a pandas Series.

        Returns:
            Series containing availability times indexed by timestamp if versioned,
            None for non-versioned datasets.
        """
        if self._version_column is None:
            return None
        if self._version_column == self.available_at_column:
            return self.data[self.available_at_column]
        return self.data.index - self.data[self.horizon_column]

    @property
    def lead_time_series(self) -> pd.Series | None:
        """Get the lead times as a pandas Series.

        Lead time is the gap between when data became available and the timestamp.

        Returns:
            Series containing lead times indexed by timestamp if versioned,
            None for non-versioned datasets.
        """
        if self._version_column is None:
            return None
        if self._version_column == self.horizon_column:
            return self.data[self.horizon_column]
        return self.data.index - self.data[self.available_at_column]

    def _copy_with_data(self, data: pd.DataFrame) -> Self:
        # Fast way to copy self with new data and skipping validation since invariants are preserved.
        new_instance = object.__new__(self.__class__)
        new_instance.__dict__.update(self.__dict__)
        new_instance.data = data
        new_instance._feature_names = [col for col in data.columns if col not in self._internal_columns]  # noqa: SLF001
        return new_instance

    @override
    def filter_by_range(self, start: datetime | None = None, end: datetime | None = None) -> Self:
        if start is None and end is None:
            return self

        start_idx, end_idx = unsafe_sorted_range_slice_idxs(data=self.data.index, start=start, end=end)
        data_filtered = self.data.iloc[start_idx:end_idx]
        return self._copy_with_data(data=data_filtered)

    @override
    def filter_by_available_before(self, available_before: datetime) -> Self:
        available_at_series = self.available_at_series
        if available_at_series is None:
            return self

        data_filtered = self.data[available_at_series <= available_before]
        return self._copy_with_data(data=data_filtered)

    @override
    def filter_by_available_at(self, available_at: AvailableAt) -> Self:
        available_at_series = self.available_at_series
        if available_at_series is None:
            return self

        cutoff = self.index.floor("D") - pd.Timedelta(available_at.lag_from_day)
        data_filtered = self.data[available_at_series <= cutoff]
        return self._copy_with_data(data=data_filtered)

    @override
    def filter_by_lead_time(self, lead_time: LeadTime) -> Self:
        lead_time_series = self.lead_time_series
        if lead_time_series is None:
            return self

        data_filtered = self.data[lead_time_series >= lead_time.value]
        return self._copy_with_data(data=data_filtered)

    @override
    def select_version(self) -> Self:
        if self._version_column is None:
            return self

        data_selected = self.data[~self.data.index.duplicated(keep="first")].drop(columns=[self._version_column])
        result = self._copy_with_data(data=data_selected)
        result._horizons = None  # noqa: SLF001 - Clear out all versioning metadata
        result._version_column = None  # noqa: SLF001 - Clear out all versioning metadata
        return result

    def filter_index(self, mask: pd.Index) -> Self:
        """Filter dataset to include only timestamps present in the mask.

        Returns:
            New dataset containing only rows with timestamps in the mask.
        """
        data_filtered = self.data.loc[self.index.isin(mask)]  # pyright: ignore[reportUnknownMemberType]

        return self._copy_with_data(data=data_filtered)

    def select_horizon(self, horizon: LeadTime) -> Self:
        """Select data for a specific forecast horizon.

        Args:
            horizon: The forecast horizon to filter the dataset by.

        Returns:
            A new TimeSeriesDataset instance containing only data for the specified horizon.
        """
        if self.horizons is None:
            return self

        data_selected = self.data[self.lead_time_series == horizon.value]
        return self._copy_with_data(data=data_selected)

    def to_pandas(self) -> pd.DataFrame:
        """Convert the dataset to a pandas DataFrame with metadata stored in attrs.

        Stores sample_interval, available_at_column, and horizon_column in the
        DataFrame's attrs dictionary for later reconstruction.

        Returns:
            DataFrame with dataset data and metadata in attrs.
        """
        df = self.data.copy()
        df.attrs["sample_interval"] = timedelta_to_isoformat(self.sample_interval)
        df.attrs["available_at_column"] = self.available_at_column
        df.attrs["horizon_column"] = self.horizon_column
        return df

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        *,
        sample_interval: timedelta | None = None,
        available_at_column: str = "available_at",
        horizon_column: str = "horizon",
    ) -> Self:
        """Create a dataset instance from a pandas DataFrame with metadata in attrs.

        Reads sample_interval, available_at_column, and horizon_column from the
        DataFrame's attrs dictionary.

        Args:
            df: DataFrame containing dataset data with metadata in attrs.
            sample_interval: Fixed interval between consecutive data points. If None, reads from attrs.
            available_at_column: Name of the column storing availability times.
            horizon_column: Name of the column storing forecast horizons.

        Returns:
            New TimeSeriesDataset instance reconstructed from the DataFrame.
        """
        if sample_interval is None:
            if "sample_interval" not in df.attrs:
                _logger.warning(
                    "Parquet file does not contain 'sample_interval' attribute. Using default value of 15 minutes."
                )
                sample_interval = timedelta(minutes=15)
            else:
                sample_interval = timedelta_from_isoformat(df.attrs.get("sample_interval", "PT15M"))

        available_at_column = df.attrs.get("available_at_column", available_at_column)
        horizon_column = df.attrs.get("horizon_column", horizon_column)

        return cls(
            data=df,
            sample_interval=sample_interval,
            available_at_column=available_at_column,
            horizon_column=horizon_column,
        )

    @override
    def to_parquet(self, path: FilePath) -> None:
        self.to_pandas().to_parquet(path=path)

    @override
    @classmethod
    def read_parquet(
        cls,
        path: FilePath,
        *,
        sample_interval: timedelta | None = None,
        timestamp_column: str = "timestamp",
        available_at_column: str = "available_at",
        horizon_column: str = "horizon",
    ) -> Self:
        df = pd.read_parquet(path=path)  # pyright: ignore[reportUnknownMemberType]
        if not isinstance(df.index, pd.DatetimeIndex):
            if timestamp_column not in df.columns:
                raise TimeSeriesValidationError(
                    "Parquet file does not have a DatetimeIndex. Please provide 'timestamp_column' to read the dataset."
                )
            df = df.set_index(timestamp_column)

        return cls.from_pandas(
            df=df,
            sample_interval=sample_interval or None,
            available_at_column=available_at_column,
            horizon_column=horizon_column,
        )

    def pipe_pandas[**P](
        self, func: Callable[Concatenate[pd.DataFrame, P], pd.DataFrame], *args: P.args, **kwargs: P.kwargs
    ) -> Self:
        """Apply a pandas DataFrame transformation function to the dataset.

        Executes a function on the underlying DataFrame and wraps the result
        back into a TimeSeriesDataset, preserving all metadata.

        Returns:
            New dataset with the transformation applied.
        """
        data_new = func(self.data, *args, **kwargs)
        return self._copy_with_data(data=data_new)

    def select_features(self, feature_names: list[str]) -> "TimeSeriesDataset":
        """Select a subset of features from the dataset.

        Args:
            feature_names: List of feature column names to retain in the dataset.

        Returns:
            A new TimeSeriesDataset instance containing only the specified features.
        """
        columns_to_select = list(feature_names)
        if self._version_column is not None:
            columns_to_select.append(self._version_column)
        data_selected = self.data[columns_to_select]
        return TimeSeriesDataset(
            data=data_selected,
            is_sorted=True,
        )

    def copy_with(self, data: pd.DataFrame, *, is_sorted: bool = False) -> "TimeSeriesDataset":
        """Create a copy of the dataset with new data.

        Args:
            data: New DataFrame to use for the dataset.
            is_sorted: Whether the new data is already sorted by timestamp.

        Returns:
            New TimeSeriesDataset instance with the provided data and same metadata.
        """
        return TimeSeriesDataset(
            data=data,
            sample_interval=self.sample_interval,
            horizon_column=self.horizon_column,
            available_at_column=self.available_at_column,
            is_sorted=is_sorted,
        )


def validate_horizons_present(dataset: TimeSeriesDataset, horizons: list[LeadTime]) -> None:
    """Validate that the specified forecast horizons are present in the dataset.

    Args:
        dataset: The time series dataset to validate.
        horizons: List of forecast horizons to check for presence in the dataset.

    Raises:
        TimeSeriesValidationError: If any of the specified horizons are not present.
    """
    if dataset.horizons is None and len(horizons) == 1:
        return  # Non-versioned dataset can satisfy single-horizon requests

    required_horizons = set(horizons or [])
    missing_horizons = [h for h in horizons if h not in required_horizons]
    if missing_horizons:
        raise TimeSeriesValidationError("Missing forecast horizons: " + ", ".join(map(str, missing_horizons)))


__all__ = ["TimeSeriesDataset", "validate_horizons_present"]
