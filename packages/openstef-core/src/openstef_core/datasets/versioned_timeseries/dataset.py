# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Versioned time series dataset for efficient multi-part composition.

This module provides an enhanced implementation of versioned time series datasets
that track data availability over time. The new architecture supports composable
datasets from multiple parts and improved filtering capabilities for realistic
backtesting and forecasting scenarios.

The key improvement over the previous implementation is the ability to combine
multiple data sources while maintaining versioning information, enabling more
flexible dataset construction for complex forecasting pipelines.
"""

import functools
import logging
import operator
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Literal, Self, cast, override

import pandas as pd
from pydantic import FilePath

from openstef_core.datasets.mixins import VersionedTimeSeriesMixin
from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.datasets.validation import validate_disjoint_columns, validate_same_sample_intervals
from openstef_core.datasets.versioned_timeseries.dataset_part import VersionedTimeSeriesPart
from openstef_core.exceptions import TimeSeriesValidationError
from openstef_core.types import AvailableAt, LeadTime
from openstef_core.utils.pandas import unsafe_sorted_range_slice_idxs

_logger = logging.getLogger(__name__)


type ConcatMode = Literal["left", "outer", "inner"]


class VersionedTimeSeriesDataset(VersionedTimeSeriesMixin):
    """A versioned time series dataset composed of multiple data parts.

    This class combines multiple VersionedTimeSeriesPart instances into a unified
    dataset that tracks data availability over time. It provides methods to filter
    datasets by time ranges, availability constraints, and lead times, as well as
    select specific versions of the data for point-in-time reconstruction.

    The dataset is particularly useful for realistic backtesting scenarios where
    data arrives with delays or gets revised over time.

    Key motivation: This architecture solves the O(n²) space complexity problem
    that occurs when concatenating DataFrames with misaligned (timestamp, available_at)
    pairs. Instead of immediately combining data, it uses lazy composition that
    delays actual DataFrame concatenation until select_version() is called.

    Attributes:
        data_parts: List of VersionedTimeSeriesPart instances that compose this dataset.
        index: Datetime index representing all timestamps across all data parts.
        sample_interval: Fixed time interval between consecutive data points.
        feature_names: Names of all available features across all data parts.

    Example:
        Create a versioned dataset by combining multiple data parts:

        >>> import pandas as pd
        >>> from datetime import datetime, timedelta
        >>>
        >>> # Create weather data part
        >>> weather_data = pd.DataFrame({
        ...     'timestamp': [datetime(2025, 1, 1, 10, 0)],
        ...     'available_at': [datetime(2025, 1, 1, 16, 0)],
        ...     'temperature': [20.5]
        ... })
        >>> weather_part = VersionedTimeSeriesPart(weather_data, timedelta(hours=1))
        >>>
        >>> # Create load data part
        >>> load_data = pd.DataFrame({
        ...     'timestamp': [datetime(2025, 1, 1, 10, 0)],
        ...     'available_at': [datetime(2025, 1, 1, 11, 0)],
        ...     'load': [150.0]
        ... })
        >>> load_part = VersionedTimeSeriesPart(load_data, timedelta(hours=1))
        >>>
        >>> # Combine into versioned dataset
        >>> dataset = VersionedTimeSeriesDataset([weather_part, load_part])
        >>> sorted(dataset.feature_names)
        ['load', 'temperature']
        >>> dataset.sample_interval
        datetime.timedelta(seconds=3600)

        Get point-in-time snapshot of data available at specific time:

        >>> snapshot = dataset.select_version(available_before=datetime(2025, 1, 1, 18, 0))
        >>> sorted(snapshot.data.columns.tolist())
        ['load', 'temperature']

    Note:
        All data parts must have identical sample intervals and disjoint feature sets.
        The final dataset index is the union of all part indices, enabling flexible
        composition of data sources with different coverage periods.
    """

    data_parts: list[VersionedTimeSeriesPart]
    index: pd.DatetimeIndex
    sample_interval: timedelta
    feature_names: list[str]

    def __init__(
        self,
        data_parts: list[VersionedTimeSeriesPart],
        index: pd.DatetimeIndex | None = None,
    ) -> None:
        """Initialize a VersionedTimeSeriesDataset from multiple data parts.

        Combines multiple VersionedTimeSeriesPart instances into a unified dataset
        that maintains versioning information across all parts. Validates that
        parts have compatible configurations before combining.

        Args:
            data_parts: List of VersionedTimeSeriesPart instances to combine.
                Must have at least one part with disjoint feature sets and
                identical sample intervals.
            index: Optional predefined datetime index covering all parts.
                If None, computed as union of all part indices.

        Raises:
            TimeSeriesValidationError: If no data parts provided, or if parts have
                overlapping features or incompatible sample intervals.

        Example:
            Combine temperature and load data parts:

            >>> # Create separate data parts for different features
            >>> temp_data = pd.DataFrame({
            ...     'timestamp': [datetime.fromisoformat('2025-01-01T10:00:00')],
            ...     'available_at': [datetime.fromisoformat('2025-01-01T10:05:00')],
            ...     'temperature': [20.0]
            ... })
            >>> load_data = pd.DataFrame({
            ...     'timestamp': [datetime.fromisoformat('2025-01-01T10:00:00')],
            ...     'available_at': [datetime.fromisoformat('2025-01-01T10:10:00')],
            ...     'load': [100.0]
            ... })
            >>> temp_part = VersionedTimeSeriesPart(temp_data, timedelta(minutes=15))
            >>> load_part = VersionedTimeSeriesPart(load_data, timedelta(minutes=15))
            >>> dataset = VersionedTimeSeriesDataset([temp_part, load_part])
            >>> sorted(dataset.feature_names)
            ['load', 'temperature']
        """
        if not data_parts:
            raise TimeSeriesValidationError("At least one data part must be provided.")

        validate_same_sample_intervals(data_parts)
        validate_disjoint_columns(data_parts)

        self.data_parts = data_parts
        self.sample_interval = data_parts[0].sample_interval
        if index is not None:
            self.index = index
        else:
            union_fn = cast(Callable[[pd.DatetimeIndex, pd.DatetimeIndex], pd.DatetimeIndex], pd.DatetimeIndex.union)
            self.index = functools.reduce(union_fn, [part.index for part in data_parts])
        self.feature_names = functools.reduce(operator.iadd, [part.feature_names for part in data_parts], [])

    @classmethod
    def from_dataframe(
        cls,
        data: pd.DataFrame,
        sample_interval: timedelta,
        timestamp_column: str = "timestamp",
        available_at_column: str = "available_at",
    ) -> Self:
        """Create a VersionedTimeSeriesDataset from a single DataFrame.

        Convenience constructor for creating a versioned dataset from a single
        DataFrame containing all features. The DataFrame is wrapped in a single
        VersionedTimeSeriesPart before creating the dataset.

        Args:
            data: DataFrame containing versioned time series data with timestamp
                and available_at columns.
            sample_interval: The regular interval between consecutive data points.
            timestamp_column: Name of the column containing timestamps. Default is 'timestamp'.
            available_at_column: Name of the column indicating when data became available.
                Default is 'available_at'.

        Returns:
            New VersionedTimeSeriesDataset instance containing the data.

        Example:
            Create dataset from a single DataFrame:

            >>> import pandas as pd
            >>> from datetime import datetime, timedelta
            >>> data = pd.DataFrame({
            ...     'timestamp': [datetime.fromisoformat('2025-01-01T10:00:00'),
            ...                   datetime.fromisoformat('2025-01-01T10:15:00')],
            ...     'available_at': [datetime.fromisoformat('2025-01-01T10:05:00'),
            ...                      datetime.fromisoformat('2025-01-01T10:20:00')],
            ...     'load': [100.0, 120.0],
            ...     'temperature': [20.0, 22.0]
            ... })
            >>> dataset = VersionedTimeSeriesDataset.from_dataframe(data, timedelta(minutes=15))
            >>> sorted(dataset.feature_names)
            ['load', 'temperature']

        Note:
            This is equivalent to creating a VersionedTimeSeriesPart and then
            wrapping it in a VersionedTimeSeriesDataset, but more convenient
            for simple cases.
        """
        part = VersionedTimeSeriesPart(
            data=data,
            sample_interval=sample_interval,
            timestamp_column=timestamp_column,
            available_at_column=available_at_column,
        )
        return cls(data_parts=[part])

    @override
    def filter_by_range(self, start: datetime | None = None, end: datetime | None = None) -> Self:
        start_idx, end_idx = unsafe_sorted_range_slice_idxs(data=cast(pd.Series, self.index), start=start, end=end)
        index = self.index[start_idx:end_idx]

        return self.__class__(
            data_parts=[part.filter_by_range(start, end) for part in self.data_parts],
            index=index,
        )

    @override
    def filter_by_available_at(self, available_at: AvailableAt) -> Self:
        return self.__class__(
            data_parts=[part.filter_by_available_at(available_at) for part in self.data_parts],
            index=self.index,
        )

    @override
    def filter_by_lead_time(self, lead_time: LeadTime) -> Self:
        return self.__class__(
            data_parts=[part.filter_by_lead_time(lead_time) for part in self.data_parts],
            index=self.index,
        )

    @override
    def select_version(self, available_before: datetime | None = None) -> TimeSeriesDataset:
        selected_parts = [part.select_version(available_before).data for part in self.data_parts]
        combined_data = pd.concat(selected_parts, axis=1).reindex(self.index)
        return TimeSeriesDataset(data=combined_data, sample_interval=self.sample_interval)

    @classmethod
    def concat(cls, datasets: Sequence[Self], mode: ConcatMode) -> Self:
        """Concatenate multiple versioned datasets into a single dataset.

        Combines multiple VersionedTimeSeriesDataset instances using the specified
        concatenation mode. Supports different strategies for handling overlapping
        time indices across datasets.

        This method is useful when you have data from different sources or time
        periods that need to be combined while preserving their versioning
        information. For example, combining weather data from different providers
        or merging historical data with recent updates.

        Args:
            datasets: Sequence of VersionedTimeSeriesDataset instances to concatenate.
                Must contain at least one dataset.
            mode: Concatenation mode determining how to handle overlapping indices:
                - "left": Use indices from the first dataset only
                - "outer": Union of all indices across datasets
                - "inner": Intersection of all indices across datasets

        Returns:
            New VersionedTimeSeriesDataset containing all data parts from input datasets.

        Raises:
            TimeSeriesValidationError: If no datasets are provided for concatenation.
        """
        if not datasets:
            raise TimeSeriesValidationError("At least one dataset must be provided for concatenation.")

        indexes = [d.index for d in datasets]
        match mode:
            case "left":
                index = indexes[0]
            case "outer":
                index = functools.reduce(lambda x, y: cast(pd.DatetimeIndex, x.union(y)), indexes)
            case "inner":
                index = functools.reduce(lambda x, y: x.intersection(y), indexes)

        return cls(
            data_parts=[part for dataset in datasets for part in dataset.data_parts],
            index=index,
        )

    def to_parquet(self, path: FilePath) -> None:
        """Save dataset to parquet file.

        Args:
            path: File path for saving.

        Raises:
            TimeSeriesValidationError: If dataset has multiple data parts.
        """
        if len(self.data_parts) > 1:
            raise TimeSeriesValidationError("to_parquet is only supported for datasets with a single data part.")

        self.data_parts[0].to_parquet(path)

    @classmethod
    def read_parquet(cls, path: FilePath) -> Self:
        """Load dataset from parquet file.

        Args:
            path: Path to parquet file.

        Returns:
            Loaded VersionedTimeSeriesDataset.
        """
        return cls(
            data_parts=[VersionedTimeSeriesPart.read_parquet(path=path)],
        )


__all__ = [
    "VersionedTimeSeriesDataset",
]
