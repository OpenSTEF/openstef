"""Versioned time series dataset implementation with improved architecture.

This module provides an enhanced implementation of versioned time series datasets
that track data availability over time. The new architecture supports composable
datasets from multiple parts and improved filtering capabilities for realistic
backtesting and forecasting scenarios.

The key improvement over the previous implementation is the ability to combine
multiple data sources while maintaining versioning information, enabling more
flexible dataset construction for complex forecasting pipelines.
"""

from enum import StrEnum
import functools
import logging
import operator
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Self, cast, override

from openstef_core.datasets.validation import validate_disjoint_columns, validate_same_sample_intervals
from openstef_core.datasets.versioned_timeseries.dataset_part import VersionedTimeSeriesPart
from openstef_core.utils.pandas import sorted_range_slice_idxs
import pandas as pd

from openstef_core.datasets.mixins import VersionedTimeSeriesMixin
from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.exceptions import InvalidColumnTypeError, TimeSeriesValidationError
from openstef_core.types import AvailableAt, LeadTime

_logger = logging.getLogger(__name__)


class ConcatMode(StrEnum):
    LEFT = "left"
    OUTER = "outer"
    INNER = "inner"


class VersionedTimeSeriesDataset(VersionedTimeSeriesMixin):
    """A versioned time series dataset composed of multiple data parts.

    This class combines multiple VersionedTimeSeriesPart instances into a unified
    dataset that tracks data availability over time. It provides methods to filter
    datasets by time ranges, availability constraints, and lead times, as well as
    select specific versions of the data for point-in-time reconstruction.

    The dataset is particularly useful for realistic backtesting scenarios where
    data arrives with delays or gets revised over time.

    Attributes:
        data_parts: List of VersionedTimeSeriesPart instances that compose this dataset.
        index: Datetime index representing all timestamps across all data parts.
        sample_interval: Fixed time interval between consecutive data points.
        feature_names: Names of all available features across all data parts.
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
        self.index = index or functools.reduce(operator.or_, [part.index for part in data_parts], pd.DatetimeIndex([]))
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
        start_idx, end_idx = sorted_range_slice_idxs(
            data=cast(pd.Series[pd.Timestamp], self.index), start=start, end=end
        )
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
    def select_version(self, available_before: datetime | None) -> TimeSeriesDataset:
        selected_parts = [part.select_version(available_before).data for part in self.data_parts]
        combined_data = pd.concat(selected_parts, axis=1).loc[self.index]
        return TimeSeriesDataset(data=combined_data, sample_interval=self.sample_interval)
    
    @classmethod
    def concat(cls, datasets: Sequence[Self], mode: ConcatMode) -> Self:
        if not datasets:
            raise TimeSeriesValidationError("At least one dataset must be provided for concatenation.")



__all__ = ["VersionedTimeSeriesDataset", ]
