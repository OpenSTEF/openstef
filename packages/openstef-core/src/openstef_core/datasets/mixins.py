# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Protocol definitions for time series dataset interfaces.

This module provides protocol classes that define the core interfaces for time series
datasets in OpenSTEF. Protocols enable type checking and documentation of expected
behavior without requiring inheritance.

Key protocols:
    - TimeSeries: Core interface for all time series datasets with filtering and versioning
    - DatasetMixin: Interface for dataset persistence operations
"""

from abc import abstractmethod
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Concatenate, Protocol, Self

from pydantic import FilePath

from openstef_core.types import AvailableAt, LeadTime

if TYPE_CHECKING:
    import pandas as pd

    from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset


class TimeSeriesMixin(Protocol):
    """Interface defining the interface for time series datasets.

    This interface defines the essential operations that all time series datasets
    must implement. It provides access to feature metadata, temporal properties,
    the dataset's temporal index, and filtering/versioning capabilities.

    Classes implementing this interface must provide:
        - Access to the datetime index
        - Sample interval information
        - Feature names list
        - Versioning status indicator
        - Filtering methods for time ranges, availability, and lead times
        - Version selection for point-in-time data reconstruction
    """

    @property
    @abstractmethod
    def index(self) -> "pd.DatetimeIndex":
        """Get the datetime index of the dataset.

        Returns:
            DatetimeIndex representing all timestamps in the dataset.
        """

    @property
    @abstractmethod
    def sample_interval(self) -> timedelta:
        """Get the fixed time interval between consecutive data points.

        Returns:
            The sample interval as a timedelta.
        """

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Get the names of all available features in the dataset.

        Returns:
            List of feature names, excluding metadata columns like timestamp,
            available_at, or horizon.
        """

    @property
    @abstractmethod
    def is_versioned(self) -> bool:
        """Check if the dataset tracks data availability over time.

        Returns:
            True if the dataset is versioned (tracks availability via horizon
            or available_at columns), False for regular time series.
        """

    @abstractmethod
    def filter_by_range(self, start: datetime | None = None, end: datetime | None = None) -> Self:
        """Filter the dataset to include only data within the specified time range.

        Args:
            start: Inclusive start time of the range. If None, no start boundary applied.
            end: Exclusive end time of the range. If None, no end boundary applied.

        Returns:
            New instance containing only data within [start, end).
        """

    @abstractmethod
    def filter_by_available_before(self, available_before: datetime) -> Self:
        """Filter to include only data available before the specified timestamp.

        Args:
            available_before: Cutoff time for data availability.

        Returns:
            New instance containing only data available before the cutoff.
        """

    @abstractmethod
    def filter_by_available_at(self, available_at: AvailableAt) -> Self:
        """Filter based on realistic daily data availability constraints.

        Args:
            available_at: Specification defining when data becomes available.

        Returns:
            New instance with data filtered by availability pattern.
        """

    @abstractmethod
    def filter_by_lead_time(self, lead_time: LeadTime) -> Self:
        """Filter to include only data available at or longer than the specified lead time.

        Args:
            lead_time: Minimum time gap required between data availability and timestamp.

        Returns:
            New instance containing only data available with the required lead time.
        """

    @abstractmethod
    def select_version(self) -> "TimeSeriesDataset":
        """Select a specific version of the dataset based on data availability.

        Creates a point-in-time snapshot by selecting the latest available version
        for each timestamp. Essential for preventing lookahead bias in backtesting.

        Returns:
            TimeSeriesDataset containing the selected version of the data.
        """

    def calculate_time_coverage(self) -> timedelta:
        """Calculates the total time span covered by the dataset.

        This method computes the total duration represented by the dataset
        based on its unique timestamps and sample interval.

        Returns:
            timedelta: Total time coverage of the dataset.
        """
        return len(self.index.unique()) * self.sample_interval


class DatasetMixin(Protocol):
    """Abstract base class for dataset persistence operations.

    This mixin defines the interface for saving and loading datasets to/from
    parquet files. It ensures datasets can be persisted with all their metadata
    and reconstructed exactly as they were saved.

    Classes implementing this mixin must:
    - Save all data and metadata necessary for complete reconstruction
    - Store metadata in parquet file attributes using attrs
    - Handle missing metadata gracefully with sensible defaults when loading

    See Also:
        TimeSeriesDataset: Implementation for standard time series datasets.
        VersionedTimeSeriesDataset: Implementation for versioned dataset segments.
    """

    @abstractmethod
    def to_parquet(self, path: FilePath) -> None:
        """Save the dataset to a parquet file.

        Stores both the dataset's data and all necessary metadata for complete
        reconstruction. Metadata should be stored in the parquet file's attrs
        dictionary.

        Args:
            path: File path where the dataset should be saved.

        See Also:
            read_parquet: Counterpart method for loading datasets.
        """

    @classmethod
    @abstractmethod
    def read_parquet(cls, path: FilePath) -> Self:
        """Load a dataset from a parquet file.

        Reconstructs a dataset from a parquet file created with to_parquet,
        including all data and metadata. Should handle missing metadata
        gracefully with sensible defaults.

        Args:
            path: Path to the parquet file to load.

        Returns:
            New dataset instance reconstructed from the file.

        See Also:
            to_parquet: Counterpart method for saving datasets.
        """

    def pipe[T, **P](self, func: Callable[Concatenate[Self, P], T], *args: P.args, **kwargs: P.kwargs) -> T:
        """Applies a function to the dataset and returns the result.

        This method allows for functional-style transformations and operations
        on the dataset, enabling method chaining and cleaner code.

        Args:
            func: A callable that takes the dataset instance and returns a value of type T.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The result of applying the function to the dataset.
        """
        return func(self, *args, **kwargs)
