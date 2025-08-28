# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Abstract base classes for time series dataset functionality.

This module provides mixins that define the core interfaces for time series
datasets in OpenSTEF. The mixins separate concerns between basic time series
operations and versioned data access capabilities.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Self

import pandas as pd

from openstef_core.types import AvailableAt, LeadTime

if TYPE_CHECKING:
    from openstef_core.datasets import TimeSeriesDataset


class TimeSeriesMixin(ABC):
    """Abstract base class for time series dataset functionality.

    This mixin defines the essential interface that all time series datasets
    must implement. It provides access to feature metadata, temporal properties,
    and the dataset's temporal index.

    Classes implementing this mixin must set the required attributes in their __init__ method.
    This interface enables consistent access patterns across different time series
    dataset implementations.

    Attributes:
        feature_names: Names of all available features, excluding metadata columns.
        sample_interval: The fixed interval between consecutive samples.
        index: Datetime index representing all timestamps in the dataset.
    """

    feature_names: list[str]
    sample_interval: timedelta
    index: pd.DatetimeIndex

    def calculate_time_coverage(self) -> timedelta:
        """Calculates the total time span covered by the dataset.

        This method computes the total duration represented by the dataset
        based on its unique timestamps and sample interval.

        Returns:
            timedelta: Total time coverage of the dataset.
        """
        return len(self.index.unique()) * self.sample_interval


class VersionedTimeSeriesMixin(TimeSeriesMixin):
    """Abstract base class for versioned time series dataset functionality.

    This mixin defines the interface for datasets that track data availability
    over time. It provides methods to filter datasets by time ranges, availability,
    and lead times, as well as select specific versions of the data.

    The key concept is that data points have both a timestamp (when they
    occurred) and an availability time (when they became available for use).
    This separation allows for accurate simulation of real-world forecasting
    constraints where data arrives with delays or gets revised.

    Classes implementing this mixin should provide:
    - Time range filtering capabilities
    - Availability-based filtering using AvailableAt specifications
    - Lead time-based filtering using LeadTime specifications
    - Version selection for point-in-time data reconstruction
    """

    @abstractmethod
    def filter_by_range(self, start: datetime | None = None, end: datetime | None = None) -> Self:
        """Filter the dataset to include only data within the specified time range.

        Args:
            start: The inclusive start time of the range. If None, no start boundary is applied.
            end: The exclusive end time of the range. If None, no end boundary is applied.

        Returns:
            A new instance of the same type containing only data within [start, end).
        """
        raise NotImplementedError

    @abstractmethod
    def filter_by_available_at(self, available_at: AvailableAt) -> Self:
        """Filter the dataset to include only data available according to the AvailableAt specification.

        This method filters data based on realistic availability constraints,
        considering when data typically becomes available relative to the end of each day.

        Args:
            available_at: Specification defining when data becomes available.

        Returns:
            A new instance of the same type containing only data that would be available
            according to the specified availability pattern.
        """
        raise NotImplementedError

    @abstractmethod
    def filter_by_lead_time(self, lead_time: LeadTime) -> Self:
        """Filter the dataset to include only data available with the specified lead time.

        This method ensures that only data points that were available at least
        `lead_time` before their timestamp are included, simulating real-world
        constraints where predictions must be made with limited recent data.

        Args:
            lead_time: The minimum time gap required between data availability and timestamp.

        Returns:
            A new instance of the same type containing only data available with the required lead time.
        """
        raise NotImplementedError

    @abstractmethod
    def select_version(self, available_before: datetime | None) -> "TimeSeriesDataset":
        """Select a specific version of the data based on availability cutoff.

        This method creates a point-in-time view of the dataset, including only
        data that was available before the specified cutoff time. When multiple
        versions of the same timestamp exist, it selects the latest version
        available before the cutoff.

        Args:
            available_before: Optional cutoff time for data availability.
                Only data available at or before this time is included.
                If None, the most recent version of all data is selected.

        Returns:
            A TimeSeriesDataset containing the selected version of the data,
            with availability metadata removed and timestamp as index.
        """
        raise NotImplementedError


__all__ = ["TimeSeriesMixin", "VersionedTimeSeriesMixin"]
