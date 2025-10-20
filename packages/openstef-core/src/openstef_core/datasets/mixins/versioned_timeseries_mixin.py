# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Abstract base classes for time series dataset functionality.

This module provides mixins that define the core interfaces for time series
datasets in OpenSTEF. The mixins separate concerns between basic time series
operations and versioned data access capabilities.
"""

from abc import abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar, Self

import pandas as pd

from openstef_core.datasets.mixins.timeseries_mixin import TimeSeriesMixin
from openstef_core.types import AvailableAt, LeadTime
from openstef_core.utils.pandas import unsafe_sorted_range_slice_idxs

if TYPE_CHECKING:
    from openstef_core.datasets import TimeSeriesDataset


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

    timestamp_column: ClassVar[str] = "timestamp"
    available_at_column: ClassVar[str] = "available_at"

    @abstractmethod
    def _to_versioned_dataframe(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def _from_versioned_dataframe(self, data: pd.DataFrame) -> Self:
        pass

    def filter_by_range(
        self, start: datetime | None = None, end: datetime | None = None, available_before: datetime | None = None
    ) -> Self:
        """Filter the dataset to include only data within the specified time range.

        Args:
            start: The inclusive start time of the range. If None, no start boundary is applied.
            end: The exclusive end time of the range. If None, no end boundary is applied.
            available_before: Optional cutoff time for data availability. If provided, only data
                available before this time will be included.

        Returns:
            A new instance of the same type containing only data within [start, end).
        """
        data = self._to_versioned_dataframe()
        start_idx, end_idx = unsafe_sorted_range_slice_idxs(data=data[self.timestamp_column], start=start, end=end)
        data_filtered = data.iloc[start_idx:end_idx]
        if available_before is not None:
            data_filtered = data_filtered[data_filtered[self.available_at_column] <= available_before]

        return self._from_versioned_dataframe(data_filtered)

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
        data = self._to_versioned_dataframe()
        cutoff = data[self.timestamp_column].dt.floor("D") - pd.Timedelta(available_at.lag_from_day)
        data_filtered = data[data[self.available_at_column] <= cutoff]
        return self._from_versioned_dataframe(data_filtered)

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
        data = self._to_versioned_dataframe()
        cutoff = data[self.timestamp_column] - pd.Timedelta(lead_time.value)
        data_filtered = data[data[self.available_at_column] <= cutoff]
        return self._from_versioned_dataframe(data_filtered)

    def select_version(self) -> "TimeSeriesDataset":
        """Select a specific version of the dataset based on data availability.

        Creates a point-in-time snapshot of the dataset containing only data that
        was available before the specified timestamp. This enables realistic
        backtesting by reconstructing what the dataset would have looked like
        at a specific point in time.

        This method is essential for preventing lookahead bias in backtesting.
        Without it, models would appear to have access to future data that
        wasn't available when the forecast was made.

        Returns:
            TimeSeriesDataset containing only data available before the cutoff,
            with the latest available version for each timestamp.
        """
        from openstef_core.datasets import TimeSeriesDataset  # noqa: PLC0415

        data = self._to_versioned_dataframe()
        data_selected = (
            data.drop_duplicates(subset=self.timestamp_column, keep="last")
            .drop(columns=[self.available_at_column])
            .set_index(self.timestamp_column)
        )

        return TimeSeriesDataset(
            data=data_selected,
            sample_interval=self.sample_interval,
        )


__all__ = ["VersionedTimeSeriesMixin"]
