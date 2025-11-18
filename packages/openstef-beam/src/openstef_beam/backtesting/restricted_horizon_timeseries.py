# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Compatibility layer for versioned time series accessors.

This module provides placeholders for accessor classes that were used
by the old implementation. These will redirect to the new V4 implementation.
"""

from datetime import datetime

from openstef_core.datasets import TimeSeriesDataset, VersionedTimeSeriesDataset


class RestrictedHorizonVersionedTimeSeries:
    """Compatibility wrapper for horizon-restricted access.

    This is a placeholder that maintains the interface of the old implementation
    while using the new V4 classes underneath.
    """

    def __init__(self, dataset: VersionedTimeSeriesDataset, horizon: datetime) -> None:
        """Initialize with dataset and horizon.

        Args:
            dataset: The underlying dataset
            horizon: The horizon restriction time
        """
        self.dataset = dataset
        self.horizon = horizon

    @property
    def feature_names(self) -> list[str]:
        """Get feature names from underlying dataset."""
        return self.dataset.feature_names

    def get_window(self, start: datetime, end: datetime, available_before: datetime | None = None) -> TimeSeriesDataset:
        """Get data window with horizon restriction.

        Returns:
            DataFrame with data from the specified window.
        """
        dataset = self.get_window_versioned(start=start, end=end, available_before=available_before)

        return dataset.select_version()

    def get_window_versioned(
        self, start: datetime, end: datetime, available_before: datetime | None = None
    ) -> VersionedTimeSeriesDataset:
        """Get data window with horizon restriction.

        Returns:
            DataFrame with data from the specified window.

        Raises:
            ValueError: If available_before is after the horizon.
        """
        if available_before is None:
            available_before = self.horizon

        if available_before > self.horizon:
            raise ValueError("available_before cannot be after the horizon")

        dataset = self.dataset.filter_by_range(start=start, end=end)
        # Make sure to only include data available before the cutoff
        return dataset.filter_by_available_before(available_before=available_before)


__all__ = [
    "RestrictedHorizonVersionedTimeSeries",
]
