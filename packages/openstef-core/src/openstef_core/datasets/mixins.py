# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import pandas as pd


class TimeSeriesMixin(ABC):
    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Returns the list of feature names available in the dataset.

        Returns:
            list[str]: Names of all available features, excluding metadata columns.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def sample_interval(self) -> timedelta:
        """Returns the time interval between consecutive data points.

        This property defines the temporal resolution of the dataset and
        should remain constant throughout the dataset.

        Returns:
            timedelta: The fixed interval between consecutive samples.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def index(self) -> pd.DatetimeIndex:
        """Returns the temporal index of the dataset.

        Returns:
            pd.DatetimeIndex: Datetime index representing all timestamps in the dataset.
        """
        raise NotImplementedError


class VersionedAccessMixin(ABC):
    @abstractmethod
    def get_window(self, start: datetime, end: datetime, available_before: datetime | None = None) -> pd.DataFrame:
        """Get a window of data from the dataset that is available before or at the specified time.

        Implementers must ensure that:

        1. Only data within [start, end) is returned
        2. If available_before is specified, only data available at or before that time is included
        3. The returned data maintains the dataset's sample interval
        4. When multiple versions of a timestamp exist, selects the latest available version

        Args:
            start: The inclusive start time of the window.
            end: The exclusive end time of the window.
            available_before: Optional timestamp that filters data to only include
                points available at or before this time. If None, all data is included.

        Returns:
            A DataFrame containing the data in the specified time window. Timestamps with unavailable data are filled
            with NaN.
        """
        raise NotImplementedError


class VersionedTimeSeriesMixin(TimeSeriesMixin, VersionedAccessMixin, ABC):
    """A mixin that combines time series and versioned access functionality.

    This mixin provides the necessary properties and methods to handle time series data
    with versioning capabilities, allowing access to historical data points based on their availability.
    """


__all__ = ["TimeSeriesMixin", "VersionedAccessMixin", "VersionedTimeSeriesMixin"]
