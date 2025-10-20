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
from datetime import timedelta
from typing import ClassVar, Self, cast, override

import pandas as pd

from openstef_core.datasets.mixins import DatasetMixin, VersionedTimeSeriesMixin
from openstef_core.types import LeadTime
from openstef_core.utils import timedelta_from_isoformat, timedelta_to_isoformat

_logger = logging.getLogger(__name__)


class TimeSeriesDataset(VersionedTimeSeriesMixin, DatasetMixin):
    """A time series dataset with regular sampling intervals.

    This class represents time series data with a consistent sampling interval
    and provides basic operations for data access and persistence. It ensures
    that the data maintains temporal ordering and provides access to temporal
    and feature metadata.

    The dataset guarantees:
        - Data is sorted by timestamp in ascending order
        - Consistent sampling interval across all data points
        - DateTime index for temporal operations

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
        >>> dataset.sample_interval
        datetime.timedelta(seconds=900)
    """

    horizon_column: ClassVar[str] = "horizon"

    data: pd.DataFrame
    _sample_interval: timedelta
    _feature_names: list[str]

    horizons: list[LeadTime] | None

    def __init__(
        self,
        data: pd.DataFrame,
        sample_interval: timedelta = timedelta(minutes=15),
    ) -> None:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("Data index must be a pandas DatetimeIndex.")

        if not data.attrs.get("is_sorted", False):
            self.data = data.sort_index(ascending=True)

        data.attrs["is_sorted"] = True

        self.data = data
        self._sample_interval = sample_interval

        if self.horizon_column in data.columns:
            if not pd.api.types.is_timedelta64_dtype(data[self.horizon_column]):
                msg = f"Horizon column '{self.horizon_column}' must be of type timedelta."
                raise TypeError(msg)

            self._feature_names = [col for col in data.columns if col != self.horizon_column]
            self.horizons = list({LeadTime(value=td) for td in data[self.horizon_column].unique()})
        else:
            self._feature_names = data.columns.tolist()
            self.horizons = None

    @property
    @override
    def index(self) -> pd.DatetimeIndex:
        return cast(pd.DatetimeIndex, self.data.index)

    @property
    @override
    def feature_names(self) -> list[str]:
        return self._feature_names

    @property
    @override
    def sample_interval(self) -> timedelta:
        return self._sample_interval

    @override
    def _to_versioned_dataframe(self) -> pd.DataFrame:
        if self.horizons is None:
            raise ValueError("Non horizon dataset cannot be converted to versioned dataframe.")

        data_versioned = self.data.reset_index(names=[self.timestamp_column])
        data_versioned[self.available_at_column] = (
            data_versioned[self.timestamp_column] - data_versioned[self.horizon_column]
        ).drop(columns=[self.horizon_column])
        return data_versioned

    @override
    def _from_versioned_dataframe(self, data: pd.DataFrame) -> Self:
        horizon = data[self.timestamp_column] - data[self.available_at_column]
        data = (
            data.drop(columns=[self.available_at_column])
            .assign(**{self.horizon_column: horizon})
            .set_index(self.timestamp_column)
        )
        return self.__class__(
            data=data,
            sample_interval=self.sample_interval,
        )

    @override
    def _to_pandas(self) -> pd.DataFrame:
        self.data.attrs["sample_interval"] = timedelta_to_isoformat(self.sample_interval)
        self.data.attrs["is_sorted"] = True
        return self.data

    @override
    @classmethod
    def _from_pandas(cls, df: pd.DataFrame) -> Self:
        if "sample_interval" not in df.attrs:
            _logger.warning(
                "Parquet file does not contain 'sample_interval' attribute. Using default value of 15 minutes."
            )

        sample_interval = timedelta_from_isoformat(df.attrs.get("sample_interval", "PT15M"))

        return cls(
            data=df,
            sample_interval=sample_interval,
        )

    @property
    def has_horizons(self) -> bool:
        """Indicates whether the dataset includes forecast horizons.

        Returns:
            bool: True if the dataset has a horizon column, False otherwise.
        """
        return self.horizons is not None

    @property
    def horizon_series(self) -> "pd.Series[pd.Timedelta]":
        """Get the horizon values as a pandas Series.

        Returns:
            pd.Series: Series containing horizon values indexed by timestamp.

        Raises:
            ValueError: If the dataset does not contain horizon information.
        """
        if not self.has_horizons:
            raise ValueError("Dataset does not contain horizon information.")

        return self.data[self.horizon_column]

    def select_horizon(self, horizon: LeadTime) -> Self:
        """Select data for a specific forecast horizon.

        Args:
            horizon: The forecast horizon to filter the dataset by.

        Returns:
            A new TimeSeriesDataset instance containing only data for the specified horizon.

        Raises:
            ValueError: If the specified horizon is not present in the dataset.
        """
        if self.horizons is None or horizon not in self.horizons:
            msg = f"Horizon {horizon} not found in dataset."
            raise ValueError(msg)

        filtered_data = self.data[self.data[self.horizon_column] == horizon.value]
        return self.__class__(
            data=filtered_data,
            sample_interval=self.sample_interval,
        )


__all__ = ["TimeSeriesDataset"]
