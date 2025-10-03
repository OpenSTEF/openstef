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
from collections import UserDict
from collections.abc import Callable, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Self, cast

import pandas as pd

from openstef_core.datasets.mixins import TimeSeriesMixin
from openstef_core.types import LeadTime
from openstef_core.utils import timedelta_from_isoformat, timedelta_to_isoformat

_logger = logging.getLogger(__name__)


class TimeSeriesDataset(TimeSeriesMixin):
    """A time series dataset with regular sampling intervals.

    This class represents time series data with a consistent sampling interval
    and provides basic operations for data access and persistence. It ensures
    that the data maintains temporal ordering and provides access to temporal
    and feature metadata.

    The dataset guarantees:
        - Data is sorted by timestamp in ascending order
        - Consistent sampling interval across all data points
        - DateTime index for temporal operations

    Attributes:
        data: DataFrame with DatetimeIndex containing time series data.
        sample_interval: Fixed time interval between consecutive data points.
        index: Datetime index representing all timestamps in the dataset.
        feature_names: Names of all available features, excluding metadata columns.

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

    data: pd.DataFrame
    sample_interval: timedelta
    index: pd.DatetimeIndex
    feature_names: list[str]

    def __init__(
        self,
        data: pd.DataFrame,
        sample_interval: timedelta,
        *,
        is_sorted: bool = False,
    ) -> None:
        """Initialize a TimeSeriesDataset with the given data and sampling interval.

        Args:
            data: DataFrame with DatetimeIndex containing time series data.
                Must have a pandas DatetimeIndex.
            sample_interval: Fixed time interval between consecutive data points.
                Must be positive.
            is_sorted: If True, assumes data is already sorted by timestamp.
                If False, data will be sorted during initialization

        Raises:
            TypeError: If data index is not a pandas DatetimeIndex.

        Note:
            Data is automatically sorted by timestamp if not already sorted.
            The 'is_sorted' attribute is set to track sorting state.
        """
        super().__init__()

        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("Data index must be a pandas DatetimeIndex.")

        if not data.attrs.get("is_sorted", False) and not is_sorted:
            data = data.sort_index(ascending=True)

        data.attrs["is_sorted"] = True
        self.data = data
        self.sample_interval = sample_interval
        self.index = cast(pd.DatetimeIndex, self.data.index)
        self.feature_names = self.data.columns.tolist()

    def to_parquet(
        self,
        path: Path,
    ) -> None:
        """Save the dataset to a parquet file.

        Stores both the time series data and metadata (sample interval, sort status, ...)
        in the parquet file for complete reconstruction.

        Args:
            path: File path where the dataset should be saved.

        Note:
            The sample interval is stored as an ISO 8601 duration string
            in the file's metadata attributes.
        """
        self.data.attrs["sample_interval"] = timedelta_to_isoformat(self.sample_interval)
        self.data.attrs["is_sorted"] = True
        self.data.to_parquet(path)

    @classmethod
    def read_parquet(
        cls,
        path: Path,
    ) -> Self:
        """Create a TimeSeriesDataset from a parquet file.

        Loads time series data and metadata from a parquet file created with
        the `to_parquet` method. Handles missing metadata gracefully with
        reasonable defaults.

        Args:
            path: Path to the parquet file to load.

        Returns:
            New TimeSeriesDataset instance with data and metadata from the file.

        Example:
            Save and reload a dataset:

            >>> from pathlib import Path
            >>> import tempfile
            >>> import pandas as pd
            >>> from datetime import timedelta
            >>> # Create a simple dataset
            >>> data = pd.DataFrame({
            ...     'temperature': [20.1, 22.3],
            ...     'load': [100, 120]
            ... }, index=pd.date_range('2025-01-01', periods=2, freq='15min'))
            >>> dataset = TimeSeriesDataset(data, sample_interval=timedelta(minutes=15))
            >>> # Test save/load cycle
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     file_path = Path(tmpdir) / "data.parquet"
            ...     dataset.to_parquet(file_path)
            ...     loaded = TimeSeriesDataset.read_parquet(file_path)
            ...     loaded.feature_names == dataset.feature_names
            True

        Note:
            If the parquet file lacks a sample_interval attribute, defaults
            to 15 minutes with a warning logged.
        """
        data = pd.read_parquet(path)
        if "sample_interval" not in data.attrs:
            _logger.warning(
                "Parquet file does not contain 'sample_interval' attribute. Using default value of 15 minutes."
            )

        sample_interval = timedelta_from_isoformat(data.attrs.get("sample_interval", "PT15M"))

        return cls(
            data=data,
            sample_interval=sample_interval,
        )


class MultiHorizon[T: TimeSeriesDataset](UserDict[LeadTime, T]):
    """A dictionary mapping forecast horizons to time series datasets.

    This class represents a collection of time series datasets, each associated
    with a specific forecast horizon. It extends the standard dictionary to
    provide additional type safety and clarity when working with multiple
    horizon-specific datasets.

    Attributes:
        keys: Forecast horizons (LeadTime) for which datasets are available.
        values: Corresponding time series datasets for each horizon.

    Example:
        Create a MultiHorizon dataset with two horizons:

        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.types import LeadTime
        >>> from openstef_core.datasets import TimeSeriesDataset, MultiHorizon
        >>> # Create sample datasets for two horizons
        >>> data_1h = pd.DataFrame({
        ...     'load': [100, 110],
        ... }, index=pd.date_range('2025-01-01', periods=2, freq='1H'))
        >>> dataset_1h = TimeSeriesDataset(data_1h, sample_interval=timedelta(hours=1))
        >>> data_24h = pd.DataFrame({
        ...     'load': [120, 130],
        ... }, index=pd.date_range('2025-01-01', periods=2, freq='24H'))
        >>> dataset_24h = TimeSeriesDataset(data_24h, sample_interval=timedelta(hours=24))
        >>> # Create MultiHorizon mapping
        >>> horizons = MultiHorizon({
        ...     LeadTime.from_string("PT1H"): dataset_1h,
        ...     LeadTime.from_string("PT24H"): dataset_24h,
        ... })
        >>> len(horizons)
        2
        >>> list(horizons.keys())
        [LeadTime('PT1H'), LeadTime('PT24H')]
    """

    def horizons(self) -> Sequence[LeadTime]:
        """Get the list of forecast horizons available in the MultiHorizon dataset.

        Returns:
            List of LeadTime objects representing the forecast horizons.
        """
        return list(self.keys())

    def map_horizons[U: TimeSeriesDataset](self, func: Callable[[T], U]) -> "MultiHorizon[U]":
        """Apply a function to each dataset in the MultiHorizon collection.

        Args:
            func: A callable that takes a TimeSeriesDataset and returns a new dataset.

        Returns:
            A new MultiHorizon instance with the transformed datasets.
        """
        return MultiHorizon({k: func(v) for k, v in self.items()})



__all__ = ["MultiHorizon", "TimeSeriesDataset"]
