# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Utilities for combining and transforming time series datasets.

This module provides accessor functions and wrapper classes for common operations
on time series datasets, including feature-wise concatenation and time window
restrictions for analysis scenarios.
"""

import operator
from datetime import datetime, timedelta
from functools import reduce
from typing import Literal, cast

import pandas as pd

from openstef_core.datasets.mixins import TimeSeriesMixin
from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.datasets.validation import check_features_are_disjoint, check_sample_intervals

type ConcatMode = Literal["left", "outer", "inner"]


class ConcatenatedTimeSeries(TimeSeriesMixin):
    """A composite dataset that concatenates features from multiple time series datasets.

    This class combines multiple time series datasets by concatenating their features
    horizontally. It validates that datasets have disjoint feature sets and compatible
    sample intervals before combining them.

    The resulting dataset provides unified access to all features while preserving
    the temporal ordering and metadata of the underlying datasets.

    Args:
        datasets: List of time series datasets to combine. Must have at least 2 datasets.
        mode: How to handle index alignment - 'left', 'outer', or 'inner'.

    Raises:
        ValueError: If fewer than 2 datasets are provided.
        TimeSeriesValidationError: If datasets have overlapping features or
            incompatible sample intervals.
    """

    def __init__(self, datasets: list[TimeSeriesDataset], mode: ConcatMode) -> None:
        """Initialize the concatenated dataset.

        Args:
            datasets: List of time series datasets with disjoint feature sets.
            mode: Index alignment strategy:
                - 'left': Use index from first dataset
                - 'outer': Union of all dataset indices
                - 'inner': Intersection of all dataset indices

        Raises:
            ValueError: If fewer than 2 datasets are provided.
        """
        if len(datasets) < 2:  # noqa: PLR2004
            msg = "At least two datasets are required for concatenation."
            raise ValueError(msg)

        check_features_are_disjoint(datasets)
        check_sample_intervals(datasets)
        self._datasets = datasets
        self._features = reduce(operator.iadd, [d.feature_names for d in datasets], [])
        self._mode = mode

        indexes = [d.index for d in datasets]
        match mode:
            case "left":
                self._index = indexes[0]
            case "outer":
                self._index = reduce(lambda x, y: cast(pd.DatetimeIndex, x.union(y)), indexes)
            case "inner":
                self._index = reduce(lambda x, y: x.intersection(y), indexes)

    @property
    def index(self) -> pd.DatetimeIndex:
        """Combined datetime index based on concatenation mode."""
        return self._index

    @property
    def feature_names(self) -> list[str]:
        """Combined feature names from all datasets."""
        return self._features

    @property
    def sample_interval(self) -> timedelta:
        """Sample interval inherited from the constituent datasets."""
        return self._datasets[0].sample_interval

    @property
    def data(self) -> pd.DataFrame:
        """Combined data from all constituent datasets.

        Returns:
            DataFrame with horizontally concatenated features from all datasets,
            aligned according to the concatenation mode.
        """
        dataframes = [d.data for d in self._datasets]
        if self._mode == "left":
            return pd.concat(dataframes, axis=1, join="outer")
        else:
            return pd.concat(dataframes, axis=1, join=self._mode)


class RestrictedTimeWindowTimeSeries(TimeSeriesMixin):
    """A dataset wrapper that restricts data to a specific time window.

    This class wraps another time series dataset and filters the data to only
    include points within a specified time range. This is useful for analysis
    scenarios where you want to focus on a particular time period or simulate
    data availability constraints.

    The restriction applies to the underlying data access, ensuring that only
    data within the specified time window is available.
    """

    def __init__(self, dataset: TimeSeriesDataset, start_time: datetime, end_time: datetime) -> None:
        """Initialize the time window restricted dataset.

        Args:
            dataset: The underlying time series dataset to wrap.
            start_time: Inclusive start time for the data window.
            end_time: Exclusive end time for the data window.

        Raises:
            ValueError: If start_time is greater than or equal to end_time.
        """
        if start_time >= end_time:
            msg = f"Start time {start_time} must be before end time {end_time}."
            raise ValueError(msg)

        self._dataset = dataset
        self._start_time = start_time
        self._end_time = end_time

        # Filter the data to the specified window
        mask = (dataset.index >= start_time) & (dataset.index < end_time)
        self._filtered_data = dataset.data[mask]

    @property
    def feature_names(self) -> list[str]:
        """Feature names from the underlying dataset."""
        return self._dataset.feature_names

    @property
    def sample_interval(self) -> timedelta:
        """Sample interval from the underlying dataset."""
        return self._dataset.sample_interval

    @property
    def index(self) -> pd.DatetimeIndex:
        """Filtered datetime index within the specified time window."""
        return cast(pd.DatetimeIndex, self._filtered_data.index)

    @property
    def data(self) -> pd.DataFrame:
        """Filtered data within the specified time window."""
        return self._filtered_data

    @property
    def start_time(self) -> datetime:
        """The start time of the restriction window."""
        return self._start_time

    @property
    def end_time(self) -> datetime:
        """The end time of the restriction window."""
        return self._end_time


class TimeSeriesAccessors:
    """Utility class providing static methods for time series dataset operations.

    This class contains factory methods for creating composite datasets
    and applying transformations to time series data without versioning
    information.
    """

    @staticmethod
    def concat_featurewise(datasets: list[TimeSeriesDataset], mode: ConcatMode = "outer") -> TimeSeriesDataset:
        """Concatenate multiple datasets by combining their features.

        Combines datasets horizontally by concatenating their feature columns.
        Validates that datasets have disjoint feature sets and compatible
        sample intervals before creating the composite dataset.

        Args:
            datasets: List of time series datasets to combine. Must have
                disjoint feature sets and identical sample intervals.
            mode: Index alignment strategy for concatenation:
                - 'left': Use index from first dataset
                - 'outer': Union of all dataset indices
                - 'inner': Intersection of all dataset indices

        Returns:
            New TimeSeriesDataset with unified access to all features.
            If only one dataset is provided, returns it unchanged.

        Example:
            Combine temperature and load datasets:

            >>> import pandas as pd
            >>> from datetime import timedelta
            >>> from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
            >>> from openstef_core.datasets.timeseries_accessors import TimeSeriesAccessors
            >>> # Create two datasets with different features
            >>> temp_data = pd.DataFrame({
            ...     'temperature': [20.0, 21.0, 22.0]
            ... }, index=pd.date_range('2025-01-01T10:00:00', periods=3, freq='15min'))
            >>> load_data = pd.DataFrame({
            ...     'load': [100.0, 110.0, 120.0]
            ... }, index=pd.date_range('2025-01-01T10:00:00', periods=3, freq='15min'))
            >>> temp_dataset = TimeSeriesDataset(temp_data, timedelta(minutes=15))
            >>> load_dataset = TimeSeriesDataset(load_data, timedelta(minutes=15))
            >>> # Concatenate the datasets
            >>> combined = TimeSeriesAccessors.concat_featurewise([temp_dataset, load_dataset])
            >>> len(combined.feature_names) == 2
            True
            >>> 'temperature' in combined.feature_names and 'load' in combined.feature_names
            True
        """
        if len(datasets) == 1:
            return datasets[0]

        concatenated = ConcatenatedTimeSeries(datasets=datasets, mode=mode)

        # Create a new TimeSeriesDataset with the concatenated data
        return TimeSeriesDataset(data=concatenated.data, sample_interval=concatenated.sample_interval)
