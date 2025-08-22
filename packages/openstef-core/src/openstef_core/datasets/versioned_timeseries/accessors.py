# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Utilities for combining and transforming versioned time series datasets.

This module provides accessor functions and wrapper classes for common operations
on versioned time series datasets, including feature-wise concatenation and
horizon restrictions for backtesting scenarios.
"""

import operator
from datetime import datetime, timedelta
from functools import reduce
from typing import Literal, cast

import pandas as pd

from openstef_core.datasets.mixins import VersionedTimeSeriesMixin
from openstef_core.datasets.validation import check_features_are_disjoint, check_sample_intervals

type ConcatMode = Literal["left", "outer", "inner"]


class ConcatenatedVersionedTimeSeries(VersionedTimeSeriesMixin):
    """A composite dataset that concatenates features from multiple versioned datasets.

    This class combines multiple versioned time series datasets by concatenating
    their features horizontally. It validates that datasets have disjoint feature
    sets and compatible sample intervals before combining them.

    The resulting dataset provides unified access to all features while preserving
    the versioned data access semantics of the underlying datasets.

    Args:
        datasets: List of versioned datasets to combine. Must have at least 2 datasets.
        mode: How to handle index alignment - 'left', 'outer', or 'inner'.

    Raises:
        ValueError: If fewer than 2 datasets are provided.
        TimeSeriesValidationError: If datasets have overlapping features or
            incompatible sample intervals.
    """

    def __init__(self, datasets: list[VersionedTimeSeriesMixin], mode: ConcatMode) -> None:
        """Initialize the concatenated dataset.

        Args:
            datasets: List of versioned datasets with disjoint feature sets.
            mode: Index alignment strategy:
                - 'left': Use index from first dataset
                - 'outer': Union of all dataset indices
                - 'inner': Intersection of all dataset indices

        Raises:
            ValueError: If fewer than 2 datasets are provided.
        """
        if len(datasets) < 2:  # noqa: PLR2004
            msg = "At least two datasets are required for ConcatFeaturewise."
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

    def get_window(self, start: datetime, end: datetime, available_before: datetime | None = None) -> pd.DataFrame:
        """Get concatenated data window from all datasets.

        Returns:
            DataFrame with horizontally concatenated features from all constituent datasets.
        """
        dataframes = [d.get_window(start=start, end=end, available_before=available_before) for d in self._datasets]
        return pd.concat(dataframes, axis=1)


class RestrictedHorizonVersionedTimeSeries(VersionedTimeSeriesMixin):
    """A dataset wrapper that restricts data availability to a specific horizon.

    This class wraps another versioned dataset and enforces a maximum horizon
    for data availability. It ensures that no data is returned that was available
    after the specified horizon time, which is useful for backtesting scenarios
    where you want to simulate real-time constraints.

    The restriction applies to the `available_before` parameter in `get_window`
    calls, ensuring it never exceeds the configured horizon.
    """

    def __init__(self, dataset: VersionedTimeSeriesMixin, horizon: datetime) -> None:
        """Initialize the horizon-restricted dataset.

        Args:
            dataset: The underlying versioned dataset to wrap.
            horizon: Maximum time for data availability cutoff.
        """
        self._dataset = dataset
        self._horizon = horizon

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
        """Datetime index from the underlying dataset."""
        return self._dataset.index

    @property
    def horizon(self) -> datetime:
        """The horizon cutoff time for data availability."""
        return self._horizon

    def get_window(self, start: datetime, end: datetime, available_before: datetime | None = None) -> pd.DataFrame:
        """Get data window with horizon restriction enforced.

        Args:
            start: Inclusive start time of the window.
            end: Exclusive end time of the window.
            available_before: Data availability cutoff. If specified and greater
                than the horizon, a ValueError is raised.

        Returns:
            DataFrame with data available before the effective cutoff time.

        Raises:
            ValueError: If available_before exceeds the configured horizon.
        """
        if available_before is not None and available_before > self._horizon:
            msg: str = f"Available before {available_before} is greater than the horizon."
            raise ValueError(msg)

        return self._dataset.get_window(start=start, end=end, available_before=available_before or self._horizon)


def concat_featurewise(datasets: list[VersionedTimeSeriesMixin], mode: ConcatMode) -> VersionedTimeSeriesMixin:
    """Concatenate multiple datasets by combining their features.

    Combines datasets horizontally by concatenating their feature columns.
    Validates that datasets have disjoint feature sets and compatible
    sample intervals before creating the composite dataset.

    Args:
        datasets: List of versioned datasets to combine. Must have
            disjoint feature sets and identical sample intervals.
        mode: Index alignment strategy for concatenation:
            - 'left': Use index from first dataset
            - 'outer': Union of all dataset indices
            - 'inner': Intersection of all dataset indices

    Returns:
        Composite dataset providing unified access to all features.
        If only one dataset is provided, returns it unchanged.

    Example:
        Combine temperature and load datasets:

        >>> from datetime import datetime, timedelta
        >>> import pandas as pd
        >>> from openstef_core.datasets.versioned_timeseries import VersionedTimeSeriesDataset, concat_featurewise
        >>> # Create two simple test datasets
        >>> temp_data = pd.DataFrame({
        ...     'timestamp': [datetime.fromisoformat('2025-01-01T10:00:00')],
        ...     'available_at': [datetime.fromisoformat('2025-01-01T10:05:00')],
        ...     'temperature': [20.0]
        ... })
        >>> load_data = pd.DataFrame({
        ...     'timestamp': [datetime.fromisoformat('2025-01-01T10:00:00')],
        ...     'available_at': [datetime.fromisoformat('2025-01-01T10:05:00')],
        ...     'load': [100.0]
        ... })
        >>> temp_dataset = VersionedTimeSeriesDataset(temp_data, timedelta(minutes=15))
        >>> load_dataset = VersionedTimeSeriesDataset(load_data, timedelta(minutes=15))
        >>> combined = concat_featurewise(
        ...     [temp_dataset, load_dataset], mode='outer'
        ... )
        >>> sorted(combined.feature_names)
        ['load', 'temperature']
    """
    if len(datasets) == 1:
        return datasets[0]

    return ConcatenatedVersionedTimeSeries(datasets=datasets, mode=mode)


def restrict_horizon(dataset: VersionedTimeSeriesMixin, horizon: datetime) -> "RestrictedHorizonVersionedTimeSeries":
    """Restrict dataset access to data available before a specific horizon.

    Creates a wrapper that enforces a maximum horizon for data availability,
    useful for backtesting scenarios where you need to simulate real-time
    data constraints.

    Args:
        dataset: The dataset to apply the horizon restriction to.
        horizon: Maximum time for data availability cutoff.

    Returns:
        Wrapped dataset with horizon restriction enforced.

    Example:
        Restrict dataset for backtesting:

        >>> from datetime import datetime, timedelta
        >>> import pandas as pd
        >>> from openstef_core.datasets.versioned_timeseries import VersionedTimeSeriesDataset, restrict_horizon
        >>> # Create test dataset
        >>> data = pd.DataFrame({
        ...     'timestamp': [datetime.fromisoformat('2025-01-01T10:00:00')],
        ...     'available_at': [datetime.fromisoformat('2025-01-01T12:00:00')],
        ...     'load': [100.0]
        ... })
        >>> dataset = VersionedTimeSeriesDataset(data, timedelta(minutes=15))
        >>> backtest_cutoff = datetime.fromisoformat('2025-01-01T11:00:00')
        >>> restricted = restrict_horizon(
        ...     dataset, backtest_cutoff
        ... )
        >>> isinstance(restricted, RestrictedHorizonVersionedTimeSeries)
        True
    """
    return RestrictedHorizonVersionedTimeSeries(dataset, horizon)


__all__ = [
    "ConcatenatedVersionedTimeSeries",
    "RestrictedHorizonVersionedTimeSeries",
    "concat_featurewise",
    "restrict_horizon",
]
