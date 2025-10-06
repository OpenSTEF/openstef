# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Utilities for splitting time series datasets into train and test sets.

This module provides classes and functions for dividing time series data into
training and testing subsets using different splitting strategies. These utilities
are essential for model evaluation and validation in time series forecasting tasks.
"""

from abc import abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Self, cast, overload, override

import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import MultiHorizon, VersionedTimeSeriesPart
from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.datasets.versioned_timeseries import VersionedTimeSeriesDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.mixins import State, Transform

if TYPE_CHECKING:
    from openstef_core.types import LeadTime


class BaseTrainTestSplitter(
    BaseConfig,
    Transform[
        TimeSeriesDataset | VersionedTimeSeriesDataset,
        tuple[TimeSeriesDataset, TimeSeriesDataset] | tuple[VersionedTimeSeriesDataset, VersionedTimeSeriesDataset],
    ],
):
    """Abstract base class for train-test splitters.

    Provides a common interface for splitting time series datasets into training
    and testing subsets. Subclasses implement different splitting strategies
    (chronological, stratified, etc.).

    Invariants:
        - transform() preserves temporal ordering in both train and test sets
        - Output datasets have the same structure as input (columns, types)
    """

    test_fraction: float = Field(
        default=0.2, description="Fraction of data to include in the test split.", ge=0.0, lt=1.0
    )

    @overload
    def transform(
        self, data: VersionedTimeSeriesDataset
    ) -> tuple[VersionedTimeSeriesDataset, VersionedTimeSeriesDataset]: ...

    @overload
    def transform(self, data: TimeSeriesDataset) -> tuple[TimeSeriesDataset, TimeSeriesDataset]: ...

    @abstractmethod
    @override
    def transform(
        self, data: TimeSeriesDataset | VersionedTimeSeriesDataset
    ) -> tuple[TimeSeriesDataset, TimeSeriesDataset] | tuple[VersionedTimeSeriesDataset, VersionedTimeSeriesDataset]:
        """Split the dataset into train and test sets.

        Args:
            data: The dataset to split.

        Returns:
            Tuple of (train_dataset, test_dataset).
        """

    def transform_multihorizon[T: TimeSeriesDataset](
        self,
        data: MultiHorizon[T],
    ) -> tuple[MultiHorizon[TimeSeriesDataset], MultiHorizon[TimeSeriesDataset]]:
        """Split a multi-horizon dataset into train and test sets.

        Applies the splitting strategy to each horizon independently.

        Args:
            data: Multi-horizon dataset to split.

        Returns:
            Tuple of (train_multihorizon, test_multihorizon).
        """
        test_data: dict[LeadTime, TimeSeriesDataset] = {}
        train_data: dict[LeadTime, TimeSeriesDataset] = {}

        for horizon, dataset in data.items():
            train_split, test_split = self.transform(dataset)
            test_data[horizon] = test_split
            train_data[horizon] = train_split

        return MultiHorizon(train_data), MultiHorizon(test_data)

    def fit_multihorizon[T: TimeSeriesDataset](
        self,
        data: MultiHorizon[T],
    ) -> None:
        """Fit the splitter on a multi-horizon dataset.

        Fits the splitter using only the first horizon's data, as all horizons
        share the same temporal index.

        Args:
            data: Multi-horizon dataset to fit on.
        """
        for dataset in data.values():
            self.fit(dataset)
            break  # Fit only on the first horizon since they have the same index

    @override
    def to_state(self) -> State:
        return None

    @override
    def from_state(self, state: State) -> Self:
        return self  # There is no need to save any state since model is trained only once


class ChronologicalTrainTestSplitter(BaseTrainTestSplitter):
    """Train-test splitter that splits data chronologically.

    Divides the dataset into training and testing sets based on temporal order,
    ensuring that all training data comes before all testing data. This is the
    standard approach for time series forecasting evaluation.

    The split point is determined by the test_fraction parameter, placing the
    most recent portion of data in the test set.
    """

    test_fraction: float = Field(
        default=0.2, description="Fraction of data to include in the test split.", ge=0.0, lt=1.0
    )

    _split_date: pd.Timestamp | None = PrivateAttr(default=None)

    @property
    @override
    def is_fitted(self) -> bool:
        return self._split_date is not None

    @override
    def fit(self, data: TimeSeriesDataset | VersionedTimeSeriesDataset) -> None:
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError("test_fraction must be between 0 and 1.")

        n_total = len(data.index)
        n_test = int(n_total * self.test_fraction)
        n_test = min(n_test, n_total - 1)  # Ensure at least one for train if possible
        if n_total > 1 and n_test == 0:
            n_test = 1  # Ensure at least one for test if possible
        n_train = n_total - n_test
        self._split_date = data.index[n_train]

    @overload
    def transform(
        self, data: VersionedTimeSeriesDataset
    ) -> tuple[VersionedTimeSeriesDataset, VersionedTimeSeriesDataset]: ...

    @overload
    def transform(self, data: TimeSeriesDataset) -> tuple[TimeSeriesDataset, TimeSeriesDataset]: ...

    @override
    def transform(
        self, data: TimeSeriesDataset | VersionedTimeSeriesDataset
    ) -> tuple[TimeSeriesDataset, TimeSeriesDataset] | tuple[VersionedTimeSeriesDataset, VersionedTimeSeriesDataset]:
        if self._split_date is None:
            raise NotFittedError(self.__class__.__name__)

        return split_by_date(dataset=data, split_date=self._split_date)


class StratifiedTrainTestSplitter(BaseTrainTestSplitter):
    """Train-test splitter with stratification on extreme values.

    Splits data while ensuring that extreme high and low values are proportionally
    represented in both training and testing sets. This helps maintain representative
    distributions for model evaluation, especially important for forecasting tasks
    where extreme events are critical.

    Falls back to chronological splitting if there are too few days for stratification.
    """

    test_fraction: float = Field(
        default=0.2, description="Fraction of data to include in the test split.", ge=0.0, lt=1.0
    )
    stratification_fraction: float = Field(
        default=0.15, description="Fraction of extreme days to consider for stratification."
    )
    target_column: str = Field(default="load", description="Column name containing the values to stratify on.")
    random_state: int = Field(default=42, description="Random seed for reproducible splits.")
    min_days_for_stratification: int = Field(default=4, description="Minimum days required for stratification.")

    _test_dates: pd.DatetimeIndex | None = PrivateAttr(default=None)
    _chronological_splitter: ChronologicalTrainTestSplitter | None = PrivateAttr(default=None)

    @property
    @override
    def is_fitted(self) -> bool:
        return self._test_dates is not None or self._chronological_splitter is not None

    @override
    def fit(self, data: TimeSeriesDataset | VersionedTimeSeriesDataset) -> None:
        index_dates = data.index.normalize()
        n_unique_days = index_dates.nunique()

        # If not enough days, fall back to simple chronological split
        if n_unique_days < self.min_days_for_stratification:
            self._chronological_splitter = ChronologicalTrainTestSplitter(test_fraction=self.test_fraction)
            self._chronological_splitter.fit(data)
            return

        rng = np.random.default_rng(self.random_state)

        # Get extreme day groups
        target_series = (
            data.data[self.target_column]
            if isinstance(data, TimeSeriesDataset)
            else data.select_version().data[self.target_column]
        )
        max_days, min_days, other_days = self._get_extreme_days(
            target_series=target_series, fraction=self.stratification_fraction
        )

        # Split each group proportionally between train and test
        _, test_max_days = self._sample_dates_for_split(dates=max_days, test_fraction=self.test_fraction, rng=rng)
        _, test_min_days = self._sample_dates_for_split(dates=min_days, test_fraction=self.test_fraction, rng=rng)
        _, test_other_days = self._sample_dates_for_split(dates=other_days, test_fraction=self.test_fraction, rng=rng)

        # Combine all train and test dates
        self._test_dates = cast(pd.DatetimeIndex, test_max_days.union(test_min_days).union(test_other_days))

    @overload
    def transform(
        self, data: VersionedTimeSeriesDataset
    ) -> tuple[VersionedTimeSeriesDataset, VersionedTimeSeriesDataset]: ...

    @overload
    def transform(self, data: TimeSeriesDataset) -> tuple[TimeSeriesDataset, TimeSeriesDataset]: ...

    @override
    def transform(
        self, data: TimeSeriesDataset | VersionedTimeSeriesDataset
    ) -> tuple[TimeSeriesDataset, TimeSeriesDataset] | tuple[VersionedTimeSeriesDataset, VersionedTimeSeriesDataset]:
        if self._chronological_splitter is not None:
            return self._chronological_splitter.transform(data)

        if self._test_dates is None:
            raise NotFittedError(self.__class__.__name__)

        return split_by_dates(dataset=data, dates_test=self._test_dates)

    @staticmethod
    def _sample_dates_for_split(
        dates: pd.DatetimeIndex, test_fraction: float, rng: np.random.Generator
    ) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
        if dates.empty:
            return pd.DatetimeIndex([]), pd.DatetimeIndex([])

        n_test = max(1, int(test_fraction * len(dates)))
        n_test = min(n_test, len(dates) - 1)  # Ensure at least one for train if possible

        if len(dates) == 1:
            # Only one date, put in train
            return dates, pd.DatetimeIndex([])

        test_dates = pd.DatetimeIndex(np.sort(rng.choice(dates, size=n_test, replace=False)))
        train_dates = dates.difference(test_dates, sort=True)  # type: ignore

        return test_dates, train_dates

    @staticmethod
    def _get_extreme_days(
        target_series: pd.Series,
        fraction: float = 0.1,
    ) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
        if not isinstance(target_series.index, pd.DatetimeIndex):
            raise TypeError("target_series must have a DatetimeIndex.")

        # Compute daily min and max once
        daily_agg: pd.DataFrame = target_series.resample("1D").agg(["min", "max"])  # type: ignore
        n_days = len(daily_agg)
        n_extremes = max(int(fraction * n_days), 2)

        # Sort once
        max_days = cast(pd.DatetimeIndex, daily_agg["max"].nlargest(n_extremes).index)
        min_days = cast(pd.DatetimeIndex, daily_agg["min"].nsmallest(n_extremes).index)

        all_days = cast(pd.DatetimeIndex, daily_agg.index)
        other_days = all_days.difference(other=max_days.union(other=min_days))  # type: ignore

        return max_days, min_days, other_days


@overload
def split_by_date(
    dataset: VersionedTimeSeriesDataset,
    split_date: datetime | pd.Timestamp,
) -> tuple[VersionedTimeSeriesDataset, VersionedTimeSeriesDataset]: ...


@overload
def split_by_date(
    dataset: TimeSeriesDataset,
    split_date: datetime | pd.Timestamp,
) -> tuple[TimeSeriesDataset, TimeSeriesDataset]: ...


def split_by_date(
    dataset: TimeSeriesDataset | VersionedTimeSeriesDataset,
    split_date: datetime | pd.Timestamp,
) -> tuple[TimeSeriesDataset, TimeSeriesDataset] | tuple[VersionedTimeSeriesDataset, VersionedTimeSeriesDataset]:
    """Split a dataset into train and test sets based on a specific date.

    Args:
        dataset: The dataset to split.
        split_date: The date to split on. Data before this date goes to train,
            data at/after goes to test.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    if isinstance(dataset, TimeSeriesDataset):
        split_idx = cast(pd.Series, dataset.index).searchsorted(split_date, side="left")
        train_data = dataset.data.iloc[:split_idx]
        test_data = dataset.data.iloc[split_idx:]

        return (
            TimeSeriesDataset(data=train_data, sample_interval=dataset.sample_interval, is_sorted=True),
            TimeSeriesDataset(data=test_data, sample_interval=dataset.sample_interval, is_sorted=True),
        )

    # VersionedTimeSeriesDataset
    split_idx = [part.data[part.timestamp_column].searchsorted(split_date, side="left") for part in dataset.data_parts]

    train_parts = [
        VersionedTimeSeriesPart(
            data=part.data.iloc[:idx],
            sample_interval=part.sample_interval,
            timestamp_column=part.timestamp_column,
            available_at_column=part.available_at_column,
            is_sorted=True,
        )
        for part, idx in zip(dataset.data_parts, split_idx, strict=True)
    ]
    test_parts = [
        VersionedTimeSeriesPart(
            data=part.data.iloc[idx:],
            sample_interval=part.sample_interval,
            timestamp_column=part.timestamp_column,
            available_at_column=part.available_at_column,
            is_sorted=True,
        )
        for part, idx in zip(dataset.data_parts, split_idx, strict=True)
    ]

    return (
        VersionedTimeSeriesDataset(data_parts=train_parts),
        VersionedTimeSeriesDataset(data_parts=test_parts),
    )


@overload
def split_by_dates(
    dataset: VersionedTimeSeriesDataset,
    dates_test: pd.DatetimeIndex,
) -> tuple[VersionedTimeSeriesDataset, VersionedTimeSeriesDataset]: ...


@overload
def split_by_dates(
    dataset: TimeSeriesDataset,
    dates_test: pd.DatetimeIndex,
) -> tuple[TimeSeriesDataset, TimeSeriesDataset]: ...


def split_by_dates(
    dataset: TimeSeriesDataset | VersionedTimeSeriesDataset,
    dates_test: pd.DatetimeIndex,
) -> tuple[TimeSeriesDataset, TimeSeriesDataset] | tuple[VersionedTimeSeriesDataset, VersionedTimeSeriesDataset]:
    """Split a dataset into train and test sets based on specific dates.

    Args:
        dataset: The dataset to split.
        dates_test: Dates to include in the test set. All other dates go to training.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    if isinstance(dataset, TimeSeriesDataset):
        mask = cast("pd.Series[bool]", dataset.index.normalize().isin(dates_test))  # type: ignore

        train_data = dataset.data[~mask]
        test_data = dataset.data[mask]

        return (
            TimeSeriesDataset(data=train_data, sample_interval=dataset.sample_interval, is_sorted=True),
            TimeSeriesDataset(data=test_data, sample_interval=dataset.sample_interval, is_sorted=True),
        )

    # VersionedTimeSeriesDataset
    masks = [
        cast("pd.Series[bool]", part.data[part.timestamp_column].dt.normalize().isin(dates_test))  # type: ignore
        for part in dataset.data_parts
    ]
    train_parts = [
        VersionedTimeSeriesPart(
            data=part.data[~mask],
            sample_interval=part.sample_interval,
            timestamp_column=part.timestamp_column,
            available_at_column=part.available_at_column,
            is_sorted=True,
        )
        for part, mask in zip(dataset.data_parts, masks, strict=True)
    ]
    test_parts = [
        VersionedTimeSeriesPart(
            data=part.data[mask],
            sample_interval=part.sample_interval,
            timestamp_column=part.timestamp_column,
            available_at_column=part.available_at_column,
            is_sorted=True,
        )
        for part, mask in zip(dataset.data_parts, masks, strict=True)
    ]

    return (
        VersionedTimeSeriesDataset(data_parts=train_parts),
        VersionedTimeSeriesDataset(data_parts=test_parts),
    )
