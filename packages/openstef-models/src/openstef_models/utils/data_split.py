# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Time series dataset splitting utilities for training and evaluation.

Provides various strategies for splitting time series datasets into training,
validation, and test sets. Supports chronological splits, stratified splits
based on extreme values, and custom date-based splits.

Key functions handle the temporal nature of forecasting data, ensuring that
training data always precedes test data to prevent information leakage.
"""

from collections.abc import Callable
from datetime import datetime
from typing import cast

import numpy as np
import pandas as pd
from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset


def split_by_dates[T: TimeSeriesDataset](
    dataset: T,
    dates_test: pd.DatetimeIndex,
) -> tuple[T, T]:
    """Split a dataset into train and test sets based on specific dates.

    Args:
        dataset: The dataset to split.
        dates_test: Dates to include in the test set. All other dates go to training.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    mask = cast("pd.Series[bool]", dataset.index.normalize().isin(dates_test))  # type: ignore
    train_data, test_data = dataset.data[~mask], dataset.data[mask]
    return dataset._copy_with_data(train_data), dataset._copy_with_data(test_data)  # noqa: SLF001 - allow protected access, invariants are maintained


def split_by_date[T: TimeSeriesDataset](
    dataset: T,
    split_date: datetime | pd.Timestamp,
) -> tuple[T, T]:
    """Split a dataset into train and test sets based on a specific date.

    Args:
        dataset: The dataset to split.
        split_date: The date to split on. Data before this date goes to train,
            data at/after goes to test.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    split_idx = cast(pd.Series, dataset.index).searchsorted(split_date, side="left")
    train_data = dataset.data.iloc[:split_idx]
    test_data = dataset.data.iloc[split_idx:]
    return dataset._copy_with_data(train_data), dataset._copy_with_data(test_data)  # noqa: SLF001 - allow protected access, invariants are maintained


def chronological_train_test_split[T: TimeSeriesDataset](
    dataset: T,
    test_fraction: float,
) -> tuple[T, T]:
    """Split a dataset into train and test sets chronologically.

    Divides the dataset into training and testing sets based on temporal order,
    ensuring that all training data comes before all testing data. This is the
    standard approach for time series forecasting evaluation.

    The split point is determined by the test_fraction parameter, placing the
    most recent portion of data in the test set.

    Args:
        dataset: The dataset to split.
        test_fraction: Fraction of data to include in the test split.

    Returns:
        Tuple of (train_dataset, test_dataset).

    Raises:
        ValueError: If test_fraction is not between 0 and 1.
    """
    if not 0.0 <= test_fraction <= 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")

    if test_fraction == 0.0:
        # No test set
        return dataset, dataset._copy_with_data(dataset.data.iloc[0:0])  # noqa: SLF001 - allow protected access, invariants are maintained

    index_unique = dataset.index.unique()

    n_total = len(index_unique)
    n_test = int(n_total * test_fraction)
    n_test = min(n_test, n_total - 1)  # Ensure at least one for train if possible
    if n_total > 1 and n_test == 0:
        n_test = 1  # Ensure at least one for test if possible
    n_train = n_total - n_test

    split_date = index_unique[n_train]

    return split_by_date(dataset=dataset, split_date=split_date)


def stratified_train_test_split[T: TimeSeriesDataset](
    dataset: T,
    test_fraction: float,
    stratification_fraction: float = 0.15,
    target_column: str = "load",
    random_state: int = 42,
    min_days_for_stratification: int = 4,
) -> tuple[T, T]:
    """Split a dataset into train and test sets with stratification on extreme values.

    Splits data while ensuring that extreme high and low values are proportionally
    represented in both training and testing sets. This helps maintain representative
    distributions for model evaluation, especially important for forecasting tasks
    where extreme events are critical.

    Args:
        dataset: The dataset to split.
        test_fraction: Fraction of data to include in the test split.
        stratification_fraction: Fraction of extreme days to consider for stratification.
        target_column: Column name containing the values to stratify on.
        random_state: Random seed for reproducible splits.
        min_days_for_stratification: Minimum days required for stratification.

    Returns:
        Tuple of (train_dataset, test_dataset).

    Raises:
        ValueError: If test_fraction is not between 0 and 1.

    Note:
        Falls back to chronological splitting if there are too few days for stratification.
    """
    if not 0.0 <= test_fraction <= 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")

    index_dates = dataset.index.normalize()
    n_unique_days = index_dates.nunique()

    # If not enough days, fall back to simple chronological split
    if n_unique_days < min_days_for_stratification:
        return chronological_train_test_split(dataset=dataset, test_fraction=test_fraction)

    rng = np.random.default_rng(random_state)

    # Get extreme day groups
    target_series = dataset.select_features([target_column]).select_version().data[target_column]
    max_days, min_days, other_days = _get_extreme_days(target_series=target_series, fraction=stratification_fraction)

    # Split each group proportionally between train and test
    _, test_max_days = _sample_dates_for_split(dates=max_days, test_fraction=test_fraction, rng=rng)
    _, test_min_days = _sample_dates_for_split(dates=min_days, test_fraction=test_fraction, rng=rng)
    _, test_other_days = _sample_dates_for_split(dates=other_days, test_fraction=test_fraction, rng=rng)

    # Combine all train and test dates
    test_dates = cast(pd.DatetimeIndex, test_max_days.union(test_min_days).union(test_other_days))

    return split_by_dates(dataset=dataset, dates_test=test_dates)


def _sample_dates_for_split(
    dates: pd.DatetimeIndex,
    test_fraction: float,
    rng: np.random.Generator,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    if dates.empty:
        return pd.DatetimeIndex([]), pd.DatetimeIndex([])

    min_test_days = 1 if test_fraction > 0.0 else 0
    n_test = max(min_test_days, int(test_fraction * len(dates)))
    n_test = min(n_test, len(dates) - 1)  # Ensure at least one for train if possible

    if len(dates) == 1:
        # Only one date, put in train
        return pd.DatetimeIndex([]), dates

    test_dates = pd.DatetimeIndex(np.sort(rng.choice(dates, size=n_test, replace=False)))
    train_dates = dates.difference(test_dates, sort=True)  # type: ignore

    return train_dates, test_dates


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


def train_val_test_split[T](
    dataset: T,
    split_func: Callable[[T, float], tuple[T, T]],
    val_fraction: float,
    test_fraction: float,
) -> tuple[T, T, T]:
    """Split a dataset into train, validation, and test sets chronologically.

    Divides the dataset into training, validation, and testing sets based on temporal order,
    ensuring that all training data comes before all validation data, which comes before all testing data.

    The split points are determined by the val_fraction and test_fraction parameters.

    Args:
        dataset: The dataset to split.
        split_func: Function to use for splitting the dataset into two parts.
        val_fraction: Fraction of data to include in the validation split.
        test_fraction: Fraction of data to include in the test split.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).

    Raises:
        ValueError: If test_fraction + val_fraction is not less than 1.0.
    """
    if not 0.0 <= val_fraction <= 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")

    if not 0.0 <= test_fraction <= 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")

    if test_fraction + val_fraction >= 1.0:
        msg = f"test_fraction ({test_fraction}) + val_fraction ({val_fraction}) must be less than 1.0"
        raise ValueError(msg)

    # First split: separate test set from train+val
    train_val, test = split_func(dataset, test_fraction)

    # Calculate adjusted validation fraction for the remaining data
    # We want val_fraction of the *original* dataset size
    # From the remaining (1 - test_fraction), we need val_fraction
    # So: adjusted = val_fraction / (1 - test_fraction).
    adjusted_val_fraction = val_fraction / (1 - test_fraction)

    # Second split: separate validation from training
    train, val = split_func(train_val, adjusted_val_fraction)

    return train, val, test


class DataSplitter(BaseConfig):
    """Handles splitting of time series data into train, validation, and test sets.

    Supports stratified splitting to ensure representative data distribution
    across splits, particularly for extreme values in forecasting scenarios.
    """

    val_fraction: float = Field(
        default=0.15,
        description="Fraction of data to reserve for the validation set when automatic splitting is used.",
    )
    test_fraction: float = Field(
        default=0.1,
        description="Fraction of data to reserve for the test set when automatic splitting is used.",
    )
    stratification_fraction: float = Field(
        default=0.15,
        description="Fraction of extreme values to use for stratified splitting into train/test sets.",
    )
    min_days_for_stratification: int = Field(
        default=4,
        description="Minimum number of unique days required to perform stratified splitting.",
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducible splits when stratification is used.",
    )

    def split_dataset[T: TimeSeriesDataset](
        self,
        data: T,
        data_val: T | None = None,
        data_test: T | None = None,
        target_column: str = "load",
    ) -> tuple[T, T | None, T | None]:
        """Prepare and split input data into train, validation, and test sets.

        Args:
            data: Full dataset to split.
            data_val: Optional pre-split validation data.
            data_test: Optional pre-split test data.
            target_column: Column name containing the target variable for stratification.

        Returns:
            Tuple of (train_data, val_data, test_data) where val_data and test_data may be None.
        """
        # Apply splitting strategy
        input_data_train, input_data_val, input_data_test = train_val_test_split(
            dataset=data,
            split_func=lambda dataset, fraction: stratified_train_test_split(
                dataset=dataset,
                test_fraction=fraction,
                stratification_fraction=self.stratification_fraction,
                target_column=target_column,
                random_state=self.random_state,
                min_days_for_stratification=self.min_days_for_stratification,
            ),
            val_fraction=self.val_fraction if data_val is None else 0.0,
            test_fraction=self.test_fraction if data_test is None else 0.0,
        )
        input_data_val = data_val or input_data_val
        input_data_test = data_test or input_data_test

        if input_data_val.index.empty:
            input_data_val = None
        if input_data_test.index.empty:
            input_data_test = None

        return (input_data_train, input_data_val, input_data_test)


__all__ = [
    "DataSplitter",
    "chronological_train_test_split",
    "split_by_date",
    "split_by_dates",
    "stratified_train_test_split",
    "train_val_test_split",
]
