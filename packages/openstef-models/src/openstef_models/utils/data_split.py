from datetime import datetime
from typing import cast

import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr

from openstef_core.base_model import BaseConfig
from openstef_core.exceptions import NotFittedError


def split_by_dates(
    index: pd.DatetimeIndex,
    data: pd.DataFrame,
    dates_test: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = cast("pd.Series[bool]", index.normalize().isin(dates_test))  # type: ignore
    train_data, test_data = data[~mask], data[mask]
    return train_data, test_data


def split_by_date(
    index: pd.DatetimeIndex,
    data: pd.DataFrame,
    split_date: datetime | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = cast(pd.Series, index).searchsorted(split_date, side="left")
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    return train_data, test_data


class ChronologicalTrainTestSplitter(BaseConfig):
    """Train-test splitter that splits data chronologically.

    Divides the dataset into training and testing sets based on temporal order,
    ensuring that all training data comes before all testing data. This is the
    standard approach for time series forecasting evaluation.

    The split point is determined by the test_fraction parameter, placing the
    most recent portion of data in the test set.
    """

    test_fraction: float = Field(
        default=0.2,
        description="Fraction of data to include in the test split. Set to 0.0 to disable splitting.",
        ge=0.0,
        le=1.0,
    )

    _split_date: pd.Timestamp | None = PrivateAttr(default=None)

    @property
    def is_fitted(self) -> bool:
        return self._split_date is not None

    def fit(self, index: pd.DatetimeIndex) -> None:
        if not 0.0 <= self.test_fraction <= 1.0:
            raise ValueError("test_fraction must be between 0 and 1.")

        # Handle no-split case (test_fraction = 0.0)
        if self.test_fraction == 0.0:
            self._split_date = index[-1]  # Set to last index so test set is empty
            return

        n_total = len(index)
        n_test = int(n_total * self.test_fraction)
        n_test = min(n_test, n_total - 1)  # Ensure at least one for train if possible
        if n_total > 1 and n_test == 0:
            n_test = 1  # Ensure at least one for test if possible
        n_train = n_total - n_test
        self._split_date = index[n_train]

    def transform(self, index: pd.DatetimeIndex, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._split_date is None:
            raise NotFittedError(self.__class__.__name__)

        return split_by_date(data=data, index=index, split_date=self._split_date)


class StratifiedTrainTestSplitter(BaseConfig):
    """Train-test splitter with stratification on extreme values.

    Splits data while ensuring that extreme high and low values are proportionally
    represented in both training and testing sets. This helps maintain representative
    distributions for model evaluation, especially important for forecasting tasks
    where extreme events are critical.

    Falls back to chronological splitting if there are too few days for stratification.
    """

    test_fraction: float = Field(
        default=0.2,
        description="Fraction of data to include in the test split. Set to 0.0 to disable splitting.",
        ge=0.0,
        le=1.0,
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
    def is_fitted(self) -> bool:
        return self._test_dates is not None or self._chronological_splitter is not None

    def fit(
        self,
        index: pd.DatetimeIndex,
        target_series: pd.Series,
    ) -> None:
        index_dates = index.normalize()
        n_unique_days = index_dates.nunique()

        # If not enough days, fall back to simple chronological split
        if n_unique_days < self.min_days_for_stratification:
            self._chronological_splitter = ChronologicalTrainTestSplitter(test_fraction=self.test_fraction)
            self._chronological_splitter.fit(index=index)
            return

        rng = np.random.default_rng(self.random_state)

        # Get extreme day groups
        max_days, min_days, other_days = self._get_extreme_days(
            target_series=target_series, fraction=self.stratification_fraction
        )

        # Split each group proportionally between train and test
        test_max_days, _ = self._sample_dates_for_split(dates=max_days, test_fraction=self.test_fraction, rng=rng)
        test_min_days, _ = self._sample_dates_for_split(dates=min_days, test_fraction=self.test_fraction, rng=rng)
        test_other_days, _ = self._sample_dates_for_split(dates=other_days, test_fraction=self.test_fraction, rng=rng)

        # Combine all train and test dates
        self._test_dates = cast(pd.DatetimeIndex, test_max_days.union(test_min_days).union(test_other_days))

    def transform(self, index: pd.DatetimeIndex, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._chronological_splitter is not None:
            return self._chronological_splitter.transform(index=index, data=data)

        if self._test_dates is None:
            raise NotFittedError(self.__class__.__name__)

        return split_by_dates(index=index, data=data, dates_test=self._test_dates)

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
            return pd.DatetimeIndex([]), dates

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
