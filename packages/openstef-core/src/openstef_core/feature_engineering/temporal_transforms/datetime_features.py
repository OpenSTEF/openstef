# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for extracting datetime-based features from time series data.

This module provides functionality to compute datetime features that indicate
temporal patterns like weekdays, weekends, and other time-based characteristics
from the datetime index of time series datasets.
"""

from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.transforms import TimeSeriesTransform

MIN_WEEKEND_IDX: int = 5  # Saturday
SUNDAY_IDX: int = 6


class DatetimeFeaturesTransform(TimeSeriesTransform):
    """Transform that adds datetime features to time series data.

    Computes features that are derived from the datetime index of the dataset.

    The features added are:
        - is_week_day: 1 if the day is a weekday (Monday to Friday),
            0 otherwise (Saturday or Sunday).
        - is_weekend_day: 1 if the day is a weekend day (Saturday or Sunday),
            0 otherwise (Monday to Friday).
        - is_sunday: 1 if the day is a Sunday, 0 otherwise.
        - month_of_year: Month of the year (1 to 12).
        - quarter_of_year: Quarter of the year (1 to 4).

    Example:
        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> from openstef_core.feature_engineering.temporal_transforms.datetime_features import (
        ...     DatetimeFeatures
        ... )
        >>>
        >>> # Create sample dataset
        >>> data = pd.DataFrame({
        ...     'load': [100, 120, 110]
        ... }, index=pd.date_range('2025-01-01', periods=3,
        ... freq='D'))
        >>> dataset = TimeSeriesDataset(data, sample_interval=timedelta(minutes=15))
        >>> transform = DatetimeFeatures()
        >>> transformed_dataset = transform.fit_transform(dataset)
        >>> transformed_dataset.feature_names
        ['load', 'is_week_day', 'is_weekend_day', 'is_sunday', 'month_of_year', 'quarter_of_year']
        >>> transformed_dataset.data["is_week_day"].tolist()
        [1, 1, 1]
        >>> transformed_dataset.data["month_of_year"].tolist()
        [1, 1, 1]
    """

    def __init__(self) -> None:
        """Initialize the DatetimeFeatures transform."""
        self._datetime_features: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def _is_weekend_day(index: pd.DatetimeIndex) -> npt.NDArray[np.int64]:
        """Check if the day is a weekend day (Saturday or Sunday).

        Args:
            index: DatetimeIndex to check for weekend days.

        Returns:
            Numpy array where 1 indicates a weekend day and 0 otherwise.
        """
        return (index.weekday >= MIN_WEEKEND_IDX).astype(int)

    @staticmethod
    def _is_weekday(index: pd.DatetimeIndex) -> npt.NDArray[np.int64]:
        """Check if the day is a weekday (Monday to Friday).

        Args:
            index: DatetimeIndex to check for weekdays.

        Returns:
            Numpy array where 1 indicates a weekday and 0 otherwise.
        """
        return (index.weekday < MIN_WEEKEND_IDX).astype(int)

    @staticmethod
    def _is_sunday(index: pd.DatetimeIndex) -> npt.NDArray[np.int64]:
        """Check if the day is a Sunday.

        Args:
            index: DatetimeIndex to check for Sundays.

        Returns:
            Numpy array where 1 indicates Sunday and 0 otherwise.
        """
        return (index.weekday == SUNDAY_IDX).astype(int)

    @staticmethod
    def _month_of_year(index: pd.DatetimeIndex) -> npt.NDArray[np.int64]:
        """Get the month of the year (1 to 12).

        Args:
            index: DatetimeIndex to extract the month from.

        Returns:
            Numpy array of months corresponding to each date in the index.
        """
        return cast(pd.Series, index.month).to_numpy()

    @staticmethod
    def _quarter_of_year(index: pd.DatetimeIndex) -> npt.NDArray[np.int64]:
        """Get the quarter of the year (1 to 4).

        Args:
            index: DatetimeIndex to extract the quarter from.

        Returns:
            Numpy array of quarters corresponding to each date in the index.
        """
        return cast(pd.Series, index.quarter).to_numpy()

    def fit(self, data: TimeSeriesDataset) -> None:
        """Fit the transform by computing datetime features from the data's index.

        Args:
            data: Time series dataset with DatetimeIndex.
        """
        self._datetime_features = pd.DataFrame(
            {
                "is_week_day": self._is_weekday(data.index).astype(int),
                "is_weekend_day": self._is_weekend_day(data.index).astype(int),
                "is_sunday": self._is_sunday(data.index).astype(int),
                "month_of_year": self._month_of_year(data.index),
                "quarter_of_year": self._quarter_of_year(data.index),
            },
            index=data.index,
        )

    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        """Transform the input time series data by adding datetime features.

        Args:
            data: Time series dataset to transform.

        Returns:
            New TimeSeriesDataset with original data plus datetime features.
        """
        return TimeSeriesDataset(
            pd.concat([data.data, self._datetime_features], axis=1),
            sample_interval=data.sample_interval,
        )
