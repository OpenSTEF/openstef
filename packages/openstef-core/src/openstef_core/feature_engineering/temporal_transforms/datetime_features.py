# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
import numpy as np
import pandas as pd

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.transforms import TimeSeriesTransform


class DatetimeFeatures(TimeSeriesTransform):
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
        >>> from openstef_core.feature_engineering.temporal_transforms.datetime_features import DatetimeFeatures  # noqa: E501
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
        self.datetime_features: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def _is_weekend_day(index: pd.DatetimeIndex) -> np.ndarray:
        """Check if the day is a weekend day (Saturday or Sunday)."""
        return (index.weekday >= 5).astype(int)

    @staticmethod
    def _is_weekday(index: pd.DatetimeIndex) -> np.ndarray:
        """Check if the day is a weekday (Monday to Friday)."""
        return (index.weekday < 5).astype(int)

    @staticmethod
    def _is_sunday(index: pd.DatetimeIndex) -> np.ndarray:
        """Check if the day is a Sunday."""
        return (index.weekday == 6).astype(int)

    @staticmethod
    def _month_of_year(index: pd.DatetimeIndex) -> np.ndarray:
        """Get the month of the year (1 to 12)."""
        return index.month.to_numpy()

    @staticmethod
    def _quarter_of_year(index: pd.DatetimeIndex) -> np.ndarray:
        """Get the quarter of the year (1 to 4)."""
        return index.quarter.to_numpy()

    def fit(self, data: TimeSeriesDataset) -> None:
        """Fit the transform by computing datetime features from the data's index.

        Args:
            data: Time series dataset with DatetimeIndex.
        """
        self.datetime_features = pd.DataFrame(
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
            pd.concat(
                [data.data, self.datetime_features],
                axis=1
            ),
            sample_interval=data.sample_interval,
        )
