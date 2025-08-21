# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""This module provides the CyclicFeatures transform, which generates cyclic
features from datetime indices in time series datasets based on sine and cosine
components.
"""

import numpy as np
import pandas as pd

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.transforms import TimeSeriesTransform

NUM_DAYS_IN_YEAR = 365.25
NUM_MONTHS_IN_YEAR = 12
NUM_DAYS_IN_WEEK = 7
NUM_SECONDS_IN_A_DAY = 24 * 60 * 60


class CyclicFeatures(TimeSeriesTransform):
    """Transform that generates cyclic temporal features from datetime indices.

    Converts temporal information into sine and cosine components that preserve
    the periodic nature of time. This encoding ensures temporal boundaries
    (e.g., end of day/week/year) are properly connected.

    Generates 6 features:
        - season_sine, season_cosine: Based on day of year (365.25 day cycle)
        - day0fweek_sine, day0fweek_cosine: Based on day of week (7 day cycle)
        - month_sine, month_cosine: Based on month of year (12 month cycle)

    Example:
        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> from openstef_core.feature_engineering.temporal_transforms.cyclic_features import CyclicFeatures
        >>>
        >>> # Create sample dataset
        >>> data = pd.DataFrame({
        ...     'load': [100, 120, 110]
        ... }, index=pd.date_range('2025-01-01', periods=3, freq='1h'))
        >>> dataset = TimeSeriesDataset(data, timedelta(hours=1))
        >>>
        >>> # Apply cyclic features
        >>> transform = CyclicFeatures()
        >>> transform.fit(dataset)
        >>> transformed = transform.transform(dataset)
        >>> len(transformed.feature_names)
        9
        >>> 'season_sine' in transformed.feature_names
        True
    """

    cyclic_features: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def _compute_sine(phase: pd.Index, period: float) -> np.ndarray:
        """Compute sine component for cyclic encoding.

        Args:
            phase: Position in cycle (e.g., day of year, hour of day).
            period: Complete cycle length (e.g., 365.25 for yearly).

        Returns:
            Sine values ranging from -1 to 1.
        """
        return np.sin(2 * np.pi * phase / period)

    @staticmethod
    def _compute_cosine(phase: pd.Index, period: float) -> np.ndarray:
        """Compute cosine component for cyclic encoding.

        Args:
            phase: Position in cycle (e.g., day of year, hour of day).
            period: Complete cycle length (e.g., 365.25 for yearly).

        Returns:
            Cosine values ranging from -1 to 1.
        """
        return np.cos(2 * np.pi * phase / period)

    def _compute_seasonal_feature(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Compute seasonal features based on day of year.

        Args:
            index: DatetimeIndex to extract day of year from.

        Returns:
            DataFrame with season_sine and season_cosine columns.
        """
        return pd.DataFrame(
            {
                "season_sine": self._compute_sine(index.dayofyear, NUM_DAYS_IN_YEAR),
                "season_cosine": self._compute_cosine(index.dayofyear, NUM_DAYS_IN_YEAR),
            },
            index=index,
        )

    def _compute_monthly_features(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Compute monthly features based on month of year.

        Args:
            index: DatetimeIndex to extract month from.

        Returns:
            DataFrame with month_sine and month_cosine columns.
        """
        return pd.DataFrame(
            {
                "month_sine": self._compute_sine(index.month, NUM_MONTHS_IN_YEAR),
                "month_cosine": self._compute_cosine(index.month, NUM_MONTHS_IN_YEAR),
            },
            index=index,
        )

    def _compute_weekday_features(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Compute weekday features based on day of week.

        Args:
            index: DatetimeIndex to extract day of week from.

        Returns:
            DataFrame with day0fweek_sine and day0fweek_cosine columns.
        """
        return pd.DataFrame(
            {
                "day0fweek_sine": self._compute_sine(index.day_of_week, NUM_DAYS_IN_WEEK),
                "day0fweek_cosine": self._compute_cosine(index.day_of_week, NUM_DAYS_IN_WEEK),
            },
            index=index,
        )

    def _compute_time_of_day_features(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Compute time of day features based on seconds in the day.

        Args:
            index: DatetimeIndex to extract time of day from.

        Returns:
            DataFrame with time0fday_sine and time0fday_cosine columns.
        """
        seconds_in_day = index.second + index.minute * 60 + index.hour * 60 * 60
        period_of_day = 2 * np.pi * seconds_in_day / NUM_SECONDS_IN_A_DAY

        return pd.DataFrame(
            {
                "time0fday_sine": self._compute_sine(period_of_day, NUM_SECONDS_IN_A_DAY),
                "time0fday_cosine": self._compute_cosine(period_of_day, NUM_SECONDS_IN_A_DAY),
            },
            index=index,
        )

    def fit(self, data: TimeSeriesDataset) -> None:
        """Fit the transform by computing cyclic features from the dataset's index.

        Args:
            data: Time series dataset with DatetimeIndex.
        """
        self.cyclic_features = pd.concat(
            [
                self._compute_seasonal_feature(data.index),
                self._compute_weekday_features(data.index),
                self._compute_monthly_features(data.index),
                self._compute_time_of_day_features(data.index),
            ],
            axis=1,
        )

    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        """Transform dataset by adding cyclic features as new columns.

        Args:
            data: Time series dataset to transform.

        Returns:
            New TimeSeriesDataset with original data plus 6 cyclic features.
        """
        return TimeSeriesDataset(
            data=pd.concat(
                [data.data, self.cyclic_features],
                axis=1,
            ),
            sample_interval=data.sample_interval,
        )
