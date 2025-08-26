# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0


"""Transform for extracting cyclic features from time series data.

This module provides the CyclicFeatures transform, which generates cyclic
features from datetime indices in time series datasets based on sine and cosine
components.
"""

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.transforms import TimeSeriesTransform

NUM_DAYS_IN_YEAR = 365.25
NUM_MONTHS_IN_YEAR = 12
NUM_DAYS_IN_WEEK = 7
NUM_SECONDS_IN_A_DAY = 24 * 60 * 60


class CyclicFeaturesConfig(BaseConfig):
    """Configuration for the CyclicFeatures transform.

    By default, all cyclic features are included.
    """

    included_features: list[Literal["timeOfDay", "season", "dayOfWeek", "month"]] = Field(
        default_factory=lambda: ["timeOfDay", "season", "dayOfWeek", "month"],
        description="List of cyclic features to include in the transformation. "
        "Options are 'timeOfDay', 'season', 'dayOfWeek', and 'month'.",
    )


class CyclicFeatures(TimeSeriesTransform):
    """Transform that generates cyclic temporal features from datetime indices.

    Converts temporal information into sine and cosine components that preserve
    the periodic nature of time. This encoding ensures temporal boundaries
    (e.g., end of day/week/year) are properly connected.

    The features generated depend on the included_features configuration:
        - season: season_sine, season_cosine (based on day of year, 365.25 day cycle)
        - dayOfWeek: day0fweek_sine, day0fweek_cosine (based on day of week, 7 day cycle)
        - month: month_sine, month_cosine (based on month of year, 12 month cycle)
        - timeOfDay: time0fday_sine, time0fday_cosine (based on seconds in day, 24 hour cycle)

    Example:
        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> from openstef_core.feature_engineering.temporal_transforms.cyclic_features import (
        ...     CyclicFeatures, CyclicFeaturesConfig
        ... )
        >>>
        >>> # Create sample dataset
        >>> data = pd.DataFrame({
        ...     'load': [100, 120, 110]
        ... }, index=pd.date_range('2025-01-01', periods=3, freq='1h'))
        >>> dataset = TimeSeriesDataset(data, timedelta(hours=1))
        >>>
        >>> # Apply cyclic features with custom configuration
        >>> config = CyclicFeaturesConfig(included_features=["season", "timeOfDay"])
        >>> transform = CyclicFeatures(config)
        >>> transformed = transform.fit_transform(dataset)
        >>> 'season_sine' in transformed.data.columns
        True
        >>> 'time0fday_sine' in transformed.data.columns
        True
        >>> 'month_sine' in transformed.data.columns
        False
    """

    def __init__(
        self,
        config: CyclicFeaturesConfig,
    ):
        """Initialize the CyclicFeatures transform with a configuration."""
        self.config = config
        self.cyclic_features: pd.DataFrame = pd.DataFrame()

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
        feature_dataframes: list[pd.DataFrame] = []

        if "season" in self.config.included_features:
            feature_dataframes.append(self._compute_seasonal_feature(data.index))

        if "dayOfWeek" in self.config.included_features:
            feature_dataframes.append(self._compute_weekday_features(data.index))

        if "month" in self.config.included_features:
            feature_dataframes.append(self._compute_monthly_features(data.index))

        if "timeOfDay" in self.config.included_features:
            feature_dataframes.append(self._compute_time_of_day_features(data.index))

        if feature_dataframes:
            self.cyclic_features = pd.concat(feature_dataframes, axis=1)
        else:
            # Create empty DataFrame with the same index if no features are selected
            self.cyclic_features = pd.DataFrame(index=data.index)

    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        """Transform dataset by adding cyclic features as new columns.

        Args:
            data: Time series dataset to transform.

        Returns:
            New TimeSeriesDataset with original data plus cyclic features
            (number depends on included_features configuration).
        """
        return TimeSeriesDataset(
            data=pd.concat(
                [data.data, self.cyclic_features],
                axis=1,
            ),
            sample_interval=data.sample_interval,
        )
