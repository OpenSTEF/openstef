# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0


"""Transform for extracting cyclic features from time series data.

This module provides the CyclicFeatures transform, which generates cyclic
features from datetime indices in time series datasets based on sine and cosine
components.
"""

from typing import Literal, override

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

type CyclicFeatureName = Literal["timeOfDay", "season", "dayOfWeek", "month"]


class CyclicFeaturesTransform(BaseConfig, TimeSeriesTransform):
    """Transform that generates cyclic temporal features from datetime indices.

    Converts temporal information into sine and cosine components that preserve
    the periodic nature of time. This encoding ensures temporal boundaries
    (e.g., end of day/week/year) are properly connected.

    The features generated depend on the included_features configuration:
        - season: season_sine, season_cosine (based on day of year, 365.25 day cycle)
        - dayOfWeek: dayOfWeek_sine, dayOfWeek_cosine (based on day of week, 7 day cycle)
        - month: month_sine, month_cosine (based on month of year, 12 month cycle)
        - timeOfDay: timeOfDay_sine, timeOfDay_cosine (based on seconds in day, 24 hour cycle)

    Example:
        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> from openstef_core.feature_engineering.temporal_transforms.cyclic_features_transform import (
        ...     CyclicFeaturesTransform,
        ... )
        >>>
        >>> # Create sample dataset
        >>> data = pd.DataFrame({
        ...     'load': [100, 120, 110]
        ... }, index=pd.date_range('2025-01-01 12:00:00', periods=3, freq='1h'))
        >>> dataset = TimeSeriesDataset(data, timedelta(hours=1))
        >>>
        >>> # Apply cyclic features with custom configuration
        >>> transform = CyclicFeaturesTransform(included_features=["season", "timeOfDay"])
        >>> transformed = transform.transform(dataset)
        >>> result = transformed.data[['season_sine', 'timeOfDay_sine']].round(3)
        >>> print(result.head(2))
                             season_sine  timeOfDay_sine
        2025-01-01 12:00:00        0.017          -0.000
        2025-01-01 13:00:00        0.017          -0.259
    """

    included_features: list[CyclicFeatureName] = Field(
        default_factory=lambda: ["timeOfDay", "season", "dayOfWeek", "month"],
        description="List of cyclic features to include in the transformation. "
        "Options are 'timeOfDay', 'season', 'dayOfWeek', and 'month'.",
    )

    @staticmethod
    def _compute_cyclic_feature(
        phase: pd.Index,
        period: float,
        index: pd.Index,
        name: str,
    ) -> pd.DataFrame:
        """Compute sine and cosine components for cyclic encoding.

        Args:
            phase: Position in cycle (e.g., day of year, hour of day).
            period: Complete cycle length (e.g., 365.25 for yearly).
            index: Index to align the resulting DataFrame.
            name: Base name for the resulting columns.

        Returns:
            DataFrame with sine and cosine columns.
        """
        t = 2 * np.pi * phase / period

        return pd.DataFrame(
            {
                f"{name}_sine": np.sin(t),
                f"{name}_cosine": np.cos(t),
            },
            index=index,
        )

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        """Transform dataset by adding cyclic features as new columns.

        Args:
            data: Time series dataset to transform.

        Returns:
            New TimeSeriesDataset with original data plus cyclic features
            (number depends on included_features configuration).
        """
        features: list[pd.DataFrame] = []
        if "season" in self.included_features:
            features.append(
                self._compute_cyclic_feature(
                    phase=data.index.dayofyear,
                    period=NUM_DAYS_IN_YEAR,
                    index=data.index,
                    name="season",
                )
            )

        if "dayOfWeek" in self.included_features:
            features.append(
                self._compute_cyclic_feature(
                    phase=data.index.day_of_week,
                    period=NUM_DAYS_IN_WEEK,
                    index=data.index,
                    name="dayOfWeek",
                )
            )

        if "month" in self.included_features:
            features.append(
                self._compute_cyclic_feature(
                    phase=data.index.month,
                    period=NUM_MONTHS_IN_YEAR,
                    index=data.index,
                    name="month",
                )
            )

        if "timeOfDay" in self.included_features:
            features.append(
                self._compute_cyclic_feature(
                    phase=data.index.hour * 3600 + data.index.minute * 60 + data.index.second,
                    period=NUM_SECONDS_IN_A_DAY,
                    index=data.index,
                    name="timeOfDay",
                )
            )

        return TimeSeriesDataset(
            data=pd.concat(
                [data.data, *features],
                axis=1,
            ),
            sample_interval=data.sample_interval,
        )


__all__ = ["CyclicFeaturesTransform"]
