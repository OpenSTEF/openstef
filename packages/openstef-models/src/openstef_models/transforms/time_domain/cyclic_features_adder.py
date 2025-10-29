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
from openstef_core.transforms import TimeSeriesTransform

NUM_DAYS_IN_YEAR = 365.25
NUM_MONTHS_IN_YEAR = 12
NUM_DAYS_IN_WEEK = 7
NUM_SECONDS_IN_A_DAY = 24 * 60 * 60

type CyclicFeatureName = Literal["time_of_day", "season", "day_of_week", "month"]


class CyclicFeaturesAdder(BaseConfig, TimeSeriesTransform):
    """Transform that generates cyclic temporal features from datetime indices.

    Converts temporal information into sine and cosine components that preserve
    the periodic nature of time. This encoding ensures temporal boundaries
    (e.g., end of day/week/year) are properly connected.

    The features generated depend on the included_features configuration:
        - season: season_sine, season_cosine (based on day of year, 365.25 day cycle)
        - day_of_week: day_of_week_sine, day_of_week_cosine (based on day of week, 7 day cycle)
        - month: month_sine, month_cosine (based on month of year, 12 month cycle)
        - time_of_day: time_of_day_sine, time_of_day_cosine (based on seconds in day, 24 hour cycle)

    Example:
        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> from openstef_models.transforms.time_domain import (
        ...     CyclicFeaturesAdder,
        ... )
        >>>
        >>> # Create sample dataset
        >>> data = pd.DataFrame({
        ...     'load': [100, 120, 110]
        ... }, index=pd.date_range('2025-01-01 12:00:00', periods=3, freq='1h'))
        >>> dataset = TimeSeriesDataset(data, timedelta(hours=1))
        >>>
        >>> # Apply cyclic features with custom configuration
        >>> transform = CyclicFeaturesAdder(included_features=["season", "time_of_day"])
        >>> transformed = transform.transform(dataset)
        >>> result = transformed.data[['season_sine', 'time_of_day_sine']].round(3)
        >>> print(result.head(2))
                             season_sine  time_of_day_sine
        timestamp
        2025-01-01 12:00:00        0.017            -0.000
        2025-01-01 13:00:00        0.017            -0.259
    """

    included_features: list[CyclicFeatureName] = Field(
        default_factory=lambda: ["time_of_day", "season", "day_of_week", "month"],
        description="List of cyclic features to include in the transformation. "
        "Options are 'time_of_day', 'season', 'day_of_week', and 'month'.",
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

        if "day_of_week" in self.included_features:
            features.append(
                self._compute_cyclic_feature(
                    phase=data.index.day_of_week,
                    period=NUM_DAYS_IN_WEEK,
                    index=data.index,
                    name="day_of_week",
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

        if "time_of_day" in self.included_features:
            features.append(
                self._compute_cyclic_feature(
                    phase=data.index.hour * 3600 + data.index.minute * 60 + data.index.second,
                    period=NUM_SECONDS_IN_A_DAY,
                    index=data.index,
                    name="time_of_day",
                )
            )

        return TimeSeriesDataset(
            data=pd.concat(
                [data.data, *features],
                axis=1,
            ),
            sample_interval=data.sample_interval,
        )

    @override
    def features_added(self) -> list[str]:
        return [f"{feature}_sine" for feature in self.included_features] + [
            f"{feature}_cosine" for feature in self.included_features
        ]


__all__ = ["CyclicFeaturesAdder"]
