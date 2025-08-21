# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.transforms.base import TimeSeriesTransform

NUM_DAYS_IN_YEAR = 365.25
NUM_DAYS_IN_WEEK = 7
NUM_MONTHS_IN_YEAR = 12


class CyclicFeatures(TimeSeriesTransform):
    """A class to handle cyclic features, such as hours of the day or days of the week.
    It provides methods to convert these features into sine and cosine components.
    """

    cyclic_features: dict[str, pd.Series] = {}

    @staticmethod
    def _compute_sine(phase: pd.Index, period: float) -> pd.Series:
        """Compute the sine component for cyclic features."""
        return pd.Series(np.sin(2 * np.pi * phase / period))

    @staticmethod
    def _compute_cosine(phase: pd.Index, period: float) -> pd.Series:
        """Compute the cosine component for cyclic features."""
        return pd.Series(np.cos(2 * np.pi * phase / period))

    def _compute_seasonal_feature(self, data: TimeSeriesDataset) -> dict[str, pd.Series]:
        return {
            "season_sine": self._compute_sine(data.index.dayofyear, NUM_DAYS_IN_YEAR),
            "season_cosine": self._compute_cosine(data.index.dayofyear, NUM_DAYS_IN_YEAR),
        }

    def _compute_weekday_feature(self, data: TimeSeriesDataset) -> dict[str, pd.Series]:
        return {
            "day0fweek_sine": self._compute_sine(data.index.day_of_week, NUM_DAYS_IN_WEEK),
            "day0fweek_cosine": self._compute_cosine(data.index.day_of_week, NUM_DAYS_IN_WEEK),
        }

    def _compute_monthly_feature(self, data: TimeSeriesDataset) -> dict[str, pd.Series]:
        return {
            "month_sine": self._compute_sine(data.index.month, NUM_MONTHS_IN_YEAR),
            "month_cosine": self._compute_cosine(data.index.month, NUM_MONTHS_IN_YEAR),
        }

    def fit(self, data: TimeSeriesDataset) -> None:
        """Fit the transform to the input time series data."""
        self.cyclic_features = {
            **self._compute_seasonal_feature(data),
            **self._compute_weekday_feature(data),
            **self._compute_monthly_feature(data),
        }

    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        """Transform the input time series data by adding cyclic features."""
        for feature_name, feature_data in self.cyclic_features.items():
            data.add_feature(feature_name, feature_data)
        return data
