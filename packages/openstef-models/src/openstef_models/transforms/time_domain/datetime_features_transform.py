# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for extracting datetime-based features from time series data.

This module provides functionality to compute datetime features that indicate
temporal patterns like weekdays, weekends, and other time-based characteristics
from the datetime index of time series datasets.
"""

from typing import override

import pandas as pd
from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.timeseries_transform import TimeSeriesTransform

MIN_WEEKEND_IDX: int = 5  # Saturday
SUNDAY_IDX: int = 6


class DatetimeFeaturesTransform(BaseConfig, TimeSeriesTransform):
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
        >>> from openstef_models.transforms.time_domain import (
        ...     DatetimeFeaturesTransform,
        ... )
        >>>
        >>> # Create sample dataset
        >>> data = pd.DataFrame({
        ...     'load': [100, 120, 110]
        ... }, index=pd.date_range('2025-01-01', periods=3, freq='D'))
        >>> dataset = TimeSeriesDataset(data, timedelta(days=1))
        >>> transform = DatetimeFeaturesTransform()
        >>> transformed_dataset = transform.transform(dataset)
        >>> sorted(transformed_dataset.data.columns.tolist())
        ['is_sunday', 'is_week_day', 'is_weekend_day', 'load', 'month_of_year', 'quarter_of_year']
        >>> transformed_dataset.data["is_week_day"].tolist()
        [1, 1, 1]
        >>> transformed_dataset.data["month_of_year"].tolist()
        [1, 1, 1]
    """

    onehot_encode: bool = Field(default=False, description="Whether to one-hot encode the features.")

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        features = [
            pd.DataFrame(
                data={
                    "is_week_day": (data.index.weekday < MIN_WEEKEND_IDX).astype(int),
                    "is_weekend_day": (data.index.weekday >= MIN_WEEKEND_IDX).astype(int),
                    "is_sunday": (data.index.weekday == SUNDAY_IDX).astype(int),
                },
                index=data.index,
            )
        ]

        if not self.onehot_encode:
            features.append(
                pd.DataFrame(
                    data={
                        "month_of_year": data.index.month,
                        "quarter_of_year": data.index.quarter,
                    },
                    index=data.index,
                )
            )
        else:
            month_dummies = pd.get_dummies(data.index.month, prefix="month", dtype=int)
            month_dummies.index = data.index
            quarter_dummies = pd.get_dummies(data.index.quarter, prefix="quarter", dtype=int)
            quarter_dummies.index = data.index
            features.extend([month_dummies, quarter_dummies])

        return TimeSeriesDataset(
            pd.concat([data.data, *features], axis=1),
            sample_interval=data.sample_interval,
        )


__all__ = ["DatetimeFeaturesTransform"]
