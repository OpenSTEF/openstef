# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform that adds holiday features to time series data.

This module provides functionality to compute holiday features based on the
datetime index of time series datasets. It uses the `holidays` library to identify
public holidays for a specified country and adds binary features indicating
whether a date is a holiday or a specific holiday.
"""

from datetime import date
from typing import override

import holidays  # TODO: we should either add this to a group, or move the transforms to models module.
import pandas as pd

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.transforms import TimeSeriesTransform


class HolidayFeaturesTransform(BaseConfig, TimeSeriesTransform):
    """Transform that adds holiday features to time series data.

    Computes features that indicate whether a date is a holiday
    based on the specified country and categories.

    The features added are:
        - is_holiday: 1 if the day is a public holiday, 0 otherwise.
        - is_{holiday_name}: 1 if the day is a specific holiday, 0 otherwise.

    Example:
        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> from openstef_core.feature_engineering.temporal_transforms.holiday_features import (
        ...     HolidayFeatures
        ... )
        >>>
        >>> # Create sample dataset
        >>> data = pd.DataFrame({
        ...     'load': [100] * 5
        ... }, index=pd.date_range('2025-12-24', periods=5, freq='D'))
        >>> dataset = TimeSeriesDataset(data, sample_interval=timedelta(days=1))
        >>> transform = HolidayFeatures(country_code='NL')
        >>> transformed_data = transform.fit_transform(dataset).data
        >>> transformed_data["is_holiday"].tolist()
        [0, 1, 1, 0, 0]
        >>> transformed_data["is_christmas_day"].tolist()
        [0, 1, 0, 0, 0]
        >>> transformed_data["is_second_day_of_christmas"].tolist()
        [0, 0, 1, 0, 0]
        >>> transformed_data["is_easter_sunday"].tolist()
        [0, 0, 0, 0, 0]
    """

    country_code: str

    def __init__(self, **kwargs: str) -> None:
        """Initialize the HolidayFeatures transform.

        Args:
            **kwargs: Configuration parameters for the transform.
        """
        super().__init__(**kwargs)
        self._holiday_features: pd.DataFrame = pd.DataFrame()

    def _get_country_holidays(self, index: pd.DatetimeIndex) -> dict[date, str]:
        """Get all holidays for the years covered by the index.

        Args:
            index: DatetimeIndex to determine years for holiday lookup.

        Returns:
            Dictionary mapping dates to holiday names.
        """
        years = range(int(index.year.min()), int(index.year.max()) + 1)  # type: ignore[reportUnknownMemberType]
        return dict(holidays.country_holidays(self.country_code, categories=["public"], years=years, language="en_US"))

    @staticmethod
    def _sanitize_holiday_name(holiday_name: str) -> str:
        """Convert holiday name to a valid Python identifier for feature names.

        Args:
            holiday_name: Original holiday name from holidays library.

        Returns:
            Sanitized feature name (lowercase, underscores, no special chars).
        """
        # Convert to lowercase and replace spaces/special chars with underscores
        sanitized = holiday_name.lower()
        sanitized = "".join(c if c.isalnum() else "_" for c in sanitized)
        # Remove consecutive underscores and strip leading/trailing underscores
        return "_".join(filter(None, sanitized.split("_")))

    @staticmethod
    def _create_general_holiday_feature(index: pd.DatetimeIndex, country_holidays: dict[date, str]) -> pd.Series:
        """Create the general is_holiday feature.

        Args:
            index: DatetimeIndex for the dataset.
            country_holidays: Dictionary of holidays.

        Returns:
            Series with 1 for holidays, 0 for non-holidays.
        """
        holiday_dates: list[date] = list(country_holidays.keys())
        return pd.Series(index.date, index=index, name="is_holiday").isin(holiday_dates).astype(int)  # type: ignore[reportUnknownMemberType]

    def _create_individual_holiday_features(
        self, index: pd.DatetimeIndex, country_holidays: dict[date, str]
    ) -> pd.DataFrame:
        """Create individual features for each specific holiday.

        Args:
            index: DatetimeIndex for the dataset.
            country_holidays: Dictionary mapping dates to holiday names.

        Returns:
            DataFrame with individual holiday features.
        """
        if not country_holidays:
            return pd.DataFrame(index=index)

        # Get unique holiday names and create sanitized feature names
        unique_holidays = set(country_holidays.values())
        holiday_feature_mapping = {holiday: f"is_{self._sanitize_holiday_name(holiday)}" for holiday in unique_holidays}

        # Initialize all holiday features with zeros
        holiday_features = pd.DataFrame(data=0, index=index, columns=list(holiday_feature_mapping.values()))

        # Set 1 for dates that match each specific holiday
        for holiday_date, holiday_name in country_holidays.items():
            feature_name = holiday_feature_mapping[holiday_name]
            # Use .loc to handle cases where holiday_date might not be in index
            mask = index.date == holiday_date
            if mask.any():
                holiday_features.loc[mask, feature_name] = 1

        return holiday_features

    @override
    def fit(self, data: TimeSeriesDataset) -> None:
        """Fit the transform to the data by identifying holidays.

        Args:
            data: Time series dataset with DatetimeIndex.
        """
        country_holidays = self._get_country_holidays(data.index)

        general_feature = self._create_general_holiday_feature(data.index, country_holidays)

        individual_features = self._create_individual_holiday_features(data.index, country_holidays)

        # Combine all features
        self._holiday_features = pd.concat([general_feature, individual_features], axis=1)

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        """Transform the dataset by adding holiday features.

        Args:
            data: Time series dataset to transform.

        Returns:
            Transformed dataset with holiday features added.
        """
        return TimeSeriesDataset(
            data=pd.concat([data.data, self._holiday_features], axis=1),
            sample_interval=data.sample_interval,
        )
