# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform that adds holiday features to time series data.

This module provides functionality to compute holiday features based on the
datetime index of time series datasets. It uses the `holidays` library to identify
public holidays for a specified country and adds binary features indicating
whether a date is a holiday or a specific holiday.
"""

import re
import unicodedata
from typing import Self, override

import holidays
import pandas as pd
from pydantic import Field
from pydantic_extra_types.country import CountryAlpha2

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.mixins import State
from openstef_core.transforms import TimeSeriesTransform


class HolidayFeatureAdder(BaseConfig, TimeSeriesTransform):
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
        >>> from openstef_models.transforms.time_domain import (
        ...     HolidayFeatureAdder
        ... )
        >>>
        >>> # Create sample dataset
        >>> data = pd.DataFrame({
        ...     'load': [100] * 5
        ... }, index=pd.date_range('2025-12-24', periods=5, freq='D'))
        >>> dataset = TimeSeriesDataset(data, sample_interval=timedelta(days=1))
        >>> transform = HolidayFeatureAdder(country_code='NL')
        >>> transformed_data = transform.transform(dataset).data
        >>> transformed_data["is_holiday"].tolist()
        [0, 1, 1, 0, 0]
        >>> transformed_data["is_christmas_day"].tolist()
        [0, 1, 0, 0, 0]
        >>> transformed_data["is_second_day_of_christmas"].tolist()
        [0, 0, 1, 0, 0]
    """

    country_code: CountryAlpha2 = Field(
        default=CountryAlpha2("NL"), description="Country code for holiday calculation."
    )

    @staticmethod
    def _sanitize_holiday_name(holiday_name: str) -> str:
        """Convert holiday name to a valid Python identifier for feature names.

        Args:
            holiday_name: Original holiday name from holidays library.

        Returns:
            Sanitized feature name (lowercase, underscores, no special chars).
        """
        # Normalize Unicode and convert to ASCII
        normalized = unicodedata.normalize("NFKD", holiday_name)
        ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
        # Replace non-alphanumeric with underscores and clean up
        return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", ascii_text.lower())).strip("_")

    def _get_holidays_dataframe(self, data: TimeSeriesDataset) -> pd.DataFrame:
        """Get holidays for the dataset's date range as a cleaned DataFrame.

        Args:
            data: Input time series dataset.

        Returns:
            DataFrame with columns: date, holiday_name, sanitized_name.
            Empty DataFrame if no holidays found.
            Sanitized names are pre-cleaned (no empty strings, duplicates handled).
        """
        years = range(int(data.index.year.min()), int(data.index.year.max()) + 1)  # type: ignore[reportUnknownMemberType]
        country_holidays = holidays.country_holidays(
            str(self.country_code), categories=["public"], years=years, language="en_US"
        )

        if not country_holidays:
            return pd.DataFrame(columns=["date", "holiday_name", "sanitized_name"])

        holidays_data = []  # type: ignore[var-annotated]
        for holiday_date, holiday_name in country_holidays.items():
            sanitized = self._sanitize_holiday_name(holiday_name)
            # Skip holidays that result in empty sanitized names
            if sanitized:
                holidays_data.append({  # type: ignore[misc]
                    "date": holiday_date,
                    "holiday_name": holiday_name,
                    "sanitized_name": sanitized,
                })

        return pd.DataFrame(holidays_data)

    @staticmethod
    def _create_general_holiday_feature(data: TimeSeriesDataset, holidays_df: pd.DataFrame) -> pd.Series:
        """Create the general is_holiday binary feature.

        Args:
            data: Input time series dataset.
            holidays_df: DataFrame with holiday information.

        Returns:
            Series with 1 for holidays, 0 for non-holidays.
        """
        # Create target DataFrame with dates from the time series
        target_df = pd.DataFrame({"date": data.index.date}, index=data.index)

        # Use merge for efficient date matching
        merged = target_df.merge(holidays_df[["date", "holiday_name"]], on="date", how="left")

        # Create binary feature and restore original index
        is_holiday = merged["holiday_name"].notna().astype(int)
        is_holiday.index = data.index  # Restore the DatetimeIndex
        is_holiday.name = "is_holiday"
        return is_holiday

    @staticmethod
    def _create_individual_features(data: TimeSeriesDataset, holidays_df: pd.DataFrame) -> pd.DataFrame:
        """Create individual holiday features using vectorized dummy variable creation.

        Args:
            data: Input time series dataset.
            holidays_df: DataFrame with holiday information (pre-cleaned).

        Returns:
            DataFrame with individual holiday binary features.
        """
        if holidays_df.empty:
            return pd.DataFrame(index=data.index)

        # Create target DataFrame with dates from the time series
        target_df = pd.DataFrame({"date": data.index.date}, index=data.index)

        # Use merge for efficient date matching
        merged = target_df.merge(holidays_df[["date", "sanitized_name"]], on="date", how="left")

        # Use get_dummies for efficient dummy variable creation
        holiday_names = merged["sanitized_name"].fillna("")  # type: ignore[call-overload]
        dummies = pd.get_dummies(holiday_names, prefix="is", dtype=int)

        # Remove empty string column (will always exist due to fillna)
        if "is_" in dummies.columns:
            dummies = dummies.drop("is_", axis=1)

        # Ensure correct index
        dummies.index = data.index
        return dummies

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        """Transform the dataset by adding holiday features.

        Args:
            data: Input time series dataset.

        Returns:
            Transformed dataset with holiday features added.
        """
        # Get holidays for the date range (pre-cleaned)
        holidays_df = self._get_holidays_dataframe(data)

        # Handle case with no holidays
        if holidays_df.empty:
            result_data = data.data.copy()
            result_data["is_holiday"] = 0
            return TimeSeriesDataset(data=result_data, sample_interval=data.sample_interval)

        # Create features
        general_feature = self._create_general_holiday_feature(data, holidays_df)
        individual_features = self._create_individual_features(data, holidays_df)

        # Combine with original data
        result_data = pd.concat([data.data, general_feature, individual_features], axis=1)
        return TimeSeriesDataset(data=result_data, sample_interval=data.sample_interval)

    @override
    def to_state(self) -> State:
        return self.model_dump(mode="json")

    @override
    def from_state(self, state: State) -> Self:
        return self.model_validate(state)

    @override
    def features_added(self) -> list[str]:
        country_holidays = holidays.country_holidays(
            str(self.country_code), categories=["public"], years=[2025], language="en_US"
        )

        return ["is_holiday"] + [
            f"is_{self._sanitize_holiday_name(name)}"
            for name in set(country_holidays.values())
            if self._sanitize_holiday_name(name)  # Exclude empty sanitized names
        ]
