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
from datetime import UTC, datetime
from functools import lru_cache
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

    country_code: CountryAlpha2 = Field(description="Country code for holiday calculation.")

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        """Transform the dataset by adding holiday features.

        Args:
            data: Input time series dataset.

        Returns:
            Transformed dataset with holiday features added.
        """
        index = data.index
        # Get holidays for the date range (pre-cleaned)
        holiday_names = get_holiday_names(country_code=self.country_code)
        holidays_df = get_holidays(index=index, country=self.country_code)

        # Merge dates with holidays
        merged = (
            pd.Series(data=index.date, index=index)
            .to_frame("date")
            .merge(holidays_df[["date", "sanitized_name"]], on="date", how="left")
            .set_index(index)
        )

        # Create dummy variables for each holiday
        categorical = pd.Categorical(values=merged["sanitized_name"], categories=holiday_names)
        features_individual = pd.get_dummies(categorical, prefix="is", dtype=int).set_index(index)
        feature_general = merged["sanitized_name"].notna().astype(int).rename("is_holiday")

        # Combine with original data
        result_data = pd.concat([data.data, feature_general, features_individual], axis=1)
        return TimeSeriesDataset(data=result_data, sample_interval=data.sample_interval)

    @override
    def to_state(self) -> State:
        return self.model_dump(mode="json")

    @override
    def from_state(self, state: State) -> Self:
        return self.model_validate(state)

    @override
    def features_added(self) -> list[str]:
        return ["is_holiday", *[f"is_{name}" for name in get_holiday_names(self.country_code)]]


def sanitize_holiday_name(holiday_name: str) -> str:
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


@lru_cache(maxsize=5)
def get_holiday_names(country_code: CountryAlpha2) -> list[str]:
    """Get list of original holiday names for the country.

    Returns:
        List of unique holiday names in their original form.
    """
    # Years used to fetch holiday names. If new holidays are added later, this needs to be updated.
    current_year = datetime.now(tz=UTC).year
    reference_years = range(current_year - 5, current_year + 1)

    country_holidays = holidays.country_holidays(
        country=str(country_code), categories=["public"], years=reference_years, language="en_US"
    )
    return sorted([
        sanitize_holiday_name(holiday_name)
        for holiday_name in set(country_holidays.values())
        if sanitize_holiday_name(holiday_name)
    ])


def get_holidays(index: pd.DatetimeIndex, country: CountryAlpha2) -> pd.DataFrame:
    """Get holidays for the dataset's date range as a cleaned DataFrame.

    Args:
        index: DatetimeIndex of the dataset.
        country: Country code for holiday calculation.

    Returns:
        DataFrame with columns: date, holiday_name, sanitized_name.
        Empty DataFrame if no holidays found.
        Sanitized names are pre-cleaned (no empty strings, duplicates handled).
    """
    years = range(int(index.year.min()), int(index.year.max()) + 1)
    country_holidays = holidays.country_holidays(str(country), categories=["public"], years=years, language="en_US")
    all_country_holiday_names = get_holiday_names(country)

    if not country_holidays:
        return pd.DataFrame(columns=["date", "holiday_name", "sanitized_name"])

    return pd.DataFrame(
        data=[
            {
                "date": holiday_date,
                "holiday_name": holiday_name,
                "sanitized_name": sanitize_holiday_name(holiday_name),
            }
            for holiday_date, holiday_name in country_holidays.items()
            if sanitize_holiday_name(holiday_name) in all_country_holiday_names
        ]
    )
