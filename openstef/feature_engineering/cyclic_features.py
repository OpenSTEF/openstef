# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

# Module for adding temporal cyclic features to time-based data for capturing seasonality and periodic patterns.
# Features include yearly, weekly, and monthly seasonality, as well as time-of-day periodicity.


import numpy as np
import pandas as pd

import structlog
import logging

from openstef.settings import Settings
from openstef import PROJECT_ROOT


structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(
        logging.getLevelName(Settings.log_level)
    )
)
logger = structlog.get_logger(__name__)

TERRESTRIAL_RADIATION_CSV_PATH: str = (
    PROJECT_ROOT / "openstef" / "data" / "NL_terrestrial_radiation.csv"
)
NUM_SECONDS_IN_A_DAY = 24 * 60 * 60


def add_daylight_terrestrial_feature(
    data: pd.DataFrame,
    path_to_terrestrial_radiation_csv: str = TERRESTRIAL_RADIATION_CSV_PATH,
) -> pd.DataFrame:
    """Add daylight terrestrial radiation feature to the input dataset. This function processes terrestrial radiation
    data and aligns it with the time indices of the input dataset. The terrestrial radiation data is normalized,
    interpolated, and merged with the main dataset to provide a feature representing terrestrial radiation.

    Args:
        data (pd.DataFrame):
            The input dataset containing a time-indexed DataFrame.
        path_to_terrestrial_radiation_csv (str):
            File path to the CSV file containing terrestrial radiation data. The CSV file
            should have a time-based index.

    Returns:
        pd.DataFrame:
            The input dataset with an added column for the terrestrial radiation feature.

    Notes:
        - The function assumes the input data and the terrestrial radiation data share
          the same time zone and frequency alignment.
        - The terrestrial radiation values are normalized using z-score normalization.

    """
    # Load the terrestrial radiation data
    terrestrial_radiation = pd.read_csv(path_to_terrestrial_radiation_csv, index_col=0)
    terrestrial_radiation.index = pd.to_datetime(terrestrial_radiation.index)

    # Align the index with the input data's year
    year_diff = data.index.min().year - terrestrial_radiation.index.min().year
    terrestrial_radiation.index += pd.DateOffset(years=year_diff)

    # Resample to 15-minute intervals, and interpolate missing values
    terrestrial_radiation = terrestrial_radiation.resample("15min").mean().interpolate()

    # Normalize the terrestrial radiation values using z-score normalization
    terrestrial_radiation = (
        terrestrial_radiation - terrestrial_radiation.mean(axis=0)
    ) / terrestrial_radiation.std(axis=0)
    terrestrial_radiation.columns = ["daylight_continuous"]

    # Make a copy of the DataFrame to avoid modifying the original
    data = data.copy()

    # Merge the terrestrial radiation data with the input dataset
    data = data.merge(
        terrestrial_radiation, left_index=True, right_index=True, how="left"
    )

    return data


def add_time_cyclic_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Adds time of the day features cyclically encoded using sine and cosine to the input data.

    Args:
        data: Dataframe indexed by datetime.

    Returns:
        DataFrame that is the same as input dataframe with extra columns for the added time of the day features.

    """
    # Ensure the index is a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Index should be a pandas DatetimeIndex")

    # Make a copy of the DataFrame to avoid modifying the original
    data = data.copy()

    second_of_the_day = (
        data.index.second + data.index.minute * 60 + data.index.hour * 60 * 60
    )
    period_of_the_day = 2 * np.pi * second_of_the_day / NUM_SECONDS_IN_A_DAY

    data["time0fday_sine"] = np.sin(period_of_the_day)
    data["time0fday_cosine"] = np.cos(period_of_the_day)

    return data


def add_seasonal_cyclic_features(
    data: pd.DataFrame, compute_features: list = None
) -> pd.DataFrame:
    """Adds cyclical features to capture seasonal and periodic patterns in time-based data.

    Args:
    - data (pd.DataFrame): DataFrame with a DatetimeIndex.
    - compute_features (list): Optional. List of features to compute. Options are:
      ['season', 'dayofweek', 'month']. Default is all features.

    Returns:
    - pd.DataFrame: DataFrame with added cyclical features.

    Example:
    >>> data = pd.DataFrame(index=pd.date_range(start='2023-01-01', periods=365, freq='D'))
    >>> data_with_features = add_cyclical_features(data)
    >>> print(data_with_features.head())

    """
    # Ensure the index is a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")

    # Make a copy of the DataFrame to avoid modifying the original
    data = data.copy()

    # Default to all features if none specified
    compute_features = compute_features or ["season", "dayofweek", "month"]

    days_in_year = 365.25  # Account for leap years

    # Add seasonality features (day of year)
    if "season" in compute_features:
        data["season_sine"] = np.sin(2 * np.pi * data.index.dayofyear / days_in_year)
        data["season_cosine"] = np.cos(2 * np.pi * data.index.dayofyear / days_in_year)

    # Add weekly features (day of the week)
    if "dayofweek" in compute_features:
        data["day0fweek_sine"] = np.sin(2 * np.pi * data.index.day_of_week / 7)
        data["day0fweek_cosine"] = np.cos(2 * np.pi * data.index.day_of_week / 7)

    # Add monthly features (month of the year)
    if "month" in compute_features:
        data["month_sine"] = np.sin(2 * np.pi * data.index.month / 12)
        data["month_cosine"] = np.cos(2 * np.pi * data.index.month / 12)

    return data
