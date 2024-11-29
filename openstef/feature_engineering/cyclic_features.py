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

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(
        logging.getLevelName(Settings.log_level)
    )
)
logger = structlog.get_logger(__name__)


def add_seasonal_cyclic_features(
    data: pd.DataFrame, compute_features: list = None
) -> pd.DataFrame:
    """
    Adds cyclical features to capture seasonal and periodic patterns in time-based data.

    Cyclical features include:
    - Yearly seasonality (using day of the year)
    - Weekly seasonality (using day of the week)
    - Monthly seasonality (using month of the year)

    Parameters:
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
        data["season_sin"] = np.sin(2 * np.pi * data.index.dayofyear / days_in_year)
        data["season_cos"] = np.cos(2 * np.pi * data.index.dayofyear / days_in_year)

    # Add weekly features (day of the week)
    if "dayofweek" in compute_features:
        data["dayofweek_sin"] = np.sin(2 * np.pi * data.index.day_of_week / 7)
        data["dayofweek_cos"] = np.cos(2 * np.pi * data.index.day_of_week / 7)

    # Add monthly features (month of the year)
    if "month" in compute_features:
        data["month_sin"] = np.sin(2 * np.pi * data.index.month / 12)
        data["month_cos"] = np.cos(2 * np.pi * data.index.month / 12)

    return data


def add_time_cyclic_features(
    data: pd.DataFrame, frequency: str = None, period: int = 24
) -> pd.DataFrame:
    """
    Adds polar time features (sine and cosine) to capture periodic patterns based on the timestamp index.

    Parameters:
    - data (pd.DataFrame): Input DataFrame with a timestamp index.
    - frequency (str): Frequency of intervals (e.g., '15min', '1H'). Defaults to the frequency inferred from the DataFrame index.
    - period (int): Total number of hours in the periodic cycle (e.g., 24 for daily periodicity).

    Returns:
    - pd.DataFrame: Original DataFrame with added 'sin_time' and 'cos_time' columns.

    Raises:
    - ValueError: If the DataFrame index is not a DatetimeIndex.
    - ValueError: If `period` is less than or equal to 0.
    """
    # Validate input DataFrame index
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")

    # Infer frequency from the DataFrame index if not provided
    if frequency is None:
        inferred_freq = pd.infer_freq(data.index)
        if inferred_freq is None:
            datetime_index = data.index
            time_diffs = datetime_index.to_series().diff().dropna()
            inferred_freq = time_diffs.mode()[0]
            logging.warning(
                f"Could not infer the timestamp frequency from the index. "
                f"Using the estimated frequency of '{inferred_freq}' based on the most common time difference between timestamps. "
                f"Consider providing the 'frequency' parameter for more accurate results."
            )
            frequency = inferred_freq

    # Validate period
    if period <= 0:
        raise ValueError("The 'period' parameter must be greater than 0.")

    # Validate frequency
    try:
        base_interval_minutes = pd.Timedelta(frequency).seconds // 60
    except ValueError:
        raise ValueError(
            f"Invalid frequency string: '{frequency}'. Ensure it follows pandas Timedelta conventions (e.g., '15min', '1H')."
        )

    # Make a copy of the DataFrame to avoid modifying the original
    data = data.copy()

    # Calculate time in minutes since the start of the day
    time_in_minutes = data.index.hour * 60 + data.index.minute

    # Calculate the time interval within the periodic cycle
    data["time_interval"] = time_in_minutes / base_interval_minutes

    # Calculate total intervals in one cycle
    intervals_per_cycle = (period * 60) / base_interval_minutes

    # Add sine and cosine features
    data["sin_time"] = np.sin(2 * np.pi * data["time_interval"] / intervals_per_cycle)
    data["cos_time"] = np.cos(2 * np.pi * data["time_interval"] / intervals_per_cycle)

    # Drop intermediate column for cleaner output
    return data.drop(columns=["time_interval"])
