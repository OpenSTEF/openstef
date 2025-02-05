# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import pandas as pd


def add_rolling_aggregate_features(
    data: pd.DataFrame, rolling_window: str = "24h"
) -> pd.DataFrame:
    """
    Adds rolling aggregate features to the input dataframe.

    These features are calculated with an aggregation over a rolling window of the data.
    A list of requested features is used to determine whether to add the rolling features
    or not.

    Args:
        data: Input dataframe to which the rolling features will be added.
        rolling_window: Rolling window size in str format following
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

    Returns:
        DataFrame with added rolling features.
    """
    # Ensure the index is a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")

    if "load" not in data.columns:
        raise ValueError("The DataFrame must contain a 'load' column.")
    rolling_window_load = data["load"].rolling(window=rolling_window)
    data[f"rolling_median_load_{rolling_window}"] = rolling_window_load.median()
    data[f"rolling_max_load_{rolling_window}"] = rolling_window_load.max()
    data[f"rolling_min_load_{rolling_window}"] = rolling_window_load.min()
    return data
