# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from datetime import timedelta

import pandas as pd

from openstef.data_classes.prediction_job import PredictionJobDataClass
from pydantic import TypeAdapter


def convert_timedelta_to_isoformat(td: timedelta) -> str:
    """
    Converts a timedelta to an ISO 8601 formatted period string.

    Args:
        td: timedelta object to convert.

    Returns:
        ISO 8601 formatted period string.
    """
    timedelta_adapter = TypeAdapter(timedelta)
    return timedelta_adapter.dump_python(td, mode="json")


def add_rolling_aggregate_features(
    data: pd.DataFrame,
    pj: PredictionJobDataClass,
    rolling_window: timedelta = timedelta(hours=24),
) -> pd.DataFrame:
    """
    Adds rolling aggregate features to the input dataframe.

    These features are calculated with an aggregation over a rolling window of the data.
    A list of requested features is used to determine whether to add the rolling features
    or not.

    Args:
        data: Input dataframe to which the rolling features will be added.
        pj: Prediction job data.
        rolling_window: Rolling window size for the aggregation.

    Returns:
        DataFrame with added rolling features.
    """
    # Ensure the index is a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")

    if "load" not in data.columns:
        raise ValueError("The DataFrame must contain a 'load' column.")

    # remove rows with NaN values in the load column for rolling window calculation
    rolling_window_load = data["load"].dropna().rolling(window=rolling_window)

    for aggregate_func in pj["rolling_aggregate_features"]:
        col_name = f"rolling_{aggregate_func.value}_load_{convert_timedelta_to_isoformat(rolling_window)}"
        data[col_name] = rolling_window_load.aggregate(aggregate_func.value)
        # Fill missing values with the last known value
        data[col_name] = data[col_name].ffill()

    return data
