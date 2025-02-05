# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from datetime import timedelta

import pandas as pd

from openstef.data_classes.prediction_job import PredictionJobDataClass


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
    rolling_window_load = data["load"].rolling(window=rolling_window)

    for aggregate_func in pj["rolling_aggregate_features"]:
        data[
            f"rolling_{aggregate_func.value}_load_{rolling_window}"
        ] = rolling_window_load.aggregate(aggregate_func.value)
    return data
