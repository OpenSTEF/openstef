# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import numpy as np
import pandas as pd

NUM_SECONDS_IN_A_DAY = 24 * 60 * 60


def add_time_of_the_day_cyclic_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Adds time of the day features cyclically encoded using sine and cosine to the input data.

    Args:
        data: Dataframe indexed by datetime.

    Returns:
        DataFrame that is the same as input dataframe with extra columns for the added time of the day features.

    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Index should be a pandas DatetimeIndex")

    second_of_the_day = (
        data.index.second + data.index.minute * 60 + data.index.hour * 60 * 60
    )
    period_of_the_day = 2 * np.pi * second_of_the_day / NUM_SECONDS_IN_A_DAY

    data["time_of_day_sine"] = np.sin(period_of_the_day)
    data["time_of_day_cosine"] = np.cos(period_of_the_day)

    return data
