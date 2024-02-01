# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd


def replace_repeated_values_with_nan(
    df: pd.DataFrame, threshold: int, column_name: str
) -> pd.DataFrame:
    """Replace sequentially repeated values with NaN.

    Args:
        df: Data with potential repeating values.
        threshold: The minimum number of squentially repeated values needed to trigger the replacement with NaN.
        column_name: Column name of input dataframe with repeating values.

    Returns:
        DataFrame, similar to df, with the desired values set to NaN.

    """
    data = df.copy()

    # Add a boolean column to mark sequential duplicates
    data["temp_is_duplicate"] = data[column_name].eq(data[column_name].shift(1))

    # Create an unique identifier for each sequence with the same value, so we can easily remove the correct sequences
    data["temp_repeated_group"] = (~data["temp_is_duplicate"]).cumsum()

    # Create mask of sequences larger than or equal to the threshold value
    mask = (
        data.groupby("temp_repeated_group")[column_name].transform("count") >= threshold
    )

    # Replace the masked values with NaN
    data.loc[mask, column_name] = np.nan

    # Drop temporary columns
    data = data.drop(["temp_is_duplicate", "temp_repeated_group"], axis=1)

    return data
