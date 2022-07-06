# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd


def replace_repeated_values_with_nan(
    df: pd.DataFrame, max_length: int, column_name: str
) -> pd.DataFrame:
    """Replace sequentially repeated values with NaN.

    Args:
        df (pandas.DataFrame): Data with potential repeating values.
        max_length (int): Maximum length of sequence. Above are set to NaN.
        column_name (string): Column name of input dataframe with repeating values.

    Returns:
        pandas.DataFrame: Data, similar to df, with the desired values set to NaN.
    """
    data = df.copy(deep=True)
    sequentials = data[column_name].diff().ne(0).cumsum()
    grouped_sequentials_over_max = sequentials.groupby(sequentials).head(max_length)
    data.loc[~data.index.isin(grouped_sequentials_over_max.index), column_name] = np.nan
    return data
