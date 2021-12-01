# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import numpy as np


def replace_repeated_values_with_nan(df, max_length, column_name):
    """Replace repeated values with NaN.

        Replace repeated values (sequentially repeating values), which repeat longer
        than a set max_length (in data points) with NaNs.

    Args:
        df (pandas.DataFrame): Data from which you would like to set repeating values to nan
        max_length (int): If a value repeats more often, sequentially, than this value, all those points are set to NaN
        column_name (string): the pandas dataframe column name of the column you want to process

    Rrturns:
        pandas.DataFrame: data, similar to df, with the desired values set to NaN.
    """
    data = df.copy(deep=True)
    indices = []
    old_value = -1000000000000
    value = 0
    for index, r in data.iterrows():
        value = r[column_name]
        if value == old_value:
            indices.append(index)
        elif (value != old_value) & (len(indices) > max_length):
            indices = indices[max_length:]
            data.at[indices, column_name] = np.nan
            indices = []
            indices.append(index)
        elif (value != old_value) & (len(indices) <= max_length):
            indices = []
            indices.append(index)
        old_value = value
    if len(indices) > max_length:
        data.at[indices, column_name] = np.nan
    return data


def replace_invalid_data(df, suspicious_moments):
    """Function that detects invalid data using the nonzero_flatliner function and converts the output to NaN values.

    Input:
    - df: pd.dataFrame(index=DatetimeIndex, columns = [load1, ..., loadN]). Load_corrections should be indicated by 'LC_'
    - suspicious_moments (pd.dataframe): output of function nonzero_flatliner in new variable


    return:
    - pd.DataFrame without invalid data (converted to NaN values)"""
    if suspicious_moments is not None:
        for index, row in suspicious_moments.iterrows():
            df[
                (row[0]) : (row[1] - timedelta(seconds=1))
            ] = np.nan  # subtract 1 second to make it exclusive
        return df
    return df
