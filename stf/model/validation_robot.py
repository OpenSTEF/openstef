# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

# -*- coding: utf-8 -*-
from datetime import timedelta

import numpy as np
import pandas as pd


def nonzero_flatliner(df, threshold):
    """Function that detects a stationflatliner and returns a list of datetimes.

    Args:
        df: pd.dataFrame(index=DatetimeIndex, columns = [load1, ..., loadN]).
            Load_corrections should be indicated by 'LC_'
        threshold: after how many timesteps should the function detect a flatliner.

    Returns:
    # TODO: function returns None or a DataFrame
        list: flatline moments
    """

    if len(df) == 0:
        return

    # remove load corrections
    df = df.loc[:, ~df.columns.str.startswith("LC_")]

    # Remove moments when total load is 0
    df = df[(df.T != 0).any()]

    # Add columns with diff
    diff_columns = ["diff_" + x for x in df.columns]
    df[diff_columns] = df.diff().abs()
    # We are looking for when total station is a flatliner, so all
    # Select where all diff columns are zero
    flatliner_total = (df[diff_columns] == 0).mean(axis=1) == 1
    # Find all first and last occurences of 0
    first_zeros = flatliner_total[:-1][
        (flatliner_total.iloc[:-1] == 0).values * (flatliner_total.iloc[1:] != 0).values
    ]
    last_zeros = flatliner_total[1:][
        (flatliner_total.iloc[:-1] != 0).values * (flatliner_total.iloc[1:] == 0).values
    ]
    last_zeros.index = last_zeros.index - pd.Timedelta("15m")

    # Give as output from:to
    interval_df = pd.DataFrame(
        {
            "from_time": first_zeros.index,
            "to_time": last_zeros.index,
            "duration_h": last_zeros.index - first_zeros.index,
        },
        index=range(len(first_zeros)),
    )

    # Only keep periods which exceed threshold
    interval_df = interval_df[interval_df["duration_h"] >= timedelta(hours=threshold)]
    if len(interval_df) == 0:
        interval_df = None
    return interval_df


def check_data_for_each_trafo(df, col):
    """Function that detects if each column contains zero-values at all, only zero-values and NaN values.

    Args:
        df: pd.dataFrame(index=DatetimeIndex, columns = [load1, ..., loadN]).
            Load_corrections should be indicated by 'LC_'
        col: column of pd.dataFrame

    Returns:
        bool: False if column contains above specified or True if not"""
    if df is not None:
        # Check for each column the data on the following: (Skipping if true)
        # Check if there a zero-values at all
        if (df[col] != 0).all(axis=0):
            print(
                "No zero values found - at all at trafo {a}, skipping column".format(
                    a=col
                )
            )
            return False
        # Check if all values are zero in column
        elif (df[col] == 0).all(axis=0):
            print("Load at trafo {a} is zero, skipping column".format(a=col))
            return False
        # Check if all values are NaN in the column
        elif np.all(pd.isna(col)):
            print("Load at trafo {a} is missing, skipping column".format(a=col))
            return False
        return True
    else:
        return False


def zero_flatliner(df, threshold, window=timedelta(minutes=30), load_threshold=0.3):
    """Function that detects a zero value where the load is not compensated by the other trafo's of the station.

    Input:
    - df: pd.dataFrame(index=DatetimeIndex, columns = [load1, ..., loadN]). Load_corrections should be indicated by 'LC_'
    - threshold (float): after how many hours should the function detect a flatliner.
    - window (timedelta object): for how many hours before the zero-value should the mean load be calculated.
    - load_threshold (fraction): how big may the difference be between the total station load
    before and during the zero-value(s).

    return:
    - pd.DataFrame of timestamps, or None if none"""
    result_df = pd.DataFrame()

    for col in df.columns:
        # Check for each trafo if it contains zero-values or NaN values
        if check_data_for_each_trafo(df, col):
            # Copy column in order to manipulate data
            df_col = df[col].copy()
            # Create list of zero_values to check length
            zero_values = df_col == 0
            last_zeros = zero_values[:-1][
                (zero_values.iloc[:-1] != 0).values * (zero_values.iloc[1:] == 0).values
            ]
            first_zeros = zero_values[1:][
                (zero_values.iloc[:-1] == 0).values * (zero_values.iloc[1:] != 0).values
            ]
            interval_df = pd.DataFrame(
                {
                    "from_time": first_zeros.index,
                    "to_time": last_zeros.index,
                    "duration_h": last_zeros.index - first_zeros.index,
                },
                index=range(len(first_zeros)),
            )

            # Keep dataframe with zero_value > threshold
            interval_df = interval_df[
                interval_df["duration_h"] >= timedelta(hours=threshold)
            ]

            # Check if there are missing values in the data
            num_nan = np.sum(pd.isna(df_col))
            if np.sum(num_nan) > 0:
                print(
                    "Found {a} missing values at trafo {b}, dropping NaN values".format(
                        a=num_nan, b=col
                    )
                )
                df_col = df_col.dropna()

            non_compensated_df = pd.DataFrame()
            print(
                "Found {b} suspicious moments were load is a zero at trafo {a}".format(
                    a=col, b=len(interval_df)
                )
            )
            # Forloop all moments of zero_values, check if delta of to_zero_trafo is compesated by other trafos
            for i, candidate in interval_df.iterrows():
                # Calculate mean load before and after zero-moment
                mean_load_before = (
                    df[candidate.from_time - window : candidate.from_time]
                    .sum(axis=1)
                    .mean()
                )
                mean_full_timespan = (
                    df[candidate.from_time : candidate.from_time + window]
                    .sum(axis=1)
                    .mean()
                )

                # Compare load and detect zero-value flatliner
                if (
                    np.abs(
                        (mean_load_before - mean_full_timespan)
                        / np.max([np.abs(mean_load_before), np.abs(mean_full_timespan)])
                    )
                    > load_threshold
                ):
                    non_compensated_df = non_compensated_df.append(candidate)
                    print(
                        "Found a non-compensated zero value:",
                        candidate.to_string(index=False),
                    )

            # after checking all candidates for a single trafo, add moments to result_df
            result_df = result_df.append(non_compensated_df)

    if len(result_df) == 0:
        result_df = None
    return result_df


def replace_invalid_data(df, suspicious_moments):
    """Function that detects invalid data using the nonzero_flatliner function and converts the output to NaN values.

    Input:
    - df: pd.dataFrame(index=DatetimeIndex, columns = [load1, ..., loadN]). Load_corrections should be indicated by 'LC_'
    - suspicious_moments (pd.dataframe): output of function nonzero_flatliner in new variable


    return:
    - pd.DataFrame without invalid data (converted to NaN values)"""
    if suspicious_moments is not None:
        for index, row in suspicious_moments.iterrows():
            df[(row[0]) : (row[1])] = np.nan
        return df
    return df
