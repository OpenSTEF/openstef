# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta
from typing import Union

import numpy as np
import pandas as pd
import structlog

from openstef.preprocessing.preprocessing import (
    replace_invalid_data,
    replace_repeated_values_with_nan,
)

# TODO make this config more central
# Set thresholds
COMPLETENESS_THRESHOLD = 0.5
MINIMAL_TABLE_LENGTH = 100

FLATLINER_TRESHOLD = 24


def validate(
    pj_id: Union[int, str],
    data: pd.DataFrame,
    flatliner_threshold: int = FLATLINER_TRESHOLD,
) -> pd.DataFrame:
    """Validate prediction job and timeseries data."""
    logger = structlog.get_logger(__name__)
    # Drop 'false' measurements. e.g. where load appears to be constant.
    data = replace_repeated_values_with_nan(
        data, max_length=flatliner_threshold, column_name=data.columns[0]
    )
    num_repeated_values = len(data) - len(data.iloc[:, 0].dropna())
    if num_repeated_values > 0:
        frac_const_load_values = num_repeated_values / len(data.index)

        logger.info(
            f"Found {num_repeated_values} values of constant load (repeated values), converted to NaN value.",
            cleansing_step="repeated_values",
            pj_id=pj_id,
            num_values=num_repeated_values,
            frac_values=frac_const_load_values,
        )

    # Check for repeated load observations due to invalid measurements
    nonzero_flatliners = find_nonzero_flatliner(data, threshold=flatliner_threshold)
    if nonzero_flatliners is not None:
        # Covert repeated load observations to NaN values
        data = replace_invalid_data(data, nonzero_flatliners)
        # Calculate number of NaN values
        # TODO should this not be part of the replace_invalid_data function?
        num_nan_values = sum([True for i, row in data.iterrows() if all(row.isnull())])
        frac_nan_values = num_nan_values / len(data.index)

        logger.info(
            f"Found {num_nan_values} nonzero flatliner data points, converted to NaN value.",
            cleansing_step="nonzero_flatliner_data_points",
            pj_id=pj_id,
            num_values=num_nan_values,
            frac_values=frac_nan_values,
        )

    return data


def clean(data: pd.DataFrame) -> pd.DataFrame:
    logger = structlog.get_logger(__name__)
    len_original = len(data)
    # Remove where load is NA, NaN features are preserved
    data = data.loc[np.isnan(data.iloc[:, 0]) != True, :]  # noqa E712
    num_removed_values = len_original - len(data)
    if num_removed_values > 0:
        logger.info(
            f"Removed {num_removed_values} NaN values",
            num_removed_values=num_removed_values,
        )

    return data


def is_data_sufficient(data: pd.DataFrame) -> bool:
    """Check if enough data is left after validation and cleaning to continue
        with model training.

    Args:
        data: pd.DataFrame() with cleaned input data.

    Returns:
        (bool): True if amount of data is sufficient, False otherwise.

    """
    logger = structlog.get_logger(__name__)
    # Set output variable
    is_sufficient = True

    # Calculate completeness
    completeness = calc_completeness(data, time_delayed=True, homogenise=False)
    table_length = data.shape[0]

    # Check if completeness is up to the standards
    if completeness < COMPLETENESS_THRESHOLD:
        logger.warning(
            "Input data is not sufficient, completeness too low",
            completeness=completeness,
            completeness_threshold=COMPLETENESS_THRESHOLD,
        )
        is_sufficient = False

    # Check if absolute amount of rows is sufficient
    if table_length < MINIMAL_TABLE_LENGTH:
        logger.warning(
            "Input data is not sufficient, table length too short",
            table_length=table_length,
            table_length_threshold=MINIMAL_TABLE_LENGTH,
        )
        is_sufficient = False

    return is_sufficient


def calc_completeness(
    df: pd.DataFrame,
    weights: np.array = None,
    time_delayed: bool = False,
    homogenise: bool = True,
) -> float:
    """Calculate the (weighted) completeness of a dataframe.

    NOTE: NA values count as incomplete

    Args:
        df (pd.DataFrame): Dataframe with a datetimeIndex index
        weights: Array-compatible with size equal to columns of df.
            used to weight the completeness of each column
        time_delayed (bool): Should there be a correction for T-x columns
        homogenise (bool): Should the index be resampled to median time delta -
            only available for DatetimeIndex

    Returns:
        float: Completeness
    """

    if weights is None:
        weights = np.array([1] * len(df.columns))
    weights = np.array(weights)

    if homogenise and isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:

        median_timediff = int(
            df.reset_index().iloc[:, 0].diff().median().total_seconds() / 60.0
        )
        df = df.resample("{:d}T".format(median_timediff)).mean()

    if time_delayed is False:
        # Calculate completeness
        # Completeness per column
        completeness_per_column = df.count() / len(df)

    # if timeDelayed is True, we correct that time-delayed columns
    # also in the best case will have NA values. E.g. T-2d is not available
    # for times ahead of more than 2 days
    elif time_delayed:
        # assume 15 minute forecast resolution
        # timecols: {delay:number of points expected to be missing}
        # number of points expected to be missing = numberOfPointsUpToTwoDaysAhead - numberOfPointsAvailable
        timecols = {
            x: len(df) - eval(x[2:].replace("min", "/60").replace("d", "*24.0")) / 0.25
            for x in df.columns
            if x[:2] == "T-"
        }

        non_na_count = df.count()
        for col, value in timecols.items():
            if value >= 0:
                non_na_count[col] += value

        # Correct for APX being only expected to be available up to 24h
        if "APX" in non_na_count.index:
            non_na_count["APX"] += max([len(df) - 96, 0])

        completeness_per_column = non_na_count / len(df)

    # scale to weights and normalize
    completeness = (completeness_per_column * weights).sum() / weights.sum()

    return completeness


def find_nonzero_flatliner(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Function that detects a stationflatliner and returns a list of datetimes.

    Args:
        df: pd.dataFrame(index=DatetimeIndex, columns = [load1, ..., loadN]).
            Load_corrections should be indicated by 'LC_'
        threshold: after how many timesteps should the function detect a flatliner.

    Returns:
    # TODO: function returns None or a DataFrame
        list: flatline moments

    TODO: a lot of the logic of this function can be improved using: mnts.label
    ```
    import scipy.ndimage.measurements as mnts
    mnts.label
    ```
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

    # If a zero-value is at start or end of df, remove from last_* list
    if len(flatliner_total) > 0:
        if flatliner_total.iloc[0]:
            last_zeros = last_zeros[1:]
        if flatliner_total.iloc[-1]:
            first_zeros = first_zeros[:-1]

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


def find_zero_flatliner(
    df: pd.DataFrame,
    threshold: float,
    window: timedelta = timedelta(minutes=30),
    load_threshold: float = 0.3,
) -> pd.DataFrame or None:
    """Function that detects a zero value where the load is not compensated by the other trafo's of the station.
    If zero value is at start or end, ignore that block.

    Input:
    - df: pd.dataFrame(index=DatetimeIndex, columns = [load1, ..., loadN]). Load_corrections should be indicated by 'LC_'
    - threshold (float): after how many hours should the function detect a flatliner.
    - window (timedelta object): for how many hours before the zero-value should the mean load be calculated.
    - load_threshold (fraction): how big may the difference be between the total station load
    before and during the zero-value(s).

    return:
    - pd.DataFrame of timestamps, or None if none

    TODO: a lot of the logic of this function can be improved using: mnts.label
    ```
    import scipy.ndimage.measurements as mnts
    mnts.label
    ```
    """
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

            # If a zero-value is at start or end of df, remove from last_* list
            if len(zero_values) > 0:
                if zero_values.iloc[0]:
                    last_zeros = last_zeros[1:]
                if zero_values.iloc[-1]:
                    first_zeros = first_zeros[:-1]

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


def check_data_for_each_trafo(df: pd.DataFrame, col: pd.Series) -> bool:
    """Function that detects if each column contains zero-values at all, only
        zero-values and NaN values.

    Args:
        df: pd.dataFrame(index=DatetimeIndex, columns = [load1, ..., loadN]).
            Load_corrections should be indicated by 'LC_'
        col: column of pd.dataFrame

    Returns:
        bool: False if column contains above specified or True if not
    """
    if df is None:
        return False

    # Check for each column the data on the following: (Skipping if true)
    # Check if there a zero-values at all
    if (df[col] != 0).all(axis=0):
        print(f"No zero values found - at all at trafo {col}, skipping column")
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
