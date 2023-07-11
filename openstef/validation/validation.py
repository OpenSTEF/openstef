# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from datetime import timedelta
from typing import Union

import numpy as np
import pandas as pd
import structlog

from openstef.preprocessing.preprocessing import replace_repeated_values_with_nan
from openstef.model.regressors.regressor import OpenstfRegressor


def validate(
    pj_id: Union[int, str],
    data: pd.DataFrame,
    flatliner_threshold: Union[int, None],
) -> pd.DataFrame:
    """Validate prediction job and timeseries data.

    Steps:
    1. Replace repeated values for longer than flatliner_threshold with NaN
    # TODO: The function description suggests it
    'validates' the PJ and Data, but is appears to 'just' replace repeated observations with NaN.

    Args:
        pj_id: ind/str, used to identify log statements
        data: pd.DataFrame where the first column should be the target. index=datetimeIndex
        flatliner_threshold: int of max repetitions considered a flatline.
            if None, the validation is effectively skipped

    Returns:
        Dataframe where repeated values are set to None

    """
    logger = structlog.get_logger(__name__)

    # Check if DataFrame has datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Input dataframe does not have a datetime index.")

    if flatliner_threshold is None:
        logger.info("Skipping validation of input data", pj_id=pj_id)
        return data

    # Drop 'false' measurements. e.g. where load appears to be constant.
    data = replace_repeated_values_with_nan(
        data, max_length=flatliner_threshold, column_name=data.columns[0]
    )
    num_repeated_values = len(data) - len(data.iloc[:, 0].dropna())
    if num_repeated_values > 0:
        frac_const_load_values = num_repeated_values / len(data.index)

        logger.info(
            f"Found {num_repeated_values} values of constant load (repeated values),"
            " converted to NaN value.",
            cleansing_step="repeated_values",
            pj_id=pj_id,
            num_values=num_repeated_values,
            frac_values=frac_const_load_values,
        )

    return data


def drop_target_na(data: pd.DataFrame) -> pd.DataFrame:
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


def is_data_sufficient(
    data: pd.DataFrame,
    completeness_threshold: float,
    minimal_table_length: int,
    model: OpenstfRegressor = None,
) -> bool:
    """Check if enough data is left after validation and cleaning to continue with model training.

    Args:
        data: pd.DataFrame() with cleaned input data.
        model: model which contains all information regarding trained model
        completeness_threshold: float with threshold for completeness:
            1 for fully complete, 0 for anything could be missing.
        minimal_table_length: int with minimal table length (in rows)

    Returns:
        True if amount of data is sufficient, False otherwise.

    """
    if model is None:
        weights = None  # Remove horizon & load column
    else:
        weights = model.feature_importance_dataframe

    logger = structlog.get_logger(__name__)
    # Set output variable
    is_sufficient = True

    # Calculate completeness
    completeness = calc_completeness_features(
        data, weights, time_delayed=True, homogenise=False
    )
    table_length = data.shape[0]

    # Check if completeness is up to the standards
    if completeness < completeness_threshold:
        logger.warning(
            "Input data is not sufficient, completeness too low",
            completeness=completeness,
            completeness_threshold=completeness_threshold,
        )
        is_sufficient = False

    # Check if absolute amount of rows is sufficient
    if table_length < minimal_table_length:
        logger.warning(
            "Input data is not sufficient, table length too short",
            table_length=table_length,
            table_length_threshold=minimal_table_length,
        )
        is_sufficient = False

    return is_sufficient


def calc_completeness_features(
    df: pd.DataFrame,
    weights: pd.DataFrame,
    time_delayed: bool = False,
    homogenise: bool = True,
) -> float:
    """Calculate the (weighted) completeness of a dataframe.

    NOTE: NA values count as incomplete

    Args:
        df: Dataframe with a datetimeIndex index
        weights: Array-compatible with size equal to columns of df
            (excl. load&horizon), used to weight the completeness of each column
        time_delayed: Should there be a correction for T-x columns
        homogenise: Should the index be resampled to median time delta -
            only available for DatetimeIndex

    Returns:
        Fraction of completeness

    """
    df_copy = df.copy(
        deep=True
    )  # Make deep copy to maintain original dataframe in pipeline

    # Remove load and horizon from data_with_features dataframe to make sure columns are equal
    if "load" in df_copy:
        df_copy.drop("load", inplace=True, axis=1)
    if "horizon" in df_copy:
        df_copy.drop("horizon", inplace=True, axis=1)

    if weights is None:
        weights = np.array([1] * ((len(df_copy.columns))))

    length_weights = len(weights)
    length_features = len(df_copy.columns)

    # Returns the list
    if type(weights) != np.ndarray:
        list_features = weights.index.tolist()
        df_copy = df_copy[list_features]  # Reorder the df to match weights index (list)
        weights = weights.weight

    if length_weights != length_features:
        raise ValueError(
            "Input data is not sufficient, number of features used in model is unequal to amount of columns in data"
        )
    completeness_per_column_dataframe = calc_completeness_dataframe(
        df_copy, time_delayed, homogenise
    )

    # scale to weights and normalize
    completeness = (completeness_per_column_dataframe * weights).sum() / weights.sum()

    return completeness


def find_nonzero_flatliner(
    df: pd.DataFrame, threshold: int = None
) -> Union[pd.DataFrame, None]:
    """Function that detects a stationflatliner and returns a list of datetimes.

    Args:
        df: Example pd.dataFrame(index=DatetimeIndex, columns = [load1, ..., loadN]).
            Load_corrections should be indicated by 'LC_'
        threshold: after how many timesteps should the function detect a flatliner.
            If None, the check is not executed

    Returns:
        Flatline moments or None

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
    flatliner_window: timedelta,
    flatliner_load_threshold: float,
) -> pd.DataFrame or None:
    """Detect a zero value where the load is not compensated by the other trafo's of the station.

    If zero value is at start or end, ignore that block.

    Args:
        df: DataFrame such as pd.dataFrame(index=DatetimeIndex, columns = [load1, ..., loadN]). Load_corrections should be indicated by 'LC_'
        threshold: after how many hours should the function detect a flatliner.
        flatliner_window: for how many hours before the zero-value should the mean load be calculated.
        flatliner_load_threshold: how big may the difference be between the total station load
            before and during the zero-value(s).

    Return:
        DataFrame of timestamps, or None if none

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
                    df[candidate.from_time - flatliner_window : candidate.from_time]
                    .sum(axis=1)
                    .mean()
                )
                mean_full_timespan = (
                    df[candidate.from_time : candidate.from_time + flatliner_window]
                    .sum(axis=1)
                    .mean()
                )

                # Compare load and detect zero-value flatliner
                if (
                    np.abs(
                        (mean_load_before - mean_full_timespan)
                        / np.max([np.abs(mean_load_before), np.abs(mean_full_timespan)])
                    )
                    > flatliner_load_threshold
                ):
                    transposed_candidate = candidate.to_frame().T
                    non_compensated_df = pd.concat(
                        [non_compensated_df, transposed_candidate]
                    )
                    print(
                        "Found a non-compensated zero value:",
                        candidate.to_string(index=False),
                    )

            # after checking all candidates for a single trafo, add moments to result_df
            result_df = pd.concat([result_df, non_compensated_df])

    if len(result_df) == 0:
        result_df = None
    return result_df


def check_data_for_each_trafo(df: pd.DataFrame, col: pd.Series) -> bool:
    """Function that detects if each column contains zero-values at all, only zero-values and NaN values.

    Args:
        df: DataFrama such as pd.dataFrame(index=DatetimeIndex, columns = [load1, ..., loadN]).
            Load_corrections should be indicated by 'LC_'
        col: column of pd.dataFrame

    Returns:
        False if column contains above specified or True if not

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


def calc_completeness_dataframe(
    df: pd.DataFrame,
    time_delayed: bool = False,
    homogenise: bool = True,
) -> pd.DataFrame:
    """Calculate the completeness of each column in dataframe.

    NOTE: NA values count as incomplete

    Args:
        df: Dataframe with a datetimeIndex index
        time_delayed: Should there be a correction for T-x columns
        homogenise: Should the index be resampled to median time delta -
            only available for DatetimeIndex

    Returns:
        Dataframe with fraction of completeness per column

    """
    logger = structlog.get_logger(__name__)

    if homogenise and isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:

        median_timediff = int(
            df.reset_index().iloc[:, 0].diff().median().total_seconds() / 60.0
        )
        df = df.resample("{:d}T".format(median_timediff)).mean()

    if time_delayed is False:
        # Calculate completeness
        # Completeness per column
        completeness_per_column_dataframe = df.count() / len(df)

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
            if non_na_count[col] > value:
                logger.warning(
                    "The provided input data (features) contains more values than is to be expected from analysis",
                    feature=col,
                    number_non_na=non_na_count[col],
                    expected_numbers_timedelayed=value,
                )

        # Correct for APX being only expected to be available up to 24h
        if "APX" in non_na_count.index:
            non_na_count["APX"] += max([len(df) - 96, 0])

        completeness_per_column_dataframe = non_na_count / (len(df))

    return completeness_per_column_dataframe
