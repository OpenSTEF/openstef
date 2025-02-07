# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import logging
import math
from datetime import datetime, timedelta, UTC
from typing import Union

import numpy as np
import pandas as pd
import structlog

from openstef.exceptions import InputDataOngoingZeroFlatlinerError
from openstef.model.regressors.regressor import OpenstfRegressor
from openstef.preprocessing.preprocessing import replace_repeated_values_with_nan
from openstef.settings import Settings


def validate(
    pj_id: Union[int, str],
    data: pd.DataFrame,
    flatliner_threshold_minutes: Union[int, None],
    resolution_minutes: int,
) -> pd.DataFrame:
    """Validate prediction job and timeseries data.

    Steps:
    1. Check if input dataframe has a datetime index.
    1. Check if a zero flatliner pattern is ongoing (i.e. all recent measurements are zero).
    2. Replace repeated values for longer than flatliner_threshold_minutes with NaN.

    Args:
        pj_id: ind/str, used to identify log statements
        data: pd.DataFrame where the first column should be the target. index=datetimeIndex
        flatliner_threshold_minutes: int indicating the number of minutes after which constant load is considered a flatline.
            if None, the validation is effectively skipped
        resolution_minutes: The forecasting resolution in minutes.

    Returns:
        Dataframe where repeated values are set to None

    Raises:
        InputDataOngoingZeroFlatlinerError: If all recent load measurements are zero.

    """
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(Settings.log_level)
        )
    )
    logger = structlog.get_logger(__name__)

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Input dataframe does not have a datetime index.")

    if flatliner_threshold_minutes is None:
        logger.info("Skipping validation of input data", pj_id=pj_id)
        return data

    zero_flatliner_ongoing = detect_ongoing_zero_flatliner(
        load=data.iloc[:, 0], duration_threshold_minutes=flatliner_threshold_minutes
    )

    if zero_flatliner_ongoing:
        raise InputDataOngoingZeroFlatlinerError(
            "All recent load measurements are zero."
        )

    flatliner_threshold_repetitions = math.ceil(
        flatliner_threshold_minutes / resolution_minutes
    )

    # Drop 'false' measurements. e.g. where load appears to be constant.
    data = replace_repeated_values_with_nan(
        data, threshold=flatliner_threshold_repetitions, column_name=data.columns[0]
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
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(Settings.log_level)
        )
    )
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

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(Settings.log_level)
        )
    )
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
        weights = np.array([1] * (len(df_copy.columns)))

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


def detect_ongoing_zero_flatliner(
    load: pd.Series,
    duration_threshold_minutes: int,
) -> bool:
    """Detects if the latest measurements follow a zero flatliner pattern.

    Args:
        load (pd.Series): A timeseries of measured load with a datetime index.
        duration_threshold_minutes (int): A zero flatliner is only detected if it exceeds the threshold duration.

    Returns:
        bool: Indicating whether or not there is a zero flatliner ongoing for the given load.

    """
    # remove all timestamps in the future
    load = load[load.index <= datetime.now(tz=UTC)]
    latest_measurement_time = load.dropna().index.max()
    latest_measurements = load[
        latest_measurement_time - timedelta(minutes=duration_threshold_minutes) :
    ].dropna()

    return (latest_measurements == 0).all() & (not latest_measurements.empty)


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
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(Settings.log_level)
        )
    )
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
            column: len(df)
            - eval(column[2:].replace("min", "/60").replace("d", "*24.0")) / 0.25
            for column in df.columns
            if column.startswith("T-")
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

        # Correct for day_ahead_electricity_price being only expected to be available up to 24h
        if "day_ahead_electricity_price" in non_na_count.index:
            non_na_count["day_ahead_electricity_price"] += max([len(df) - 96, 0])

        completeness_per_column_dataframe = non_na_count / (len(df))

    return completeness_per_column_dataframe
