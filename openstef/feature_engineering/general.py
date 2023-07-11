# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
"""This modelu contains various helper functions."""

import numpy as np
import pandas as pd
import structlog


def add_missing_feature_columns(
    input_data: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    """Adds feature column for features in the featurelist.

    Add feature columns for features in the feature list if these columns don't
    exist in the input data. If a column is added, its value is set to NaN.
    This is especially usefull to make sure the required columns are in place when
    making a prediction.

    .. note::
        This function is intended as a final check to prevent errors during predicion.
        In an ideal world this function is not nescarry.

    Args:
        input_data: DataFrame with input data and featurs.
        features: List of requiered features.

    Returns:
        Input dataframe with missing columns filled with ``np.N=nan``.

    """
    logger = structlog.get_logger(__name__)

    if features is None:
        features = []

    missing_features = [f for f in features if f not in list(input_data)]

    for feature in missing_features:
        logger.warning(
            f"Adding NaN column for missing feature: {feature}", missing_feature=feature
        )
        input_data[feature] = np.nan

    return input_data


def remove_non_requested_feature_columns(
    input_data: pd.DataFrame, requested_features: list[str]
) -> pd.DataFrame:
    """Removes features that are provided in the input data but not in the feature list.

    This should not be nescesarry but serves as an extra failsave for making predicitons

    Args:
        input_data: DataFrame with features
        requested_features: List of reuqested features

    Returns:
        Model input data with features.

    """
    logger = structlog.get_logger(__name__)

    if requested_features is None:
        requested_features = []

    not_requested_features = [
        f for f in list(input_data) if f not in requested_features
    ]

    # Do not see "load" or "horizon" as an extra feature as it is no feature
    if "load" in not_requested_features:
        not_requested_features.remove("load")

    num_not_requested_features = len(not_requested_features)

    if num_not_requested_features != 0:
        logger.warning(
            f"Removing {num_not_requested_features} unrequested features!",
            num_not_requested_features=num_not_requested_features,
        )

    return input_data.drop(not_requested_features, axis=1)


def enforce_feature_order(input_data: pd.DataFrame) -> pd.DataFrame:
    """Enforces correct order of features.

    Alphabetically orders the feature columns. The load column remains the first column
    and the horizons column remains the last column.
    Everything in between is alphabetically sorted:
    The order eventually looks like this:
    ["load"] -- [alphabetically sorted features] -- ['horizon']

    This function assumes the first column contains the to be predicted variable
    Furthermore the "horizon" is moved to the last position if it is pressent.

    Args:
        input_data: Input data with features.

    Returns:
        Properly sorted input data.

    """
    # Extract first column name
    first_column_name = input_data.columns.to_list()[
        0
    ]  # Most of the time this is "load"

    # Sort columns
    columns = list(np.sort(input_data.columns.to_list()))

    # Remove first column and add to the start
    columns.remove(first_column_name)
    column_order = [first_column_name] + columns

    # If "Horzion" column is available add to the end
    if "horizon" in columns:
        # "horizon" is pressent in the training procces
        # but not in the forecasting process
        column_order.remove("horizon")
        column_order = column_order + ["horizon"]

    # Return dataframe with columns in the correct order
    return input_data.loc[:, column_order]
