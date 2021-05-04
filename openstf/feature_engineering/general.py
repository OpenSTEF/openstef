# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


def add_missing_feature_columns(
    input_data: pd.DataFrame, featurelist: list
) -> pd.DataFrame:
    """Adds feature column for features in the featurelist.

    Add feature columns for features in the feature list if these columns don't
    exist in the input data. If a column is added, its value is set to NaN.
    This is especially usefull to make sure the required columns are in place when making a prediction.

    NOTE: this function is intended as a final check to prevent errors during predicion.
        In an ideal world this function is not nescarry.

    Args:
        input_data (pd.DataFrame): DataFrame with input data and featurs.
        featurelist (list): List of requiered features
    """
    missing_features = [f for f in featurelist if f not in list(input_data)]

    for feature in missing_features:
        logger.warning(
            f"Adding NaN column for missing feature: {feature}", missing_feature=feature
        )
        input_data[feature] = np.nan

    return input_data


def remove_extra_feature_columns(
    input_data: pd.DataFrame, featurelist: list
) -> pd.DataFrame:
    """Removes any features that are provided in the input data but not in the feature list.
    This should not be nescesarry but serves as an extra failsave for making predicitons

    Args:
        input_data: (pd.Dataframe) DataFrame with features
        featurelist: (list) list of reuqested features

    Returns: pd.DataFrame with model input data and fetaures

    """
    extra_features = [f for f in list(input_data) if f not in featurelist]

    num_not_requested_features = len(extra_features)

    if num_not_requested_features != 0:
        logger.warning(
            f"Removing {num_not_requested_features} unrequested features!",
            num_not_requested_features=num_not_requested_features,
        )

    return input_data.drop(extra_features, axis=1)
