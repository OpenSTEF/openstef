# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

# -*- coding: utf-8 -*-
""" This module contains all general feature engineering functions"""
import numpy as np


def add_missing_feature_columns(input_data, featurelist):
    """Adds feature column for features in the featurelist.

    Add feature columns for features in the feature list if these columns don't
    exist in the input data. If a column is added, its value is set to NaN.
    """
    missing_features = [f for f in featurelist if f not in list(input_data)]

    for feature in missing_features:
        print(f"Warning: adding NaN column for missing feature: {feature}")
        input_data[feature] = np.nan

    return input_data


def remove_extra_feature_columns(input_data, featurelist):
    """Removes any features that are provided in the input data but not in the feature list.
    This should not be nescesarry but serves as an extra failsave for making predicitons

    Args:
        input_data: (pd.Dataframe) DataFrame with features
        featurelist: (list) list of reuqested features

    Returns: pd.DataFrame with model input data and fetaures

    """
    extra_features = [f for f in list(input_data) if f not in featurelist]

    if len(extra_features) > 0:
        print(f"Warning: Removing {len(extra_features)} unrequested features!")

    return input_data.drop(extra_features, axis=1)
