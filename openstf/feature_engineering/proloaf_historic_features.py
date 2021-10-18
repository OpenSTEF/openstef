# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

""" This module contains all proloaf related functions used for feature engineering.

"""
from typing import List
import pandas as pd

def add_additional_proloaf_features(
    data: pd.DataFrame, feature_names: List[str] = None
) -> (pd.DataFrame, List[str]):
    """Adds additional proloaf features to the input data. historic_load (equal to the load)

    Args:
        data (pd.DataFrame): Dataframe to which the wind features have to be added
        feature_names (List[str]): List of requested features to transform in historic_load

    Returns:
        pd.DataFrame same as input dataframe with extra columns for the added proloaf features
        feature_names (List[str]): List of requested features to transform in historic_load

    """
    if feature_names is None:
        feature_names = list(data.columns)
        feature_names.append("historic_load")
        data["historic_load"] = data["load"]
    elif any([s for s in feature_names if "T-" in s]):
        feature_names.remove([s for s in feature_names if "T-" in s][0])
        feature_names.append("historic_load")
        data["historic_load"] = data["load"]

    return data, feature_names