# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

""" This module contains historic load related functions used for feature engineering.

"""
from typing import List
import pandas as pd
import structlog
from openstef_dbc.services.prediction_job import PredictionJobDataClass


def add_historic_load_as_a_feature(
    data: pd.DataFrame,
    pj: PredictionJobDataClass = None,
) -> pd.DataFrame:
    """Adds additional proloaf features to the input data, historic_load (equal to the load)

    Args:
        data (pd.DataFrame): Dataframe to which the wind features have to be added
        pj (PredictionJobDataClass): Prediction job.

    Returns:
        pd.DataFrame same as input dataframe with extra columns for the added proloaf features

    """
    logger = structlog.get_logger(__name__)

    if pj is not None and pj["model"] == "proloaf":
        data["historic_load"] = data["load"]
        logger.warning(
            "The historic_load is added to the data, this is a copy of the load. "
            "Adding this feature is in most of the cases not a good idea, it is designed for the proloaf model."
        )

    return data
