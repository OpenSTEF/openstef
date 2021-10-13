# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from datetime import datetime, timedelta
from typing import Tuple, Union, List

import pandas as pd
from openstf_dbc.services.prediction_job import PredictionJobDataClass

from openstf.exceptions import (
    InputDataInsufficientError,
    InputDataWrongColumnOrderError,
)
from openstf.feature_engineering.feature_applicator import TrainFeatureApplicator
from openstf.validation import validation


def generate_forecast_datetime_range(
    resolution_minutes: int, horizon_minutes: int
) -> Tuple[datetime, datetime]:
    # get current date and time UTC
    datetime_utc = datetime.utcnow()
    # Datetime range for time interval to be predicted
    forecast_start = datetime_utc - timedelta(minutes=resolution_minutes)
    forecast_end = datetime_utc + timedelta(minutes=horizon_minutes)

    return forecast_start, forecast_end


def data_cleaning(
    pj: Union[PredictionJobDataClass, dict],
    input_data: pd.DataFrame,
    horizons: List[float],
) -> pd.DataFrame:
    """Clean the input data and perform some checks

    Args:
        pj: Prediction job
        input_data: input dataframe
        horizons: horizons to train on in hours

    Returns:
        pd.DataFrame: Cleaned input dataframe
    """
    if input_data.empty:
        raise InputDataInsufficientError("Input dataframe is empty")
    elif "load" not in input_data.columns:
        raise InputDataWrongColumnOrderError(
            "Missing the load column in the input dataframe"
        )

    # Validate and clean data
    validated_data = validation.clean(validation.validate(pj["id"], input_data))

    # Check if sufficient data is left after cleaning
    if not validation.is_data_sufficient(validated_data):
        raise InputDataInsufficientError(
            f"Input data is insufficient for {pj['name']} "
            f"after validation and cleaning"
        )

    return TrainFeatureApplicator(horizons=horizons).add_features(input_data)
