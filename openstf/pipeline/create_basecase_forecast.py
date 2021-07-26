# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import structlog
from openstf.enums import ForecastType
from openstf.feature_engineering.feature_applicator import (
    OperationalPredictFeatureApplicator,
)
from openstf.model.basecase import BaseCaseModel
from openstf.model.confidence_interval_applicator import ConfidenceIntervalApplicator
from openstf.pipeline.utils import generate_forecast_datetime_range
from openstf.postprocessing.postprocessing import (
    add_prediction_job_properties_to_forecast,
    add_components_base_case_forecast,
)
from openstf.validation import validation

MODEL_LOCATION = Path(".")
BASECASE_HORIZON = 60 * 24 * 14  # 14 days ahead
BASECASE_RESOLUTION = 15

class BaseCaseNoForecastException(Exception):
    """Exception if no basecaseforecast could be made"""

    def __init__(self, message):
        self.message = message


def create_basecase_forecast_pipeline(
    pj: dict, input_data: pd.DataFrame
) -> pd.DataFrame:
    """Computes the base case forecast and confidence intervals for a given prediction job and input data.

    Args:
        pj: (dict) prediction job
        input_data (pandas.DataFrame): data frame containing the input data necessary for the prediction.

    Returns:
        basecase_forecast (pandas.DataFrame)
    """

    logger = structlog.get_logger(__name__)

    logger.info("Preprocessing data for basecase forecast")
    # Validate and clean data - use a very long flatliner threshold.
    # If a measurement was constant for a long period, a basecase should still be made.
    validated_data = validation.validate(input_data, flatliner_threshold=4*24*14+1)

    # Add features
    data_with_features = OperationalPredictFeatureApplicator(
        horizons=[0.25],
        feature_names=[
            "T-7d",
            "T-14d",
        ],  # Generate features for load 7 days ago and load 14 days ago these are the same as the basecase forecast.
    ).add_features(validated_data)

    # Select desired period for basecase forecast.
    # This should NOT include the first 24 hours
    # TODO it would make more sense to include this check right before writing the data to the database,
    # or at least at the Task level
    data_selected_period = data_with_features.loc[data_with_features.index > (pd.to_datetime(datetime.utcnow(), utc=True)+timedelta(hours=48)), :]

    if len(data_selected_period.dropna(how='all')) == 0:
        raise BaseCaseNoForecastException('Length of input for basecaseforecast was zero or all NA. No basecase forecast could be made')

    # Initialize model
    model = BaseCaseModel()
    logger.info("Making basecase forecast")
    # Make basecase forecast
    basecase_forecast = BaseCaseModel().predict(data_selected_period)

    # Estimate the stdev by using the stdev of the hour for historic (T-14d) load
    model.standard_deviation = generate_basecase_confidence_interval(data_with_features)
    logger.info("Postprocessing basecase forecast")
    # Apply confidence interval
    basecase_forecast = ConfidenceIntervalApplicator(
        model, data_selected_period
    ).add_confidence_interval(basecase_forecast, pj, default_confindence_interval=True)

    # Add basecase for the component forecasts
    basecase_forecast = add_components_base_case_forecast(basecase_forecast)

    # Do further postprocessing
    basecase_forecast = add_prediction_job_properties_to_forecast(
        pj=pj,
        forecast=basecase_forecast,
        algorithm_type="basecase_lastweek",
        forecast_quality="not_renewed",
    )

    return basecase_forecast


def generate_basecase_confidence_interval(data_with_features):
    confidence_interval = (
        data_with_features[["T-14d"]]  # Select only the T-14d column as a DataFrame
        .groupby(data_with_features.index.hour)  # Get the std for every hour
        .std()
        .rename(columns={"T-14d": "stdev"})  # Rename the column to stdev
    )
    confidence_interval["hour"] = confidence_interval.index
    confidence_interval["horizon"] = 48
    return confidence_interval
