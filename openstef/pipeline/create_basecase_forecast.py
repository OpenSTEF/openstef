# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from pathlib import Path

import pandas as pd
import structlog
from openstef_dbc.services.prediction_job import PredictionJobDataClass

from openstef.feature_engineering.feature_applicator import (
    OperationalPredictFeatureApplicator,
)
from openstef.model.basecase import BaseCaseModel
from openstef.model.confidence_interval_applicator import ConfidenceIntervalApplicator
from openstef.pipeline.utils import generate_forecast_datetime_range
from openstef.postprocessing.postprocessing import (
    add_components_base_case_forecast,
    add_prediction_job_properties_to_forecast,
)
from openstef.validation import validation

MODEL_LOCATION = Path(".")
BASECASE_HORIZON_MINUTES = 60 * 24 * 14  # 14 days ahead
BASECASE_RESOLUTION_MINUTES = 15


def create_basecase_forecast_pipeline(
    pj: PredictionJobDataClass,
    input_data: pd.DataFrame,
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
    validated_data = validation.validate(
        pj["id"], input_data, flatliner_threshold=4 * 24 * 14 + 1
    )

    # Add features
    data_with_features = OperationalPredictFeatureApplicator(
        horizons=[0.25],
        feature_names=[
            "T-7d",
            "T-14d",
        ],  # Generate features for load 7 days ago and load 14 days ago these are the same as the basecase forecast.
    ).add_features(validated_data)

    # Similarly to the forecast pipeline, only try to make a forecast for moments in the future
    # TODO, do we want to be this strict on time window of forecast in this place?
    # see issue https://github.com/OpenSTEF/openstef/issues/121
    forecast_start, forecast_end = generate_forecast_datetime_range(data_with_features)
    forecast_input = data_with_features[forecast_start:forecast_end]

    # Initialize model
    model = BaseCaseModel()
    logger.info("Making basecase forecast")
    # Make basecase forecast
    basecase_forecast = BaseCaseModel().predict(forecast_input)

    # Estimate the stdev by using the stdev of the hour for historic (T-14d) load
    model.standard_deviation = generate_basecase_confidence_interval(forecast_input)
    logger.info("Postprocessing basecase forecast")
    # Apply confidence interval
    basecase_forecast = ConfidenceIntervalApplicator(
        model, forecast_input
    ).add_confidence_interval(basecase_forecast, pj, quantile_confidence_interval=False)

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
