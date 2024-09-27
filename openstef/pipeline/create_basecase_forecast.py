# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import logging
from pathlib import Path

import pandas as pd
import structlog

from openstef.data_classes.prediction_job import PredictionJobDataClass
from openstef.exceptions import InputDataOngoingZeroFlatlinerError, NoRealisedLoadError
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
from openstef.settings import Settings
from openstef.validation import validation

MODEL_LOCATION = Path(".")
BASECASE_HORIZON_MINUTES = 60 * 24 * 14  # 14 days ahead
BASECASE_RESOLUTION_MINUTES = 15


def create_basecase_forecast_pipeline(
    pj: PredictionJobDataClass,
    input_data: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the base case forecast and confidence intervals for a given prediction job and input data.

    Args:
        pj: Prediction job
        input_data: data frame containing the input data necessary for the prediction.

    Returns:
        Base case forecast

    Raises:
        NoRealisedLoadError: When no realised load for given datetime range.

    """
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(Settings.log_level)
        )
    )
    logger = structlog.get_logger(__name__)

    logger.info("Preprocessing data for basecase forecast")

    forecast_start, forecast_end = generate_forecast_datetime_range(input_data)

    if not isinstance(input_data.index, pd.DatetimeIndex):
        raise ValueError("Input dataframe does not have a datetime index.")

    zero_flatliner_ongoing = validation.detect_ongoing_zero_flatliner(
        load=input_data.iloc[:, 0],
        duration_threshold_minutes=pj.flatliner_threshold_minutes,
    )

    if zero_flatliner_ongoing:
        # Set historic load to zero to force the basecase forecasts to be zero.
        input_data.loc[input_data.index < forecast_start, "load"] = 0

    # Add features
    data_with_features = OperationalPredictFeatureApplicator(
        horizons=[0.25],
        feature_names=[
            "T-7d",
            "T-14d",
        ],  # Generate features for load 7 days ago and load 14 days ago these are the same as the basecase forecast.
    ).add_features(input_data)

    forecast_input = data_with_features[forecast_start:forecast_end]

    # Initialize model
    model = BaseCaseModel()
    logger.info("Making basecase forecast")
    # Make basecase forecast
    basecase_forecast = BaseCaseModel().predict(forecast_input)

    # Check if input data is available
    if len(basecase_forecast) == 0:
        raise NoRealisedLoadError(pj["id"])

    # Estimate the stdev by using the stdev of the hour for historic (T-14d) load
    model.standard_deviation = generate_basecase_confidence_interval(forecast_input)
    logger.info("Postprocessing basecase forecast")
    # Apply confidence interval
    basecase_forecast = ConfidenceIntervalApplicator(
        model, forecast_input
    ).add_confidence_interval(basecase_forecast, pj)

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


def generate_basecase_confidence_interval(
    data_with_features: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate confidence interval for a basecase forecast.

    Args:
        data_with_features: Input dataframe that is used to make the basecase forecast.

    Returns:
        Dataframe with the confidence interval.

    """
    confidence_interval = (
        data_with_features[["T-14d"]]  # Select only the T-14d column as a DataFrame
        .groupby(data_with_features.index.hour)  # Get the std for every hour
        .std()
        .rename(columns={"T-14d": "stdev"})  # Rename the column to stdev
    )
    confidence_interval["hour"] = confidence_interval.index
    confidence_interval["horizon"] = 48
    return confidence_interval
