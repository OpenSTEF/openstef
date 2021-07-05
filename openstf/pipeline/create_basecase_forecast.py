# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from pathlib import Path

import pandas as pd
import pytz
from datetime import timedelta
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
    # Validate and clean data
    validated_data = validation.validate(input_data)

    # Prep forecast input by selecting only the forecast datetime interval
    forecast_start, forecast_end = generate_forecast_datetime_range(
        BASECASE_RESOLUTION, BASECASE_HORIZON
    )

    # Dont forecast the horizon of the regular models
    forecast_start = forecast_start + timedelta(minutes=pj["horizon_minutes"])

    # Make sure forecast interval is available in the input interval
    new_date_range = pd.date_range(
        validated_data.index.min().to_pydatetime(),  # Start of the new range is start of the input interval
        forecast_end.replace(
            tzinfo=pytz.utc
        ),  # End of the new range is end of the forecast interval
        freq=f'{pj["resolution_minutes"]}T',  # Resample to the desired time resolution
    )
    validated_data = validated_data.reindex(new_date_range)  # Reindex to new date range

    # Add features
    data_with_features = OperationalPredictFeatureApplicator(
        horizons=[0.25],
        feature_names=[
            "T-7d",
            "T-14d",
        ],  # Generate features for load 7 days ago and load 14 days ago these are the same as the basecase forecast.
    ).add_features(validated_data)

    # Select the basecase forecast interval
    forecast_input_data = data_with_features[forecast_start:forecast_end]

    # Initialize model
    model = BaseCaseModel()
    logger.info("Making basecase forecast")
    # Make basecase forecast
    basecase_forecast = BaseCaseModel().predict(forecast_input_data)

    # Estimate the stdev by using the stdev of the hour for historic (T-14d) load
    model.standard_deviation = generate_basecase_confidence_interval(data_with_features)
    logger.info("Postprocessing basecase forecast")
    # Apply confidence interval
    basecase_forecast = ConfidenceIntervalApplicator(
        model, forecast_input_data
    ).add_confidence_interval(basecase_forecast, pj, default_confindence_interval=True)

    # Add basecase for the component forecasts
    basecase_forecast = add_components_base_case_forecast(basecase_forecast)

    # Do further postprocessing
    basecase_forecast = add_prediction_job_properties_to_forecast(
        pj=pj,
        forecast=basecase_forecast,
        algorithm_type="basecase_lastweek",
        forecast_type=ForecastType.BASECASE,
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
