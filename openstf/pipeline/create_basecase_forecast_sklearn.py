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
from openstf.model.confidence_interval_applicator import (
    ConfidenceIntervalApplicator
)
from openstf.pipeline.create_forecast_sklearn import generate_forecast_datetime_range
from openstf.postprocessing.postprocessing import (
    add_prediction_job_properties_to_forecast,
    add_components_base_case_forecast,
)
from openstf.validation import validation

MODEL_LOCATION = Path(".")
BASECASE_HORIZON = 60 * 24 * 14  # 14 days ahead
BASECASE_RESOLUTION = 15


def basecase_pipeline(pj: dict, input_data: pd.DataFrame) -> pd.DataFrame:
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
    validated_data = validated_data.reindex(
        pd.date_range(
            validated_data.index.min().to_pydatetime(),
            forecast_end.replace(tzinfo=pytz.utc),
            freq=f'{pj["resolution_minutes"]}T',
        )
    )

    # Add features
    data_with_features = OperationalPredictFeatureApplicator(
        horizons=[0.25],
        feature_names=["T-7d", "T-14d"],
    ).add_features(validated_data)

    # Select the basecase forecast interval
    forecast_input_data = data_with_features[forecast_start:forecast_end]

    # Initialize model
    model = BaseCaseModel()
    logger.info("Making basecase forecast")
    # Make basecase forecast
    basecase_forecast = BaseCaseModel().predict(forecast_input_data)

    # Estimate the stdev by using the stdev of the hour for historic (T-14d) load
    model.confidence_interval = generate_basecase_confidence_interval(data_with_features)
    logger.info("Postprocessing basecase forecast")
    # Apply confidence interval
    basecase_forecast = ConfidenceIntervalApplicator(
        model
    ).add_confidence_interval(basecase_forecast, pj['quantiles'])

    # Add basecase for the component forecasts
    basecase_forecast = add_components_base_case_forecast(basecase_forecast)

    # Do further postprocessing
    basecase_forecast = add_prediction_job_properties_to_forecast(
        pj=pj,
        forecast=basecase_forecast,
        forecast_type=ForecastType.BASECASE,
        forecast_quality="not_renewed",
    )

    return basecase_forecast

def generate_basecase_confidence_interval(data_with_features):
    confidence_interval = (
        data_with_features[["T-14d"]]
            .groupby(data_with_features.index.hour)
            .std()
            .rename(columns={"T-14d": "stdev"})
    )
    confidence_interval['hour'] = confidence_interval.index
    confidence_interval['horizon'] = 48
    return confidence_interval
