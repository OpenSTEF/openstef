# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import pandas as pd
import structlog
from openstef_dbc.services.prediction_job import PredictionJobDataClass

import openstef.postprocessing.postprocessing as postprocessing


def create_components_forecast_pipeline(
    pj: PredictionJobDataClass, input_data, weather_data, split_coefs
):
    """Pipeline for creating a component forecast

    Args:
        pj (PredictionJobDataClass): Prediction job
        input_data (pd.DataFrame): Input forecast for the components forecast.
        weather_data (pd.DataFrame): Weather data with 'radiation' and 'windspeed_100m' columns
        split_coefs (dict): coefieicnts for the splitting that are determined earlier

    Returns:
        pd.DataFrame with component forecasts. The dataframe contains these columns: "forecast_wind_on_shore",
                "forecast_solar",
                "forecast_other",
                "pid",
                "customer",
                "description",
                "type",
                "algtype"
    """
    logger = structlog.get_logger(__name__)
    logger.info("Make components prediction", pid=pj["id"])

    # Make component forecasts
    try:
        forecasts = postprocessing.split_forecast_in_components(
            input_data, weather_data, split_coefs
        )
    except Exception as e:
        # In case something goes wrong we fall back on aan empty dataframe
        logger.warning(
            f"Could not make component forecasts: {e}, falling back on series of zeros!",
            exc_info=e,
        )
        forecasts = pd.DataFrame()

    # Prepare for output
    forecasts = postprocessing.add_prediction_job_properties_to_forecast(
        pj, forecasts, algorithm_type="component"
    )

    return forecasts
