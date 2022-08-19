# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
import joblib
import pandas as pd
import structlog

import openstef.postprocessing.postprocessing as postprocessing
from openstef.data_classes.prediction_job import PredictionJobDataClass
from openstef.model.regressors.better_dazls import BetterDazls
from openstef.enums import ForecastType

from openstef import PROJECT_ROOT

# Set the path for the BetterDazls stored model
BETTER_DAZLS_STORED = PROJECT_ROOT / "openstef" / "data" / "better_dazls_stored.sav"



def create_input(pj, input_data, weather_data):

    """
    This function prepares the input data, which will be used for the BetterDazls model prediction, so they will be
    according BetterDazls model requirements.

    :param pj: pj (PredictionJobDataClass): Prediction job
    :param input_data: (pd.DataFrame): Input forecast for the components forecast.
    :param weather_data: (pd.DataFrame): Weather data with 'radiation' and 'windspeed_100m' columns

    :return: input_df (pd.Dataframe): It outputs a dataframe which will be used fot the BetterDazls prediction function.
    """

    # Prepare raw input data

    input_df = weather_data[["radiation", "windspeed_100m"]].merge(input_data[["forecast"]].rename(columns={"forecast":"total_substation"}), how="inner", right_index=True, left_index=True)


    # Add additional features
    input_df["lat"] = pj["lat"]
    input_df["lon"] = pj["lon"]

    input_df["solar_on"] = 1
    input_df["wind_on"] = 1
    input_df["hour"] = input_df.index.hour
    input_df["minute"] = input_df.index.minute

    input_df["var0"] = input_df["radiation"].var()
    input_df["var1"] = input_df["windspeed_100m"].var()
    input_df["var2"] = input_df["total_substation"].var()

    input_df["sem0"] = input_df.sem(axis = 1)
    input_df["sem1"] = input_df.sem(axis = 1)


    return input_df


def create_components_forecast_pipeline(
    pj: PredictionJobDataClass, input_data, weather_data
):

    """
    Pipeline for creating a component forecast using BetterDazls prediction model

    Args:
        pj (PredictionJobDataClass): Prediction job
        input_data (pd.DataFrame): Input forecast for the components forecast.
        weather_data (pd.DataFrame): Weather data with 'radiation' and 'windspeed_100m' columns

    Returns:
        pd.DataFrame with component forecasts. The dataframe contains these columns:
                "forecast_wind_on_shore",
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

    input_data = create_input(pj, input_data, weather_data)

    # Save and load the model as .sav file
    # The code for this is the train_component_model.ipynb file
    better_dazls_model = joblib.load(BETTER_DAZLS_STORED)

    # Use the predict function of BetterDazls model
    # As input data we use the input_data function which takes into consideration what we want as an input for the -
    # forecast and what BetterDazls can accept as an input
    forecasts = better_dazls_model.predict(test_features=input_data)

    # Set the columns for the output forecast dataframe
    # Make forecasts for the components: forecast_wind_on_shore","forecast_solar" and "forecast_other"
    forecasts = pd.DataFrame(forecasts, columns=["forecast_wind_on_shore", "forecast_solar"], index = input_data.index)

    forecasts["forecast_solar"] = postprocessing.post_process_wind_solar(forecasts["forecast_solar"], forecast_type=ForecastType.SOLAR)
    forecasts["forecast_wind_on_shore"] = postprocessing.post_process_wind_solar(forecasts["forecast_wind_on_shore"],
                                                                         forecast_type=ForecastType.WIND)
    forecasts["forecast_other"] = input_data["total_substation"] -  forecasts["forecast_solar"]  -forecasts["forecast_wind_on_shore"]

    # Prepare for output
    # Add more prediction properties to the forecast ("pid","customer","description","type","algtype)
    forecasts = postprocessing.add_prediction_job_properties_to_forecast(
        pj, forecasts, algorithm_type="component"
    )
    return forecasts
