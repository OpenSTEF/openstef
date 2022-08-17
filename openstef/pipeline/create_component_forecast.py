# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import pandas as pd
import structlog

import openstef.postprocessing.postprocessing as postprocessing
from openstef.data_classes.prediction_job import PredictionJobDataClass

from openstef.model.regressors.better_dazls import BetterDazls

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
    input_df = pd.concat([input_data[["load"]].rename(columns={"load":"total_substation"}), weather_data[["radiation", "windspeed_100m"]]], axis=1)

    # Add additional features
    input_df["lat"] = pj["lat"]
    input_df["lon"] = pj["lon"]

    input_df["solar_on"] = 1
    input_df["wind_on"] = 1
    input_df['hour'] = input_df.index.hour
    input_df['minute'] = input_df.index.minute

    input_df["var0"] = input_df["radiation"].var()
    input_df["var1"] = input_df["windspeed_100m"].var()
    input_df["var2"] = input_df["total_substation"].var()

    input_df["sem0"] = input_df.sem(axis = 0)
    input_df["sem1"] = input_df.sem(axis = 1)

    return input_df

def create_components_forecast_pipeline(
    pj: PredictionJobDataClass, input_data, weather_data):

    """
    Pipeline for creating a component forecast using BetterDazls prediction model

    Args:
        pj (PredictionJobDataClass): Prediction job
        input_data (pd.DataFrame): Input forecast for the components forecast.
        weather_data (pd.DataFrame): Weather data with 'radiation' and 'windspeed_100m' columns

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

    input_data = create_input(pj, input_data, weather_data)

    # Make component forecasts
    forecasts = BetterDazls.predict(input_data)

    forecasts["forecast_other"] = input_data["total_substation"] -  forecasts["forecast_solar"]  -forecasts["forecast_wind_on_shore"]


    # Prepare for output (it adds the extra columns in the output)
    forecasts = postprocessing.add_prediction_job_properties_to_forecast(
        pj, forecasts[['total_solar_part', 'total_wind_part', 'forecast_other']].rename(columns={"total_solar_part":"forecast_solar", "total_wind_part":"forecast_wind_on_shore"}), algorithm_type="component"
    )

    return forecasts
