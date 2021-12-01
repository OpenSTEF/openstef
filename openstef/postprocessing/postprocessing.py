# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from enum import Enum

import numpy as np
import pandas as pd
import structlog
from openstef_dbc.services.prediction_job import PredictionJobDataClass

from openstef.enums import ForecastType
from openstef.feature_engineering import weather_features


def normalize_and_convert_weather_data_for_splitting(weather_data):
    """Normalize and converts weather data for use in energy splitting.

    Args:
        weather_data (pd.DataFrame): Weather data with "windspeed_100m" and "radiation".

    Returns:
         pd.DataFrame: Dataframe with "windpower" and "radiation" columns.

    """

    # Check we have "windspeed_100m" and "radiation" available
    if not all(
        elem in weather_data.columns for elem in ["windspeed_100m", "radiation"]
    ):
        raise ValueError("weather data does not contain required data!")

    # Prepare output dataframe
    output_dataframe = pd.DataFrame()

    # Normalize weather data
    output_dataframe["radiation"] = (
        weather_data["radiation"]
        / np.percentile(weather_data["radiation"].dropna(), 99.0)
        * -1
    )
    wind_ref = weather_features.calculate_windspeed_at_hubheight(
        weather_data["windspeed_100m"], fromheight=100
    )
    wind_ref = wind_ref / np.abs(np.amax(wind_ref)) * -1

    output_dataframe["windpower"] = wind_ref

    return output_dataframe


def split_forecast_in_components(forecast, weather_data, split_coefs):
    """Function that makes estimates of energy components based on given forecast,
        previously determine splitting coefficients and relevant weather data.

    Args:
        forecast(pd.DataFrame): KTP load forecast
        weather_data (pd.DataFrame): Weather data for energy splitting, at least:
            "windspeed_100m" and "radiation"
        split_coefs (dict): Previously determined splitting coefs for prediction job

    Returns:
        dict: Forecast dataframe for each component

    """

    # Normalize weather data
    weather_ref_profiles = normalize_and_convert_weather_data_for_splitting(
        weather_data
    )

    # Check input
    if not all(
        elem in ["windpower", "radiation"]
        for elem in list(weather_ref_profiles.columns)
    ):
        raise ValueError("weather data does not contain required data!")

    # Merge to ensure datetime index is the same
    weather_ref_profiles = forecast.merge(
        weather_ref_profiles, how="outer", right_index=True, left_index=True
    )
    # Drop rows with duplicate indices
    weather_ref_profiles = weather_ref_profiles[
        ~weather_ref_profiles.index.duplicated()
    ]
    weather_ref_profiles.replace([np.inf, -np.inf], np.nan).dropna(inplace=True)

    # Prepare output dictionary and list of forecast types
    components = forecast.copy(deep=True)

    # Calculate profiles of estimated components
    components["forecast_wind_on_shore"] = (
        split_coefs["wind_ref"] * weather_ref_profiles["windpower"]
    )
    components["forecast_solar"] = (
        split_coefs["pv_ref"] * weather_ref_profiles["radiation"]
    )
    components["forecast_other"] = (
        weather_ref_profiles["forecast"]
        - components["forecast_solar"]
        - components["forecast_wind_on_shore"]
    )

    # Check that sign of production components is negative and not positive, change if sign is wrong
    if components["forecast_wind_on_shore"].sum() > 0:
        raise ValueError("Sign of estimated wind_on_shore component is positive!")
    if components["forecast_solar"].sum() > 0:
        raise ValueError("Sign of estimated solar component is positive!")

    # Post process predictions to ensure realistic values
    components["forecast_solar"] = post_process_wind_solar(
        components["forecast_solar"], ForecastType.SOLAR
    )
    components["forecast_wind_on_shore"] = post_process_wind_solar(
        components["forecast_wind_on_shore"], ForecastType.WIND
    )

    return components.drop("forecast", axis=1).drop("stdev", axis=1).dropna()


def post_process_wind_solar(forecast: pd.Series, forecast_type):
    """Function that caries out postprocessing for wind and solar power generators.

        As these points will always produce energy, predicted energy consumption is
        set to zero. This function will automatically detect the sign as this can
        vary from case to case. The largest volume either positive or negative is
        assumed to be the energy production and the other side the energy consumption.

    Args:
        forecast (pd.Series): Series with forecast data.
        forecast_type (ForecastType): Specifies the type of forecast. This can be retrieved
            from the prediction job as pj['forecast_type']

    Returns:
        forecast (pd.DataFrame): post-processed forecast.

    """
    logger = structlog.get_logger(__name__)

    if forecast_type not in [ForecastType.WIND, ForecastType.SOLAR]:
        return forecast

    # NOTE this part is really generic, we should probably make it a generic function
    forecast_data_sum = forecast.sum()
    # Determine sign of sum
    if forecast_data_sum > 0:
        # Set all values smaller than zero to zero, since this is not realistic
        forecast.loc[forecast < 0] = 0
    elif forecast_data_sum < 0:
        # Likewise for all values greater than zero
        forecast.loc[forecast > 0] = 0
    else:
        logger.warning(
            f"Could not determine sign of the forecast, skip post-processing. Sum was {forecast_data_sum}"
        )

    # write changed back to forecast
    return forecast


def add_components_base_case_forecast(basecase_forecast: pd.DataFrame) -> pd.DataFrame:
    """Makes a basecase forecast for the forecast_other component. This will make a
        simple basecase components forecast available and ensures that the sum of
        the components (other, wind and solar) is equal to the normal basecase forecast
        This is important for sending GLMD messages correctly to TenneT!

    Args:
        basecase_forecast: pd.DataFrame with basecase forecast

    Returns:
        basecase_forecast: pd.DataFrame with extra "forecast_other component"

    """
    #
    basecase_forecast["forecast_other"] = basecase_forecast["forecast"]
    return basecase_forecast


def add_prediction_job_properties_to_forecast(
    pj: PredictionJobDataClass,
    forecast: pd.DataFrame,
    algorithm_type: str,
    forecast_type: Enum = None,
    forecast_quality: str = None,
) -> pd.DataFrame:

    logger = structlog.get_logger(__name__)

    logger.info("Postproces in preparation of storing")
    if forecast_type is None:
        forecast_type = pj["forecast_type"]
    else:
        # get the value from the enum
        forecast_type = forecast_type.value

    # NOTE this field is only used when making the babasecase forecast and fallback
    if forecast_quality is not None:
        forecast["quality"] = forecast_quality

    # TODO rename prediction job typ to type
    # TODO algtype = model_file_path, perhaps we can find a more logical name
    # TODO perhaps better to make a forecast its own class!
    # TODO double check and sync this with make_basecase_forecast (other fields are added)
    # !!!!! TODO fix the requirement for customer
    forecast["pid"] = pj["id"]
    forecast["customer"] = pj["name"]
    forecast["description"] = pj["description"]
    forecast["type"] = forecast_type
    forecast["algtype"] = algorithm_type

    return forecast
