# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import logging
from enum import Enum

import numpy as np
import pandas as pd
import structlog

from openstef.data_classes.prediction_job import PredictionJobDataClass
from openstef.enums import ForecastType
from openstef.feature_engineering import weather_features
from openstef.settings import Settings

# this is the default for "Lagerwey100"
TURBINE_DATA = {
    "rated_power": 1,
    "slope_center": 8.07,
    "steepness": 0.664,
}

# Set value to define precission of power, this is needed because comparing to zero sometimes leads to issues for very small values.
SMALLEST_POWER_UNIT: float = 0.000001


def normalize_and_convert_weather_data_for_splitting(
    weather_data: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize and converts weather data for use in energy splitting.

    Args:
        weather_data: Weather data with "windspeed_100m" and "radiation".

    Returns:
         Dataframe with "windpower" and "radiation" columns.

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
    wind_ref_series = weather_features.calculate_windspeed_at_hubheight(
        weather_data["windspeed_100m"], fromheight=100
    )
    wind_ref = wind_ref_series.to_frame()
    wind_ref = calculate_wind_power(wind_ref)
    wind_ref *= -1

    output_dataframe["windpower"] = wind_ref
    return output_dataframe


def calculate_wind_power(
    windspeed_100m: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the generated wind power based on the wind speed.

    Values are related through the power curve, which is
    described by turbine_data. Default values are used and are normalized to 1MWp.

    Args:
        windspeed_100m: Example: ``pd.DataFrame (index = datetime, columns = ["windspeed_100m"])``

    Returns:
        Example output ``pd.DataFrame(index = datetime, columns = ["windenergy"])``

    """
    generated_power = TURBINE_DATA["rated_power"] / (
        1
        + np.exp(
            -TURBINE_DATA["steepness"] * (windspeed_100m - TURBINE_DATA["slope_center"])
        )
    )
    return generated_power["windspeed_100m"].rename("windenergy").to_frame()


def split_forecast_in_components(
    forecast: pd.DataFrame, weather_data: pd.DataFrame, split_coefs: dict
) -> dict[str, pd.DataFrame]:
    """Make estimates of energy components based on given forecast.

    Args:
        forecast: KTP load forecast
        weather_data: Weather data for energy splitting, at least; "windspeed_100m" and "radiation"
        split_coefs: Previously determined splitting coefs for prediction job

    Returns:
        Forecast dataframe for each component

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


def post_process_wind_solar(
    forecast: pd.Series, forecast_type: ForecastType
) -> pd.DataFrame:
    """Function that caries out postprocessing for wind and solar power generators.

        As these points will always produce energy, predicted energy consumption is
        set to zero. This function enforces the assumption that production is negative
        and consuption positive.

    Args:
        forecast: Series with forecast data.
        forecast_type: Specifies the type of forecast. This can be retrieved
            from the prediction job as pj['forecast_type']

    Returns:
        Post-processed forecast.

    """
    if forecast_type not in [ForecastType.WIND, ForecastType.SOLAR]:
        return forecast

    # For wind and solar forecasted value should always be negative.
    forecast.loc[forecast > (-1 * SMALLEST_POWER_UNIT)] = 0

    # write changed back to forecast
    return forecast


def add_components_base_case_forecast(basecase_forecast: pd.DataFrame) -> pd.DataFrame:
    """Makes a basecase forecast for the forecast_other component.

    This will make a simple basecase components forecast
    available and ensures that the sum of the components (other, wind and solar) is equal to the normal basecase
    forecast This is important for sending GLMD messages correctly to TenneT!

    Args:
        basecase_forecast: pd.DataFrame with basecase forecast

    Returns:
        basecase_forecast: pd.DataFrame with extra "forecast_other component"

    """
    basecase_forecast["forecast_other"] = basecase_forecast["forecast"]
    return basecase_forecast


def add_prediction_job_properties_to_forecast(
    pj: PredictionJobDataClass,
    forecast: pd.DataFrame,
    algorithm_type: str,
    forecast_type: Enum = None,
    forecast_quality: str = None,
) -> pd.DataFrame:
    """Adds prediciton job meta data to a forecast dataframe.

    Args:
        pj: Prediciton job.
        forecast: Forecast dataframe
        algorithm_type: Type of algirithm used for making the forecast.
        forecast_type: Type of the forecast. Defaults to None.
        forecast_quality: Quality of the forecast. Defaults to None.

    Returns:
        Dataframe with added metadata.

    """
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(Settings.log_level)
        )
    )
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

    forecast["pid"] = pj["id"]
    forecast["customer"] = pj["name"]
    forecast["description"] = pj["description"]
    forecast["type"] = forecast_type
    forecast["algtype"] = algorithm_type

    return forecast


def sort_quantiles(
    forecast: pd.DataFrame, quantile_col_start="quantile_P"
) -> pd.DataFrame:
    """Sort quantile values so quantiles do not cross.

    This function assumes that all quantile columns start with 'quantile_P' For more academic details on why this is
    mathematically sounds, please refer to Quantile and Probability Curves Without Crossing (Chernozhukov, 2010)

    """
    p_columns = [col for col in forecast.columns if col.startswith(quantile_col_start)]

    if len(p_columns) == 0:
        return forecast

    # sort the columns
    p_columns = np.sort(p_columns)

    forecast.loc[:, p_columns] = forecast[p_columns].apply(sorted, axis=1).to_list()

    return forecast
