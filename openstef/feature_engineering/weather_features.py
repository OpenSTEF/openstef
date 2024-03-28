# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

"""This module contains all wheather related functions used for feature engineering."""
import logging
from typing import Union

import numpy as np
import pandas as pd
import pvlib
import structlog
from pvlib.location import Location

from openstef.data_classes.prediction_job import PredictionJobDataClass
from openstef.settings import Settings

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(
        logging.getLevelName(Settings.log_level)
    )
)
logger = structlog.get_logger(__name__)


# Set some (nameless) constants for the Antoine equation:
A: float = 6.116
M: float = 7.6
TN: float = 240.7
# Set some constants
TORR: float = 133.322368  # 1 torr = 133 Pa
# 1.168 is the mass of 1 m^3 of air on sea level with standard pressure.
D: float = 1.168

DEFAULT_LAT: float = 52.132633
DEFAULT_LON: float = 5.291266


def calc_saturation_pressure(
    temperature: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Calculate the water vapour pressure from the temperature.

    See https://www.vaisala.com/sites/default/files/documents/Humidity_Conversion_Formulas_B210973EN-F.pdf.

    Args:
        temperature: Temperature in C
    Returns:
        The saturation pressure of water at the respective temperature

    """
    psat = A * 10 ** ((M * temperature) / (temperature + TN))
    return psat


def calc_vapour_pressure(
    rh: Union[float, np.ndarray], psat: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Calculates the vapour pressure.

    Args:
        rh: Relative humidity
        psat: Saturation pressure: see calc_saturation_pressure
    Returns:
        The water vapour pressure

    """
    return rh * psat


def calc_dewpoint(
    vapour_pressure: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Calculates the dewpoint, see https://en.wikipedia.org/wiki/Dew_point for mroe info.

    Args:
        vapour_pressure: The vapour pressure for which the dewpoint should be calculated
    Returns:
        Dewpoint

    """
    return TN / ((M / np.log10(vapour_pressure / A)) - 1)


def calc_air_density(
    temperature: Union[float, np.ndarray],
    pressure: Union[float, np.ndarray],
    rh: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Calculates the dewpoint.

    Args:
        temperature: The temperature in C
        pressure: the atmospheric pressure in Pa
        rh: Relative humidity

    Returns:
        The air density (kg/m^3)

    """
    # Calculate saturation pressure
    psat = calc_saturation_pressure(temperature)
    # Calculate the current vapour pressure
    vapour_pressure = calc_vapour_pressure(rh, psat)

    # Set tempareture to K
    temperature_k = temperature + 273.15

    # Calculate air density
    air_density = (
        D
        * (273.15 / temperature_k)
        * ((pressure - 0.3783 * vapour_pressure) / 760 / TORR)
    )

    return air_density


def add_humidity_features(
    data: pd.DataFrame, feature_names: list[str] = None
) -> pd.DataFrame:
    """Adds humidity features to the input dataframe.

    These features are calculated using functions defines in this module. A list of
    requested features is used to determine whether to add the humidity features or not.

    Args:
        data: Input dataframe to which features have to be added
        feature_names: list of requested features.

    Returns:
        Same as input dataframe with extra columns for the humidty features.

    """
    # If features is none add humidity feature anyway
    if feature_names is None:
        humidity_features = True

    # Otherwise check if they are among the reuqested features
    else:
        humidity_features = any(
            x
            in [
                "saturation_pressure",
                "vapour_pressure",
                "dewpoint",
                "air_density",
            ]
            for x in feature_names
        )

    # Check if any of the humidity features are requested and add them
    if humidity_features:
        # Try to add humidity  calculations, ignore if required columns are missing
        try:
            humidity_df = humidity_calculations(data.temp, data.humidity, data.pressure)
            data = data.join(humidity_df)
        except AttributeError:
            pass  # This happens when a required column for humidity_calculations
            # is not present

    return data


def humidity_calculations(
    temperature: Union[float, np.ndarray],
    rh: Union[float, np.ndarray],
    pressure: Union[float, np.ndarray],
) -> Union[dict, np.ndarray]:
    """Function that calculates weather features based on humidity..

    These features are:
        - Saturation pressure
        - Vapour pressure
        - Dewpoint
        - Air density

    Args:
        temperature: Temperature in C
        rh: Relative humidity in %
        pressure: The air pressure in hPa

    Returns:
        If the input is an np.ndarray; a pandas dataframe with the calculated moisture indices,
        if the input is numeric; a dict with the calculated moisture indices

    """
    # First: a sanity check on the relative humidity and the air pressure
    # We only check on the type of temperature, because they need to be the same anyway
    is_series = isinstance(temperature, (np.ndarray, pd.Series))
    is_scalar = isinstance(temperature, (float, int))

    if is_scalar is False and is_series is False:
        raise TypeError(
            "The input should be a pandas series or np.ndarry, or float or int"
        )

    # Suppres copy warnings
    with pd.option_context("mode.chained_assignment", None):
        if is_series:
            rh[rh > 1] = rh / 100  # This triggers copy warnings
            pressure[pressure < 80000] = np.nan  # This triggers copy warnings
        else:
            if rh > 1:
                rh /= 100
            if pressure < 80000:
                pressure = np.nan

    # If the input is a dataframe or np.ndarrays: return a dataframe
    if is_series:
        humidity_df = pd.DataFrame(
            columns=[
                "saturation_pressure",
                "vapour_pressure",
                "dewpoint",
                "air_density",
            ]
        )
        humidity_df["saturation_pressure"] = calc_saturation_pressure(temperature)
        humidity_df["vapour_pressure"] = calc_vapour_pressure(
            rh, humidity_df.saturation_pressure
        )
        humidity_df["dewpoint"] = calc_dewpoint(humidity_df.vapour_pressure)
        humidity_df["air_density"] = calc_air_density(temperature, pressure, rh)

        return humidity_df

    # Else: if the input is numeric: return a dict
    psat = calc_saturation_pressure(temperature)
    pw = calc_vapour_pressure(rh, psat)
    td = calc_dewpoint(pw)
    air_density = calc_air_density(temperature, pressure, rh)
    return {
        "saturation_pressure": psat,
        "vapour_pressure": pw,
        "dewpoint": td,
        "air_density": air_density,
    }


def calculate_windspeed_at_hubheight(
    windspeed: Union[float, pd.Series],
    fromheight: float = 10.0,
    hub_height: float = 100.0,
) -> pd.Series:
    """Calculate windspeed at hubheight.

    Calculates the windspeed at hubheigh by extrapolation from a given height to a given
    hub height using the wind power law https://en.wikipedia.org/wiki/Wind_profile_power_law

    Args:
        windspeed: float OR pandas series of windspeed at height = height
        fromheight: height (m) of the windspeed data. Default is 10m
        hubheight: height (m) of the turbine

    Returns:
        Windspeed at hubheight.

    """
    alpha = 0.143

    if not isinstance(windspeed, (np.ndarray, float, int, pd.Series)):
        raise TypeError(
            "The windspeed is not of the expected type!\n                        Got"
            " {}, expected np.ndarray, pd series or numeric".format(type(windspeed))
        )

    try:
        if any(windspeed < 0):
            raise ValueError(
                "The windspeed cannot be negative, as it is the lenght of a vector"
            )
    except TypeError:
        if windspeed < 0:
            raise ValueError(
                "The windspeed cannot be negative, as it is the lenght of a vector"
            )
        windspeed = abs(windspeed)

    return windspeed * (hub_height / fromheight) ** alpha


def calculate_windturbine_power_output(
    windspeed: pd.Series, n_turbines: int = 1, turbine_data: dict = None
) -> pd.Series:
    """Calculate wind turbine power output.

    These values are related through the power curve, which is described by turbine_data.
    If no turbine_data is given, default values are used and results are normalized to 1MWp.
    If n_turbines=0, the result is normalized to a rated power of 1.

    Args:
        windspeed: pd.DataFrame(index = datetime, columns = ["windspeedHub"])
        nTurbines: The number of turbines
        turbineData: slope_center, rated_power, steepness

    Returns:
        pd.DataFrame(index = datetime, columns = ["forecast"])

    """
    if turbine_data is None:
        turbine_data = {
            "name": "Lagerwey L100",  # not used here
            "cut_in": 3,  # not used here
            "cut_off": 25,  # not used here
            "kind": "onshore",  # not used here
            "manufacturer": "Lagerwey",  # not used here
            "peak_capacity": 1,  # not used here
            "rated_power": 1,
            "slope_center": 8.07,
            "steepness": 0.664,
        }
    else:
        required_properties = ["rated_power", "steepness", "slope_center"]
        for prop in required_properties:
            if prop not in turbine_data.keys():
                raise KeyError(f"Required property '{prop}' not set in turbine data")

    generated_power = turbine_data["rated_power"] / (
        1
        + np.exp(
            -turbine_data["steepness"] * (windspeed - turbine_data["slope_center"])
        )
    )
    generated_power *= n_turbines

    return generated_power


def add_additional_wind_features(
    data: pd.DataFrame, feature_names: list[str] = None
) -> pd.DataFrame:
    """Adds additional wind features to the input data.

    Args:
        data: Dataframe to which the wind features have to be added
        feature_names: List of requested features

    Returns:
        DataFrame same as input dataframe with extra columns for the added wind features

    """
    if feature_names is None:
        additional_wind_features = True
    else:
        additional_wind_features = any(
            x
            in [
                "windspeed_100mExtrapolated",
                "windPowerFit_extrapolated",
                "windpowerFit_harm_arome",
            ]
            for x in feature_names
        )

    # Add add_additional_wind_features
    if "windspeed" in data.columns and additional_wind_features:
        data["windspeed_100mExtrapolated"] = calculate_windspeed_at_hubheight(
            data["windspeed"]
        )

        data["windPowerFit_extrapolated"] = calculate_windturbine_power_output(
            data["windspeed_100mExtrapolated"]
        )

    # Do extra check
    if "windspeed_100m" in data.columns and additional_wind_features:
        data["windpowerFit_harm_arome"] = calculate_windturbine_power_output(
            data["windspeed_100m"].astype(float)
        )

    return data


def calculate_dni(radiation: pd.Series, pj: PredictionJobDataClass) -> pd.Series:
    """Calculate the direct normal irradiance (DNI).

    This function uses the predicted radiation and information derived from the location (obtained from pj)


    Args:
        radiation: predicted radiation including DatetimeIndex with right time-zone
        pj: PredictJob including information about the location (lat, lon)

    Returns:
        Direct normal irradiance (DNI).

    """
    loc = Location(pj.get("lat", DEFAULT_LAT), pj.get("lon", DEFAULT_LON), tz="utc")
    times = radiation.index

    # calculate data for loc(ation) at times with clear_sky, as if there would be a clear sky.
    cs = loc.get_clearsky(times)

    # get solar position variable(s) for loc(ation) at times
    solpos = pvlib.solarposition.get_solarposition(times, loc.latitude, loc.longitude)
    solar_zenith = solpos.apparent_zenith

    # convert radiation (ghi) to right unit (J/m^2 to kWh/m^2)
    ghi_forecasted = radiation / 3600
    # convert ghi to dni
    dni_converted = pvlib.irradiance.dni(
        ghi_forecasted, cs.dhi, solar_zenith, clearsky_dni=cs.dni
    )
    dni_converted = dni_converted.fillna(0)
    return dni_converted


def calculate_gti(
    radiation: pd.Series,
    pj: PredictionJobDataClass,
    surface_tilt: float = 34.0,
    surface_azimuth: float = 180,
) -> pd.Series:
    """Calculate the GTI/POA using the radiation.

    This function assumes Global Tilted Irradiance (GTI) = Plane of Array (POA)

    Args:
        radiation: pandas series with DatetimeIndex with right timezone information
        pj: prediction job which should at least contain the latitude and longitude location.
        surface_tilt: The tilt of the surface of, for example, your PhotoVoltaic-system.
        surface_azimuth: The way the surface is facing. South facing 180 degrees, North facing 0 degrees, East facing 90 degrees and West facing 270 degrees

    Returns:
        Global Tilted Irradiance (GTI)

    """
    loc = Location(pj.get("lat", DEFAULT_LAT), pj.get("lon", DEFAULT_LON), tz="utc")
    times = radiation.index

    # calculate data for loc(ation) at times with clear_sky, as if there would be a clear sky.
    cs = loc.get_clearsky(times)
    dni = calculate_dni(radiation, pj)

    # get solar position variable(s) for loc(ation) at times
    solpos = pvlib.solarposition.get_solarposition(times, loc.latitude, loc.longitude)
    solar_zenith = solpos.apparent_zenith
    solar_azimuth = solpos.azimuth

    ghi_forecasted = radiation / 3600
    gti = pvlib.irradiance.get_total_irradiance(
        surface_tilt,
        surface_azimuth,
        solar_zenith,
        solar_azimuth,
        dni=dni,
        ghi=ghi_forecasted,
        dhi=cs.dhi,
    )

    return gti["poa_global"]


def add_additional_solar_features(
    data: pd.DataFrame,
    pj: PredictionJobDataClass = None,
    feature_names: list[str] = None,
) -> pd.DataFrame:
    """Adds additional solar features to the input data.

    Args:
        data: Dataframe to which the solar features have to be added
        pj: prediction job which should at least contain the latitude and longitude location.
        feature_names: List of requested features

    Returns:
        DataFrame same as input dataframe with extra columns for the added solar features

    """
    # If pj is none add solar features with Utrecht as default location
    if pj is None:
        logger.info(
            "No prediction job, default location will be used for additional radiation features."
        )
        pj = {"lat": DEFAULT_LAT, "lon": DEFAULT_LON}

    # If features is none add solar feature anyway
    if feature_names is None:
        additional_solar_features = True

    # Otherwise check if they are among the requested features
    else:
        additional_solar_features = any(x in ["dni", "gti"] for x in feature_names)

    # Add add_additional_solar_features
    if "radiation" in data.columns and additional_solar_features:
        data["dni"] = calculate_dni(data["radiation"], pj)
        data["gti"] = calculate_gti(data["radiation"], pj)

    return data
