# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from enum import Enum
from app_settings import AppSettings


# TODO replace this with ModelType (MLModelType == Machine Learning model type)
class MLModelType(Enum):
    XGB = "xgb"
    XGB_QUANTILE = "xgb_quantile"
    LGB = "lgb"
    LINEAR = "linear"
    ProLoaf = "proloaf"
    ARIMA = "arima"


class ForecastType(Enum):
    DEMAND = "demand"
    WIND = "wind"
    SOLAR = "solar"
    BASECASE = "basecase"


class TracyJobResult(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    UNKNOWN = "unknown"


class PipelineType(Enum):
    FORECAST = "forecast"
    TRAIN = "train"
    HYPER_PARMATERS = "hyper_parameters"


class WeatherColumnName(Enum):
    TEMPERATURE = AppSettings.weather_column_name_temperature
    RADIATION = AppSettings.weather_column_name_radiation
    WINDSPEED = AppSettings.weather_column_name_windspeed 
    WINDSPEED_100M = AppSettings.weather_column_name_windspeed_100m
    WINDSPEED_100M_EXTRAPOLATED = AppSettings.weather_column_name_windspeed_100m_extrapolated
    SATURATION_PRESSURE = AppSettings.weather_column_name_saturation_pressure
    VAPOUR_PRESSURE = AppSettings.weather_column_name_vapour_pressure
    DEWPOINT = AppSettings.weather_column_name_dewpoint
    AIR_DENSITY = AppSettings.weather_column_name_air_density
    HUMIDITY = AppSettings.weather_column_name_humidity
    PRESSURE = AppSettings.weather_column_name_pressure
    WIND_EXTRAPOLATED = AppSettings.weather_column_name_wind_power_fit_extrapolated
    WIND_HARM_AROME = AppSettings.weather_column_name_wind_power_fit_harm_arome


class LocationColumnName(Enum):
    LAT = AppSettings.location_column_name_latitude
    LON = AppSettings.location_column_name_longitude
    