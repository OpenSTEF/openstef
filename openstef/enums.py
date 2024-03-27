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


class WeatherConstants(Enum):
    TEMPERATURE = AppSettings.weather_temperature
    RADIATION = AppSettings.weather_radiation
    WINDSPEED_100M = AppSettings.weather_windspeed_100m
    SATURATION_PRESSURE = AppSettings.weather_saturation_pressure
    VAPOUR_PRESSURE = AppSettings.weather_vapour_pressure
    DEWPOINT = AppSettings.weather_dewpoint
    AIR_DENSITY = AppSettings.weather_air_density
    HUMIDITY = AppSettings.weather_humidity
    PRESSURE = AppSettings.weather_pressure


class LocationColumnName(Enum):
    LAT = AppSettings.location_column_name_latitude
    LON = AppSettings.location_column_name_longitude
    