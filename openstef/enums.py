# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from enum import Enum
from openstef.settings import Settings


# TODO replace this with ModelType (MLModelType == Machine Learning model type)
class MLModelType(Enum):
    XGB = "xgb"
    XGB_QUANTILE = "xgb_quantile"
    LGB = "lgb"
    LINEAR = "linear"
    ARIMA = "arima"


class ForecastType(Enum):
    DEMAND = "demand"
    WIND = "wind"
    SOLAR = "solar"
    BASECASE = "basecase"


class PipelineType(Enum):
    FORECAST = "forecast"
    TRAIN = "train"
    HYPER_PARMATERS = "hyper_parameters"


class WeatherColumnName:
    TEMPERATURE = "temp"
    RADIATION = "radiation"
    WINDSPEED = "windspeed"
    WINDSPEED_100M = "windspeed_100m"
    WINDSPEED_100M_EXTRAPOLATED = "windspeed_100mExtrapolated"
    SATURATION_PRESSURE = "saturation_pressure"
    VAPOUR_PRESSURE = "vapour_pressure"
    DEWPOINT = "dewpoint"
    AIR_DENSITY = "air_density"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    WIND_EXTRAPOLATED = "windPowerFit_extrapolated"
    WIND_HARM_AROME = "windpowerFit_harm_arome"
    WIND_HUB_HEIGHT = "hub_height"
    TURBINE_TYPE = "turbine_type"
    NUMBER_TURBINES = "n_turbines"


class LocationColumnName:
    LAT = "lat"
    LON = "lon"


class ForecastColumnName:
    PID = "pid"
    CUSTOMER = "customer"
    DESCRIPTION = "description"
    TYPE = "type"
    GENERAL_TYPE = "algtype"
    HORIZON_MINUTES = "horizon_minutes"
