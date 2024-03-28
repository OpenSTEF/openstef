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


class WeatherColumnName(Enum):
    TEMPERATURE = Settings.weather_column_name_temperature
    RADIATION = Settings.weather_column_name_radiation
    WINDSPEED = Settings.weather_column_name_windspeed
    WINDSPEED_100M = Settings.weather_column_name_windspeed_100m
    WINDSPEED_100M_EXTRAPOLATED = (
        Settings.weather_column_name_windspeed_100m_extrapolated
    )
    SATURATION_PRESSURE = Settings.weather_column_name_saturation_pressure
    VAPOUR_PRESSURE = Settings.weather_column_name_vapour_pressure
    DEWPOINT = Settings.weather_column_name_dewpoint
    AIR_DENSITY = Settings.weather_column_name_air_density
    HUMIDITY = Settings.weather_column_name_humidity
    PRESSURE = Settings.weather_column_name_pressure
    WIND_EXTRAPOLATED = Settings.weather_column_name_wind_power_fit_extrapolated
    WIND_HARM_AROME = Settings.weather_column_name_wind_power_fit_harm_arome
    WIND_HUB_HEIGHT = Settings.weather_column_name_wind_hub_height
    TURBINE_TYPE = Settings.weather_column_name_turbine_type
    NUMBER_TURBINES = Settings.weather_column_name_number_turbines


class LocationColumnName(Enum):
    LAT = Settings.location_column_name_latitude
    LON = Settings.location_column_name_longitude


class ForecastColumnName(Enum):
    PID = Settings.forecast_column_name_pid
    CUSTOMER = Settings.forecast_column_name_customer
    DESCRIPTION = Settings.forecast_column_name_description
    TYPE = Settings.forecast_column_name_type
    GENERAL_TYPE = Settings.forecast_column_name_general_type
    HORIZON_MINUTES = Settings.forecast_column_name_horizon_minutes
