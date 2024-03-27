# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Global app settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Weather column names
    weather_column_name_temperature: str = "temp"
    weather_column_name_radiation: str = "radiation"
    weather_column_name_windspeed: str = "windspeed"
    weather_column_name_windspeed_100m: str = "windspeed_100m"
    weather_column_name_windspeed_100m_extrapolated: str = "windspeed_100mExtrapolated"
    weather_column_name_saturation_pressure: str = "saturation_pressure"
    weather_column_name_vapour_pressure: str = "vapour_pressure"
    weather_column_name_dewpoint: str = "dewpoint"
    weather_column_name_air_density: str = "air_density"
    weather_column_name_humidity: str = "humidity"
    weather_column_name_pressure: str = "pressure"
    weather_column_name_wind_power_fit_extrapolated: str = "windPowerFit_extrapolated"
    weather_column_name_wind_power_fit_harm_arome: str = "windpowerFit_harm_arome"
    weather_column_name_wind_hub_height: str = "hub_height"

    weather_column_name_turbine_type: str = "turbine_type"
    weather_column_name_number_turbines: str = "n_turbines"

    location_column_name_latitude: str = "lat"
    location_column_name_longitude: str = "lon"

    forecast_column_name_pid: str = "pid"
    forecast_column_name_customer: str = "customer"
    forecast_column_name_description: str = "description"
    forecast_column_name_type: str = "type"
    forecast_column_name_general_type: str = "algtype"
    forecast_column_name_horizon_minutes: str = "horizon_minutes"

    # Logging settings.
    log_level: str = Field("INFO", description="Log level used for logging statements.")

    model_config = SettingsConfigDict(
        env_prefix="openstef_", env_file=".env", extra="ignore"
    )
