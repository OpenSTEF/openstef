from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Global app settings"""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Weather column names
    weather_column_name_temperature = "temp"
    weather_column_name_radiation = "radiation"
    weather_column_name_windspeed = "windspeed"
    weather_column_name_windspeed_100m = "windspeed_100m"
    weather_column_name_windspeed_100m_extrapolated = "windspeed_100mExtrapolated"
    weather_column_name_saturation_pressure = "saturation_pressure"
    weather_column_name_vapour_pressure = "vapour_pressure"
    weather_column_name_dewpoint = "dewpoint"
    weather_column_name_air_density = "air_density"
    weather_column_name_humidity = "humidity"
    weather_column_name_pressure = "pressure"
    weather_column_name_wind_power_fit_extrapolated = "windPowerFit_extrapolated"
    weather_column_name_wind_power_fit_harm_arome = "windpowerFit_harm_arome"
    weather_column_name_wind_hub_height = "hub_height"

    weather_column_name_turbine_type = "turbine_type"
    weather_column_name_number_turbines = "n_turbines"

    location_column_name_latitude = "lat"
    location_column_name_longitude = "lon"

    forecast_column_name_pid = "pid"
    forecast_column_name_customer = "customer"
    forecast_column_name_description = "description"
    forecast_column_name_type = "type"
    forecast_column_name_general_type = "algtype"
    forecast_column_name_horizon_minutes = "horizon_minutes"
