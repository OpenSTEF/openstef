from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    """Global app settings"""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Weather column names
    weather_column_name_temperature = "temp"
    weather_column_name_radiation = "radiation"
    weather_column_name_windspeed_100m = "windspeed_100m"
    weather_column_name_saturation_pressure = "saturation_pressure"
    weather_column_name_vapour_pressure = "vapour_pressure"
    weather_column_name_dewpoint = "dewpoint"
    weather_column_name_air_density = "air_density"
    weather_column_name_humidity = "humidity"
    weather_column_name_pressure = "pressure"

    location_column_name_latitude = "lat"
    location_column_name_longitude = "lon"
    