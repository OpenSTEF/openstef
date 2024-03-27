from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    """Global app settings"""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Weather settings
    weather_temperature = "temp"
    weather_radiation = "radiation"
    weather_windspeed_100m = "windspeed_100m"
    weather_saturation_pressure = "saturation_pressure"
    weather_vapour_pressure = "vapour_pressure"
    weather_dewpoint = "dewpoint"
    weather_air_density = "air_density"
    weather_humidity = "humidity"
    weather_pressure = "pressure"
    