from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Global app settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    post_teams_messages = True
