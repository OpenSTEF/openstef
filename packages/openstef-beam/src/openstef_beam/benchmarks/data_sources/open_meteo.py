from datetime import timedelta
from enum import StrEnum
from logging import Logger
import logging
from typing import Any, Literal, cast, override

import pandas as pd
from pydantic import Field, PrivateAttr, SecretStr
from pydantic_extra_types.coordinate import Coordinate

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.exceptions import MissingExtraError
from openstef_core.types import DatetimeRange, LeadTime

try:
    import niquests
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry  # pyright: ignore[reportUnknownVariableType]
except ImportError as e:
    raise MissingExtraError(extra="all", package="openstef-beam") from e


type OpenMeteoFeature = Literal[
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation_probability",
    "precipitation",
    "rain",
    "showers",
    "snowfall",
    "snow_depth",
    "weather_code",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "visibility",
    "evapotranspiration",
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_speed_120m",
    "wind_speed_180m",
    "wind_direction_10m",
    "wind_direction_80m",
    "wind_direction_120m",
    "wind_direction_180m",
    "wind_gusts_10m",
    "temperature_80m",
    "temperature_120m",
    "temperature_180m",
    "soil_temperature_0cm",
    "soil_temperature_6cm",
    "soil_temperature_18cm",
    "soil_temperature_54cm",
    "soil_moisture_0_to_1cm",
    "soil_moisture_1_to_3cm",
    "soil_moisture_3_to_9cm",
    "soil_moisture_9_to_27cm",
    "soil_moisture_27_to_81cm",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "global_tilted_irradiance",
    "terrestrial_radiation",
    "shortwave_radiation_instant",
    "direct_radiation_instant",
    "diffuse_radiation_instant",
    "direct_normal_irradiance_instant",
    "global_tilted_irradiance_instant",
    "terrestrial_radiation_instant",
]

_DEFAULT_FEATURES: list[OpenMeteoFeature] = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_direction_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
]


type OpenMeteoFeatureVersioned = Literal[
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "showers",
    "snowfall",
    "weather_code",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "global_tilted_irradiance",
    "shortwave_radiation_instant",
    "direct_radiation_instant",
    "diffuse_radiation_instant",
    "direct_normal_irradiance_instant",
    "global_tilted_irradiance_instant",
    "terrestrial_radiation_instant",
]


_DEFAULT_FEATURES_VERSIONED: list[OpenMeteoFeatureVersioned] = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
]


class OpenMeteoVersions(StrEnum):
    DAY0 = "day0"
    DAY1 = "day1"
    DAY2 = "day2"
    DAY3 = "day3"
    DAY4 = "day4"
    DAY5 = "day5"
    DAY6 = "day6"
    DAY7 = "day7"

    def to_lead_time(self) -> LeadTime:
        return LeadTime(value=timedelta(days=int(self.name[-1])))


_DEFAULT_VERSIONS: list[OpenMeteoVersions] = [
    OpenMeteoVersions.DAY0,
    OpenMeteoVersions.DAY1,
    OpenMeteoVersions.DAY2,
    OpenMeteoVersions.DAY3,
    OpenMeteoVersions.DAY4,
    OpenMeteoVersions.DAY5,
]


class OpenMeteoDataRepository(BaseConfig):
    api_key: SecretStr | None = Field(
        default=None, description="API key for Open-Meteo. If not provided, free tier will be used."
    )

    forecast_previous_runs_url: str = Field(default="https://previous-runs-api.open-meteo.com/v1/forecast")
    forecast_historical_url: str = Field(default="https://historical-forecast-api.open-meteo.com/v1/forecast")

    _client: openmeteo_requests.Client = PrivateAttr()
    _logger: Logger = PrivateAttr(default_factory=lambda: logging.getLogger(__name__))

    @override
    def model_post_init(self, context: Any) -> None:
        cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
        retry_session = cast(niquests.Session, retry(cache_session, retries=5, backoff_factor=0.2))
        self._client = openmeteo_requests.Client(session=retry_session)

    def fetch_weather_data_versioned(
        self,
        coordinate: Coordinate,
        range: DatetimeRange,
        features: list[OpenMeteoFeatureVersioned] = _DEFAULT_FEATURES_VERSIONED,
        versions: list[OpenMeteoVersions] = _DEFAULT_VERSIONS,
    ) -> VersionedTimeSeriesDataset:
        request_features = [f"{feature}_{version}" for feature in features for version in versions]

        responses = self._client.weather_api(  # pyright: ignore[reportUnknownMemberType]
            url=self.forecast_previous_runs_url,
            params={
                "latitude": coordinate.latitude,
                "longitude": coordinate.longitude,
                "hourly": request_features,
                "start_date": range.start.date().isoformat(),
                "end_date": range.end.date().isoformat(),
                "apikey": self.api_key.get_secret_value() if self.api_key else None,
            },
        )

        if len(responses) != 1:
            msg = f"Expected a single response, got {len(responses)}"
            raise RuntimeError(msg)

        response = responses[0]
        hourly = response.Hourly()
        if hourly is None:
            raise RuntimeError("No hourly data in response")

        index = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
            name="timestamp",
        )
        split_feature_data = {
            feature: pd.Series(
                data=not_none(hourly.Variables(i)).ValuesAsNumpy(),
                index=index,
                name=feature,
            )
            for i, feature in enumerate(request_features)
        }

        versioned_data = pd.concat(
            [
                pd.DataFrame(
                    data={
                        str(feature): split_feature_data[f"{feature}_{version}"]
                        for feature in features
                        if f"{feature}_{version}" in split_feature_data
                    }
                    | {
                        "timestamp": index,
                        "available_at": index - OpenMeteoVersions.to_lead_time(version).value,
                    },
                    index=index,
                ).reset_index(drop=True)
                for version in versions
            ],
            axis=0,
        )

        return VersionedTimeSeriesDataset.from_dataframe(
            data=versioned_data,
            sample_interval=timedelta(hours=1),
            timestamp_column="timestamp",
            available_at_column="available_at",
        )

    def fetch_weather_data(
        self,
        coordinate: Coordinate,
        range: DatetimeRange,
        features: list[OpenMeteoFeature] = _DEFAULT_FEATURES,
    ) -> TimeSeriesDataset:
        responses = self._client.weather_api(  # pyright: ignore[reportUnknownMemberType]
            url=self.forecast_historical_url,
            params={
                "latitude": coordinate.latitude,
                "longitude": coordinate.longitude,
                "hourly": features,
                "start_date": range.start.date().isoformat(),
                "end_date": range.end.date().isoformat(),
                "apikey": self.api_key.get_secret_value() if self.api_key else None,
            },
        )

        if len(responses) != 1:
            msg = f"Expected a single response, got {len(responses)}"
            raise RuntimeError(msg)

        response = responses[0]
        hourly = response.Hourly()
        if hourly is None:
            raise RuntimeError("No hourly data in response")

        index = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
            name="timestamp",
        )

        data = pd.DataFrame(
            data={
                str(feature): pd.Series(
                    data=not_none(hourly.Variables(i)).ValuesAsNumpy(),
                    index=index,
                    name=feature,
                )
                for i, feature in enumerate(features)
            },
            index=index,
        )

        return TimeSeriesDataset(
            data=data,
            sample_interval=timedelta(hours=1),
        )


def not_none[T](value: T | None) -> T:
    if value is None:
        raise ValueError("Unexpected None value")
    return value


def resample_predictors(df: pd.DataFrame, sample_interval: timedelta) -> pd.DataFrame:
    """Resample predictor data to the desired sample interval using linear interpolation."""
    return df.resample(sample_interval).interpolate(method="linear")
