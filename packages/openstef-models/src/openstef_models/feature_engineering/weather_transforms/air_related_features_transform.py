# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from typing import Any, Literal

import numpy as np
from openstef_core.exceptions import MissingColumnsError
from openstef_core.types import override
import pandas as pd
from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.transforms import TimeSeriesTransform

type AirRelatedFeatureName = Literal["saturation_vapour_pressure", "vapour_pressure", "dewpoint", "air_density"]


class AirRelatedFeaturesTransform(BaseConfig, TimeSeriesTransform):
    included_features: list[AirRelatedFeatureName] = Field(
        default_factory=lambda: ["saturation_vapour_pressure", "vapour_pressure", "dewpoint", "air_density"],
        description="List of air related features to include.",
    )
    temperature_column: str = Field(
        default="temperature",
        description="Name of the temperature (Celsius) column.",
    )
    pressure_column: str = Field(
        default="pressure",
        description="Name of the pressure (hPa) column.",
    )
    relative_humidity_column: str = Field(
        default="relative_humidity",
        description="Name of the relative humidity (%) column.",
    )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the AirRelatedFeaturesTransform."""
        super().__init__(**kwargs)
        self._air_related_features: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def _calculate_saturation_vapour_pressure(temperature: pd.Series) -> pd.Series:
        """Calculate saturation vapour pressure of water using the Buck equation.

        Args:
            temperature: Air temperature in degrees Celsius.

        Returns:
            Vapour pressure of water in Pa.

        References:
            https://en.wikipedia.org/wiki/Vapour_pressure_of_water
        """
        # Set some (nameless) constants for the Buck equation:
        A: float = 0.61121
        B: float = 18.678
        C: float = 234.5
        D: float = 257.14
        K: int = 1000  # Conversion factor from kPa to Pa
        return pd.Series(A * np.exp((B - temperature / C) * (temperature / (D + temperature))) * K)

    def _calculate_vapour_pressure(self, temperature: pd.Series, relative_humidity: pd.Series) -> pd.Series:
        saturation_vapour_pressure = self._calculate_saturation_vapour_pressure(temperature)
        return relative_humidity * saturation_vapour_pressure

    @staticmethod
    def _calculate_dewpoint(temperature: pd.Series, relative_humidity: pd.Series) -> pd.Series:
        """
        Calculate the dew point using the Magnus Formula

        Args:
            relative_humidity: Relative humidity in %.
            temperature: Air temperature in degrees Celsius.

        Returns:
            Dew point in degrees Celsius.

        References:
        https://en.wikipedia.org/wiki/Dew_point
        """
        C: float = 243.04
        B: float = 17.625
        P: int = 100  # Conversion factor from % to fraction

        # Convert percentage to fraction
        relative_humidity = relative_humidity / P

        gamma = np.log(relative_humidity) + (B * temperature) / (C + temperature)
        return pd.Series(C * gamma / (B - gamma))
    
    def _calculate_air_density(self,
        temperature: pd.Series,  relative_humidity: pd.Series, pressure: pd.Series
    ) -> pd.Series:
        """
        Calculate the air density of humid air.

        Args:
            temperature: Air temperature in degrees Celsius.
            pressure: Air pressure in hPa.
            vapour_pressure: Water vapour pressure in Pa.

        Returns:
            Air density in kg/m^3.

        References:
        https://en.wikipedia.org/wiki/Density_of_air
        """
        R: float = 8.31446 # J/(K·mol)
        M_d: float = 0.0289652 # kg/mol
        M_v: float = 0.018016 # kg/mol
        K: float = 273.15 # To convert Celsius to Kelvin
        
        pressure = pressure * 100  # Convert hPa to Pa

        vapour_pressure = self._calculate_vapour_pressure(temperature, relative_humidity)
        dry_pressure: pd.Series[Any] = pressure - vapour_pressure

        return (dry_pressure * M_d + vapour_pressure * M_v) / (R * (temperature + K))

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        missing_columns: list[str] = []
        for col in [self.temperature_column, self.pressure_column, self.relative_humidity_column]:
            if col not in data.feature_names:
                missing_columns.append(col)
        
        if missing_columns:
            raise MissingColumnsError(missing_columns)

        temperature = data.data[self.temperature_column]
        pressure = data.data[self.pressure_column]
        relative_humidity = data.data[self.relative_humidity_column]

        self._air_related_features = pd.DataFrame(index=data.data.index)
        if "saturation_vapour_pressure" in self.included_features:
            self._air_related_features["saturation_vapour_pressure"] = self._calculate_saturation_vapour_pressure(temperature)
        if "vapour_pressure" in self.included_features:
            self._air_related_features["vapour_pressure"] = self._calculate_vapour_pressure(temperature, relative_humidity)
        if "dewpoint" in self.included_features:
            self._air_related_features["dewpoint"] = self._calculate_dewpoint(temperature, relative_humidity)
        if "air_density" in self.included_features:
            self._air_related_features["air_density"] = self._calculate_air_density(temperature, relative_humidity, pressure)

        return TimeSeriesDataset(
            data=pd.concat([data.data, self._air_related_features], axis=1),
            sample_interval=data.sample_interval
        )