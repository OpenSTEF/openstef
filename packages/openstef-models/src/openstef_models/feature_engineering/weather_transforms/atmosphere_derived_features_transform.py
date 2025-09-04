# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for calculating and adding atmosphere derived meteorological features to a time series dataset.

The transform computes saturation vapour pressure, vapour pressure, dewpoint, and air density
based on temperature, pressure, and relative humidity columns using established physical equations.
"""

from typing import Any, Literal, override

import numpy as np
import pandas as pd
from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.transforms import TimeSeriesTransform
from openstef_core.exceptions import MissingColumnsError

type AirRelatedFeatureName = Literal["saturation_vapour_pressure", "vapour_pressure", "dewpoint", "air_density"]


class AtmosphereDerivedFeaturesTransform(BaseConfig, TimeSeriesTransform):
    """Transform that calculates atmosphere derived meteorological features from basic weather data.

    This transform calculates various air-related features including saturation vapour pressure,
    vapour pressure, dewpoint, and air density using standard meteorological formulas. It requires
    temperature, pressure, and relative humidity as input columns.
    The calculated features can be used to enhance weather-based prediction models by providing
    additional atmospheric state information that may correlate with energy generation patterns.
    For example: Higher humidity reduces PV generation by scattering and absorbing sunlight
    (https://doi.org/10.1016/j.matpr.2020.08.775).


    Example:
        >>> import pandas as pd
        >>> from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
        >>> from openstef_models.feature_engineering.weather_transforms.atmosphere_derived_features_transform import (
        ...     AtmosphereDerivedFeaturesTransform
        ... )
        >>>
        >>> # Create sample weather data
        >>> data = pd.DataFrame({
        ...     'temperature': [20.0, 25.0, 15.0],
        ...     'pressure': [1013.25, 1015.0, 1010.0],
        ...     'relative_humidity': [60.0, 70.0, 80.0]
        ... },
        ... index=pd.date_range('2025-06-01 12:00:00', periods=3, freq='h'))
        >>> dataset = TimeSeriesDataset(data=data, sample_interval=pd.Timedelta(hours=1))
        >>>
        >>> # Initialize transform with specific features
        >>> transform = AtmosphereDerivedFeaturesTransform(
        ...     included_features=["dewpoint", "air_density"]
        ... )
        >>>
        >>> # Apply transformation
        >>> result = transform.transform(dataset)
        >>> result.feature_names
        ['temperature', 'pressure', 'relative_humidity', 'dewpoint', 'air_density']
    """

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
        # Buck equation constants
        a: float = 0.61121
        b: float = 18.678
        c: float = 234.5
        d: float = 257.14

        # Calculate saturation vapor pressure and convert from kPa to Pa
        return pd.Series(a * np.exp((b - temperature / c) * (temperature / (d + temperature))) * 1000)

    def _calculate_vapour_pressure(self, temperature: pd.Series, relative_humidity: pd.Series) -> pd.Series:
        saturation_vapour_pressure = self._calculate_saturation_vapour_pressure(temperature)
        return relative_humidity * saturation_vapour_pressure

    @staticmethod
    def _calculate_dewpoint(temperature: pd.Series, relative_humidity: pd.Series) -> pd.Series:
        """Calculate the dew point using the Magnus Formula.

        Args:
            relative_humidity: Relative humidity in %.
            temperature: Air temperature in degrees Celsius.

        Returns:
            Dew point in degrees Celsius.

        References:
        https://en.wikipedia.org/wiki/Dew_point
        """
        c: float = 243.04
        b: float = 17.625

        # Convert percentage to fraction
        relative_humidity /= 100

        gamma = np.log(relative_humidity) + (b * temperature) / (c + temperature)
        return pd.Series(c * gamma / (b - gamma))

    def _calculate_air_density(
        self, temperature: pd.Series, relative_humidity: pd.Series, pressure: pd.Series
    ) -> pd.Series:
        """Calculate the air density of humid air.

        Args:
            temperature: Air temperature in degrees Celsius.
            relative_humidity: Relative humidity in %.
            pressure: Air pressure in hPa.
            vapour_pressure: Water vapour pressure in Pa.

        Returns:
            Air density in kg/m^3.

        References:
        https://en.wikipedia.org/wiki/Density_of_air
        """
        r: float = 8.31446  # J/(K·mol)
        m_d: float = 0.0289652  # kg/mol
        m_v: float = 0.018016  # kg/mol
        k: float = 273.15  # To convert Celsius to Kelvin

        pressure *= 100  # Convert hPa to Pa

        vapour_pressure = self._calculate_vapour_pressure(temperature, relative_humidity)
        dry_pressure: pd.Series[Any] = pressure - vapour_pressure

        return (dry_pressure * m_d + vapour_pressure * m_v) / (r * (temperature + k))

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        missing_columns: list[str] = [
            col
            for col in [self.temperature_column, self.pressure_column, self.relative_humidity_column]
            if col not in data.feature_names
        ]

        if missing_columns:
            raise MissingColumnsError(missing_columns)

        temperature = data.data[self.temperature_column]
        pressure = data.data[self.pressure_column]
        relative_humidity = data.data[self.relative_humidity_column]

        atmosphere_derived_features = pd.DataFrame(index=data.data.index)
        if "saturation_vapour_pressure" in self.included_features:
            atmosphere_derived_features["saturation_vapour_pressure"] = self._calculate_saturation_vapour_pressure(
                temperature
            )
        if "vapour_pressure" in self.included_features:
            atmosphere_derived_features["vapour_pressure"] = self._calculate_vapour_pressure(
                temperature, relative_humidity
            )
        if "dewpoint" in self.included_features:
            atmosphere_derived_features["dewpoint"] = self._calculate_dewpoint(temperature, relative_humidity)
        if "air_density" in self.included_features:
            atmosphere_derived_features["air_density"] = self._calculate_air_density(
                temperature, relative_humidity, pressure
            )

        return TimeSeriesDataset(
            data=pd.concat([data.data, atmosphere_derived_features], axis=1), sample_interval=data.sample_interval
        )
