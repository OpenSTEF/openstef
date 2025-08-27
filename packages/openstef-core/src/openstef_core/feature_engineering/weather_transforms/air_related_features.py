# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from typing import Literal

import pandas as pd
from pydantic import Field

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.transforms import TimeSeriesTransform


class AirRelatedFeaturesConfig:
    """Configuration for AirRelatedFeatures transform."""
    included_features: list[
        Literal[
            "saturation_pressure",
            "vapour_pressure",
            "dewpoint",
            "air_density"
        ]
    ] = Field(
        default_factory=lambda: [
            "saturation_pressure",
            "vapour_pressure",
            "dewpoint",
            "air_density"
        ],
        description="List of air related features to include.",
    )


class AirRelatedFeatures(TimeSeriesTransform):

    def __init__(self, config: AirRelatedFeaturesConfig) -> None:
        """Initialize the AirRelatedFeatures transform."""
        self.config = config
        self.air_related_features: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def _calculate_water_vapour_pressure(temperature: pd.Series) -> pd.Series:
        """Calculate vapour pressure of water using the Antoine equation.

        Args:
            temperature: Air temperature in degrees Celsius.

        Returns:
            Vapour pressure of water in hPa.

        References:
            https://en.wikipedia.org/wiki/Vapour_pressure_of_water
        """
        # Set some (nameless) constants for the Antoine equation:
        A: float = 8.07131
        B: float = 1730.63
        C: float = 233.426
        return 10 ** (A - (B / (temperature + C))) * 10

    @staticmethod
    def _calculate_vapour_pressure(relative_humidity: pd.Series, water_vapour_pressure: pd.Series) -> pd.Series:
        """Calculate vapour pressure.

        Args:
            relative_humidity: Relative humidity as fraction [0, 1].
            water_vapour_pressure: Water vapour pressure in hPa.

        Returns:
            Vapour pressure in hPa.
        """
        return relative_humidity * water_vapour_pressure

    def _calculate_dewpoint(self, vapour_pressure: pd.Series) -> pd.Series:
        """Calculate dewpoint temperature.

        Args:
            vapour_pressure: Vapour pressure in hPa.

        Returns:
            Dewpoint temperature in degrees Celsius.
        """
        A =
        M =
        TN =
        return

    def _calculate_air_density(self, temperature: pd.Series, pressure: pd.Series, relative_humidity: pd.Series) -> pd.Series:
        """Calculate air density.

        Args:
            temperature: Air temperature in degrees Celsius.
            pressure: Atmospheric pressure in hPa.
            relative_humidity: Relative humidity in percentage (0-100).

        Returns:
            Air density in kg/m³.
        """
        import numpy as np
        R_d = 287.05
        R_v = 461.495
        T_k = temperature + 273.15
        e = self._calculate_vapour_pressure(relative_humidity, self._calulate_saturation_pressure(temperature))
        p = pressure * 100
        return (p - e * 100) / (R_d * T_k) + \
                (e * 100) / (R_v * T_k)

    def fit(self, data: TimeSeriesDataset) -> None:
        pass

    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        pass
