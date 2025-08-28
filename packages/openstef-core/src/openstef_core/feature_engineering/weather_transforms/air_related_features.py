# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from typing import Any, Literal

import pandas as pd
from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.transforms import TimeSeriesTransform

type AirRelatedFeatureName = Literal["saturation_pressure", "vapour_pressure", "dewpoint", "air_density"]


class AirRelatedFeatures(BaseConfig, TimeSeriesTransform):
    included_features: list[AirRelatedFeatureName] = Field(
        default_factory=lambda: ["saturation_pressure", "vapour_pressure", "dewpoint", "air_density"],
        description="List of air related features to include.",
    )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the AirRelatedFeatures transform."""
        super().__init__(**kwargs)
        self._air_related_features: pd.DataFrame = pd.DataFrame()

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
        return relative_humidity * water_vapour_pressure

    def _calculate_dewpoint(self, vapour_pressure: pd.Series) -> pd.Series:
        pass

    def _calculate_air_density(
        self, temperature: pd.Series, pressure: pd.Series, relative_humidity: pd.Series
    ) -> pd.Series:
        pass

    def fit(self, data: TimeSeriesDataset) -> None:
        pass

    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        pass
