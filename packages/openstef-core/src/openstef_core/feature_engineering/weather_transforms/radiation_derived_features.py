# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for adding radiation derived features to time series data.

This module provides functionality to compute features derived from radiation data
to enhance time series datasets with additional insights related to solar radiation.
"""

import logging
from typing import Any, Literal, cast

import pandas as pd
from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.transforms import TimeSeriesTransform

try:
    import pvlib
except ImportError as e:
    raise ImportError(
        "pvlib is required for the DaylightFeatures transform. Please install it via "
        "`uv sync --group pvlib --package openstef-core` or `uv sync --all-groups --package openstef-core`."
    ) from e

logger = logging.getLogger(__name__)


class RadiationDerivedFeatures(BaseConfig, TimeSeriesTransform):
    """Transform that adds radiation derived features to time series data.

    Computes features that are derived from radiation data based on geographical coordinates
    (latitude and longitude) and solar position.
    The features added can include:
        - dni: Direct Normal Irradiance (DNI) in kWh/m².
        - gti: Global Tilted Irradiance (GTI) in kWh/m² on a tilted surface.

    Example:
        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> from openstef_core.feature_engineering.weather_transforms.radiation_derived_features import (
        ...     RadiationDerivedFeatures,
        ... )
        >>>
        >>> # Create sample dataset with radiation data in J/m²
        >>> data = pd.DataFrame({
        ...     'radiation': [3600000, 7200000, 5400000]  # Corresponds to 1, 2, and 1.5 kWh/m²
        ... }, index=pd.date_range('2025-06-01', periods=3, freq='D', tz='Europe/Amsterdam'))
        >>> dataset = TimeSeriesDataset(data, sample_interval=timedelta(minutes=15))
        >>>
        >>> # Initialize and apply transform
        >>> transform = RadiationDerivedFeatures(
        ...     latitude=52.0,
        ...     longitude=5.0,
        ...     included_features=['dni', 'gti'],
        ...     surface_tilt=34.0,
        ...     surface_azimuth=180.0
        ... )
        >>> transformed_dataset = transform.fit_transform(dataset)
        >>> transformed_dataset.feature_names
        ['radiation', 'dni', 'gti']
        >>> transformed_dataset.data["dni"].round(2).tolist()
        [0.0, 0.0, 0.0]
        >>> transformed_dataset.data["gti"].round(2).tolist()
        [0.0, 0.0, 0.0]
    """

    included_features: list[Literal["dni", "gti"]] = Field(
        default_factory=lambda: [
            "dni",
            "gti",
        ],
        description="List of radiation derived features to include.",
    )
    latitude: float = Field(
        ...,
        description="Latitude of the location for solar calculations.",
        ge=-90.0,
        le=90.0,
    )
    longitude: float = Field(
        ...,
        description="Longitude of the location for solar calculations.",
        ge=-180.0,
        le=180.0,
    )
    surface_tilt: float = Field(
        default=34.0,
        description="Tilt angle of the surface in degrees. Default is 34 degrees.",
        ge=0.0,
        le=90.0,
    )
    surface_azimuth: float = Field(
        default=180.0,
        description="Azimuth angle of the surface in degrees. Default is 180 degrees (south-facing).",
        ge=0.0,
        le=360.0,
    )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the RadiationDerivedFeatures transform.

        Args:
            **kwargs: Configuration parameters for the transform.
        """
        super().__init__(**kwargs)
        self._derived_features: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def _check_feature_exists(data: TimeSeriesDataset, feature_name: str) -> bool:
        if feature_name not in data.feature_names:
            logger.warning("Skipping calculation because feature `%s` is missing.", feature_name)
            return False
        return True

    @staticmethod
    def _calculate_gti(
        solar_position: pd.DataFrame,
        dni: pd.Series,
        ghi: pd.Series,
        clearsky_radiation: pd.DataFrame,
        surface_tilt: float = 34.0,
        surface_azimuth: float = 180.0,
    ) -> pd.Series:
        return pvlib.irradiance.get_total_irradiance(  # type: ignore[reportUnknownMemberType]
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
            solar_zenith=solar_position["apparent_zenith"],
            solar_azimuth=solar_position["azimuth"],
            dni=dni,
            ghi=ghi,
            dhi=clearsky_radiation["dhi"],
        )["poa_global"]

    @staticmethod
    def _calculate_dni(
        solar_position: pd.DataFrame,
        clearsky_radiation: pd.DataFrame,
        ghi: pd.Series,
    ) -> pd.Series:
        """Calculate Direct Normal Irradiance (DNI) from Global Horizontal Irradiance (GHI).

        Args:
            solar_position: The solar position DataFrame containing 'apparent_zenith' and 'azimuth'.
            clearsky_radiation: The clearsky radiation DataFrame containing 'dhi' and 'dni'.
            ghi: Global Horizontal Irradiance (GHI) series in kWh/m².

        Returns:
            Series with Direct Normal Irradiance (DNI) in kWh/m².
        """
        return cast(
            pd.Series,
            pvlib.irradiance.dni(  # type: ignore[reportUnknownMemberType]
                ghi=ghi,
                dhi=clearsky_radiation["dhi"],
                zenith=solar_position["apparent_zenith"],
                dni_clear=clearsky_radiation["dni"],
            ),
        ).fillna(0.0)

    def fit(self, data: TimeSeriesDataset) -> None:
        """Fit the transform to the input time series data by calculating radiation derived features.

        Args:
            data: The time series dataset with a timezone-aware DatetimeIndex and radiation data in J/m².

        Raises:
            ValueError: If the DatetimeIndex is not timezone-aware.
        """
        if not data.index.tz:
            raise ValueError("The datetime index must be timezone-aware.")

        if "radiation" not in data.feature_names:
            self._derived_features = pd.DataFrame()
            logger.warning("Skipping calculation of radiation derived features because 'radiation' feature is missing.")
            return

        if "gti" not in self.included_features and "dni" not in self.included_features:
            self._derived_features = pd.DataFrame()
            logger.warning("No radiation derived features selected to include.")
            return

        radiation = data.data["radiation"]

        # Convert radiation from J/m² to kWh/m²
        ghi = radiation / 3600

        location = pvlib.location.Location(
            latitude=self.latitude,
            longitude=self.longitude,
            tz=str(data.index.tz),
        )

        solar_position: pd.DataFrame = pvlib.solarposition.get_solarposition(  # type: ignore[reportUnknownMemberType]
            time=data.index,
            latitude=location.latitude,
            longitude=location.longitude,
        )

        clearsky_radiation: pd.DataFrame = location.get_clearsky(data.index)  # type: ignore[reportUnknownMemberType]

        dni = self._calculate_dni(
            solar_position=solar_position,
            clearsky_radiation=clearsky_radiation,
            ghi=ghi,
        )

        gti = self._calculate_gti(
            solar_position=solar_position,
            clearsky_radiation=clearsky_radiation,
            ghi=ghi,
            dni=dni,
            surface_tilt=self.surface_tilt,
            surface_azimuth=self.surface_azimuth,
        )

        self._derived_features = pd.concat(
            [
                dni.rename("dni") if "dni" in self.included_features else pd.Series(dtype=float),
                gti.rename("gti") if "gti" in self.included_features else pd.Series(dtype=float),
            ],
            axis=1,
        )

    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        """Transform the input time series data by adding radiation derived features.

        Args:
            data: The input time series dataset to be transformed.

        Returns:
            A new instance of TimeSeriesDataset containing the original and new radiation derived features.
        """
        return TimeSeriesDataset(
            data=pd.concat(
                [data.data, self._derived_features],
                axis=1,
            ),
            sample_interval=data.sample_interval,
        )
