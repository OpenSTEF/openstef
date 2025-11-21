# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for adding radiation derived features to time series data.

This module provides functionality to compute features derived from radiation data
to enhance time series datasets with additional insights related to solar radiation.
"""

import logging
from typing import Literal, cast, override

import pandas as pd
from pydantic import Field, PrivateAttr
from pydantic_extra_types.coordinate import Coordinate

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import MissingExtraError, TimeSeriesValidationError
from openstef_core.transforms import TimeSeriesTransform

logger = logging.getLogger(__name__)


class RadiationDerivedFeaturesAdder(BaseConfig, TimeSeriesTransform):
    """Transform that adds radiation derived features to time series data.

    Computes features that are derived from radiation data (in J/m²) based on geographical coordinates
    (latitude and longitude) and solar position.
    The features added can include:
        - dni: Direct Normal Irradiance (DNI) in kWh/m².
        - gti: Global Tilted Irradiance (GTI) in kWh/m² on a tilted surface.

    Note:
        The input radiation data must be in J/m² units. The transform will automatically
        convert this to kWh/m² for internal calculations.

    Example:
        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> from openstef_models.transforms.weather_domain import (
        ...     RadiationDerivedFeaturesAdder,
        ... )
        >>> from pydantic_extra_types.coordinate import Coordinate, Latitude, Longitude
        >>>
        >>> # Create sample dataset with radiation data in J/m²
        >>> data = pd.DataFrame({
        ...     'radiation': [3600000, 7200000, 5400000]  # Corresponds to 1, 2, and 1.5 kWh/m²
        ... }, index=pd.date_range('2025-06-01', periods=3, freq='D', tz='Europe/Amsterdam'))
        >>> dataset = TimeSeriesDataset(data, sample_interval=timedelta(minutes=15))
        >>>
        >>> # Initialize and apply transform
        >>> transform = RadiationDerivedFeaturesAdder(
        ...     coordinate=Coordinate(latitude=Latitude(52.0), longitude=Longitude(5.0)),
        ...     included_features=['dni', 'gti'],
        ...     surface_tilt=34.0,
        ...     surface_azimuth=180.0
        ... )
        >>> transformed_dataset = transform.transform(dataset)
        >>> transformed_dataset.feature_names
        ['radiation', 'dni', 'gti']
        >>> len(transformed_dataset.data["dni"]) == 3  # Check we have 3 values
        True
        >>> len(transformed_dataset.data["gti"]) == 3  # Check we have 3 values
        True
    """

    included_features: list[Literal["dni", "gti"]] = Field(
        default_factory=lambda: [
            "dni",
            "gti",
        ],
        description="List of radiation derived features to include.",
        min_length=1,
    )
    coordinate: Coordinate = Field(
        description="Geographical coordinates (latitude and longitude) for solar calculations.",
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
    radiation_column: str = Field(
        default="radiation",
        description="Name of the column in the dataset containing radiation data in J/m².",
    )

    _logger: logging.Logger = PrivateAttr(default=logging.getLogger(__name__))

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        try:
            import pvlib  # noqa: PLC0415 - delayed import due to optional dependency
        except ImportError as e:
            raise MissingExtraError("pvlib", package="openstef-models") from e

        if not data.index.tz:
            raise TimeSeriesValidationError("The datetime index must be timezone-aware.")

        if self.radiation_column not in data.feature_names:
            self._logger.info(
                "Radiation column '%s' not found in dataset features: %s",
                self.radiation_column,
                data.feature_names,
            )
            return data

        # Convert radiation from J/m² to kWh/m² and rename to 'ghi'
        ghi = (data.data[self.radiation_column] / 3600).rename("ghi")

        location = pvlib.location.Location(
            latitude=self.coordinate.latitude,
            longitude=self.coordinate.longitude,
            tz=str(data.index.tz),
        )

        solar_position: pd.DataFrame = pvlib.solarposition.get_solarposition(  # type: ignore[reportUnknownMemberType]
            time=data.index,
            latitude=location.latitude,
            longitude=location.longitude,
        )

        clearsky_radiation: pd.DataFrame = location.get_clearsky(data.index)  # type: ignore[reportUnknownMemberType]

        dni = cast(
            pd.Series,
            pvlib.irradiance.dni(  # type: ignore[reportUnknownMemberType]
                ghi=ghi,
                dhi=clearsky_radiation["dhi"],
                zenith=solar_position["apparent_zenith"],
                dni_clear=clearsky_radiation["dni"],
            ),
        ).fillna(0.0)

        result = data.data.copy(deep=False)
        if "dni" in self.included_features:
            result["dni"] = dni

        if "gti" in self.included_features:
            result["gti"] = pvlib.irradiance.get_total_irradiance(  # type: ignore[reportUnknownMemberType]
                surface_tilt=self.surface_tilt,
                surface_azimuth=self.surface_azimuth,
                solar_zenith=solar_position["apparent_zenith"],
                solar_azimuth=solar_position["azimuth"],
                dni=dni,
                ghi=ghi,
                dhi=clearsky_radiation["dhi"],
            )["poa_global"]

        return data.copy_with(result)

    @override
    def features_added(self) -> list[str]:
        return list(self.included_features)
