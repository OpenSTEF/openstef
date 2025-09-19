# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for calculating wind power features from wind speed data.

The transform computes wind speed at hub height and wind power output
based on wind speed data from measurements, forecasts, or model outputs.
"""

from typing import override

import numpy as np
import pandas as pd
from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.timeseries_transform import TimeSeriesTransform
from openstef_core.exceptions import MissingColumnsError


class WindPowerTransform(BaseConfig, TimeSeriesTransform):
    """WindPowerTransform computes wind power from wind speed data.

    This transform calculates wind speed at the wind turbine hub height using
    the wind profile power law, and estimates wind power output via a parameterized power curve.
    It can utilize either wind speed at hub height (if available) or extrapolate
    from wind speed at a reference height. The input wind speed can come from measurements,
    weather forecasts, or numerical weather model outputs. The resulting wind power feature
    can significantly improve forecast accuracy, especially for locations with substantial wind resources.

    Example:
    >>> import pandas as pd
    >>> from datetime import timedelta
    >>> from openstef_core.datasets import TimeSeriesDataset
    >>> from openstef_models.transforms.energy_domain import WindPowerTransform
    >>>
    >>> # Create sample dataset
    >>> df = pd.DataFrame({
    ...     "windspeed": [5.0, 6.0, 7.0, 8.0, 9.0]
    ... }, index=pd.date_range('2025-01-01', periods=5, freq='1h'))
    >>> dataset = TimeSeriesDataset(df, timedelta(hours=1))
    >>> transform = WindPowerTransform()
    >>> transformed_dataset = transform.transform(dataset)
    >>> transformed_dataset.feature_names
    ['windspeed', 'windspeed_hub_height', 'wind_power']
    """

    windspeed_reference_column: str = Field(
        default="windspeed",
        description="Column containing wind speed at reference height (from measurements or forecasts).",
    )
    reference_height: float = Field(
        default=10.0,
        description="Height (in meters) at which the reference wind speed is provided.",
    )
    windspeed_hub_height_column: str = Field(
        default="windspeed_hub_height",
        description="Column representing wind speed at hub height.",
    )
    hub_height: float = Field(
        default=100.0,
        description="Height of the wind turbine hub.",
    )
    rated_power: float = Field(
        default=1.0,
        description="Rated power for wind power calculation, normalized to 1MWp.",
    )
    steepness: float = Field(
        default=0.664,
        description="Steepness parameter for the power curve.",
    )
    slope_center: float = Field(
        default=8.07,
        description="Slope center parameter for the power curve.",
    )

    def _calculate_wind_speed_at_hub_height(self, wind_speed: pd.Series) -> pd.Series:
        """Calculates wind speed at hub height based on wind speed at reference height.

        Args:
            wind_speed: Wind speed at the reference height.

        Returns:
            A series of wind speed values at hub height.

        Reference:
        https://en.wikipedia.org/wiki/Wind_profile_power_law
        """
        alpha = 0.143
        return wind_speed * (self.hub_height / self.reference_height) ** alpha

    def _calculate_wind_power(self, wind_speed_hub_height: pd.Series) -> pd.Series:
        """Calculates wind power from wind speed at hub height.

        Values are related through the power curve, which is described by the rated power, steepness and slope center.
        Default values are used and are normalized to 1MWp.

        Args:
            wind_speed_hub_height: Wind speed at hub height.

        Returns:
            A series of wind power values in Watt.
        """
        generated_power = self.rated_power / (1 + np.exp(-self.steepness * (wind_speed_hub_height - self.slope_center)))
        return pd.Series(generated_power, index=wind_speed_hub_height.index)

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        if self.windspeed_reference_column not in data.feature_names:
            raise MissingColumnsError([self.windspeed_reference_column])

        features = pd.DataFrame(index=data.data.index)
        if self.windspeed_hub_height_column not in data.feature_names:
            features[self.windspeed_hub_height_column] = self._calculate_wind_speed_at_hub_height(
                data.data[self.windspeed_reference_column]
            )
            features["wind_power"] = self._calculate_wind_power(features[self.windspeed_hub_height_column])
        else:
            features["wind_power"] = self._calculate_wind_power(data.data[self.windspeed_hub_height_column])

        return TimeSeriesDataset(data=pd.concat([data.data, features], axis=1), sample_interval=data.sample_interval)
