# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for extracting daylight-based features from time series data.

This module provides functionality to compute the daylight feature (terrestrial radiation)
based on geographical location and datetime indices
of time series datasets.
"""

from typing import Self, override

import pandas as pd
from pydantic import Field
from pydantic_extra_types.coordinate import Coordinate

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import MissingExtraError
from openstef_core.mixins import State
from openstef_core.transforms import TimeSeriesTransform


class DaylightFeatureAdder(BaseConfig, TimeSeriesTransform):
    """Transform that adds daylight features to time series data.

    Computes features that indicate the amount of daylight based on
    geographical coordinates (latitude and longitude).

    Example:
        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> from openstef_models.transforms.weather_domain import (
        ...     DaylightFeatureAdder,
        ... )
        >>>
        >>> # Create sample dataset with timezone
        >>> data = pd.DataFrame({
        ...     'load': [100, 120, 110]
        ... }, index=pd.date_range('2025-06-01 12:00:00', periods=3, freq='h', tz='Europe/Amsterdam'))
        >>> dataset = TimeSeriesDataset(data, timedelta(hours=1))
        >>> transform = DaylightFeatureAdder(coordinate=(52.0, 5.0))
        >>> transformed_dataset = transform.transform(dataset)
        >>> 'daylight_continuous' in transformed_dataset.data.columns
        True
    """

    coordinate: Coordinate = Field(
        description="Geographical coordinates (latitude and longitude) for daylight calculation."
    )

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        try:
            import pvlib  # noqa: PLC0415
        except ImportError as e:
            raise MissingExtraError("pvlib", package="openstef-models") from e

        location = pvlib.location.Location(self.coordinate.latitude, self.coordinate.longitude, tz=str(data.index.tz))
        clearsky_radiation: pd.DataFrame = location.get_clearsky(data.index)  # type: ignore[reportUnknownMemberType]
        daylight_continuous = clearsky_radiation[["ghi"]].rename(columns={"ghi": "daylight_continuous"})

        return TimeSeriesDataset(
            data=pd.concat(
                [data.data, daylight_continuous],
                axis=1,
            ),
            sample_interval=data.sample_interval,
        )

    @override
    def to_state(self) -> State:
        return self.model_dump(mode="json")

    @override
    def from_state(self, state: State) -> Self:
        return self.model_validate(state)

    @override
    def features_added(self) -> list[str]:
        return ["daylight_continuous"]
