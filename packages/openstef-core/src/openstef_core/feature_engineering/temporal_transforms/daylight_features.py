# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for extracting daylight-based features from time series data.

This module provides functionality to compute the daylight feature (terrestrial radiation)
based on geographical location and datetime indices
of time series datasets.
"""
from typing import cast

import pandas as pd

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.transforms import TimeSeriesTransform

try:
    import pvlib
except ImportError as e:
    raise ImportError(
        "pvlib is required for the DaylightFeatures transform. Please install it via "
        "`uv sync --group pvlib --package openstef-core` or `uv sync --all-groups --package openstef-core`."
    ) from e


class DaylightFeatures(TimeSeriesTransform):
    """Transform that adds daylight features to time series data.

    Computes features that indicate the amount of daylight based on
    geographical coordinates (latitude and longitude).

    Example:
        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> from openstef_core.feature_engineering.temporal_transforms.daylight_features import (
        ...     DaylightFeatures
        ... )
        >>> # Create sample dataset
        >>> data = pd.DataFrame({
        ...     'load': [100, 120, 110]
        ... }, index=pd.date_range('2025-06-01', periods=3, freq='D', tz='Europe/Amsterdam'))
        >>> dataset = TimeSeriesDataset(data, sample_interval=timedelta(minutes=15))
        >>> transform = DaylightFeatures(latitude=52.0, longitude=5.0)
        >>> transformed_dataset = transform.fit_transform(dataset)
        >>> transformed_dataset.feature_names
        ['load', 'daylight_continuous']
        >>> transformed_dataset.data["daylight_continuous"].tolist()  # 0 because of nighttime
        [0.0, 0.0, 0.0]
    """

    def __init__(self, latitude: float, longitude: float) -> None:
        """Initialize the transform with geographical coordinates.

        Args:
            latitude: Geographical latitude.
            longitude: Geographical longitude.
        """
        self.latitude = latitude
        self.longitude = longitude
        self.daylight_continuous: pd.DataFrame = pd.DataFrame()

    def fit(self, data: TimeSeriesDataset) -> None:
        """Fit the transform to the given dataset by computing daylight features.

        Args:
            data: Time series dataset to fit.

        Raises:
            ValueError: If the datetime index is not timezone-aware.
        """
        if not data.index.tz:
            raise ValueError("The datetime index must be timezone-aware.")
        location = pvlib.location.Location(self.latitude, self.longitude, tz=str(data.index.tz))
        clearsky_radiation: pd.DataFrame = location.get_clearsky(data.index)  # type: ignore[reportUnknownMemberType]
        self.daylight_continuous = clearsky_radiation[["ghi"]].rename(columns={"ghi": "daylight_continuous"})

    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        """Transform the given dataset by adding daylight features.

        Args:
            data: Time series dataset to transform.

        Returns:
            Transformed time series dataset with added daylight features.
        """
        return TimeSeriesDataset(
            data=pd.concat(
                [data.data, self.daylight_continuous],
                axis=1,
            ),
            sample_interval=data.sample_interval,
        )
