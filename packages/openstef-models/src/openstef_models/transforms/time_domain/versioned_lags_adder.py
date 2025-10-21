# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Lag feature transforms for versioned time series data.

This module provides transforms for creating lag features while preserving
data availability constraints in versioned datasets. Essential for energy
forecasting where temporal dependencies matter but data availability varies.
"""

from datetime import timedelta
from typing import Self, cast, override

import pandas as pd
from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.versioned_timeseries_dataset import VersionedTimeSeriesDataset
from openstef_core.exceptions import MissingColumnsError
from openstef_core.mixins import State
from openstef_core.transforms import VersionedTimeSeriesTransform
from openstef_core.utils import timedelta_to_isoformat


class VersionedLagsAdder(BaseConfig, VersionedTimeSeriesTransform):
    """Create lag features while preserving data availability constraints.

    This transform creates lagged versions of a column for capturing temporal dependencies
    in energy forecasting. Unlike traditional lag transforms, this preserves data availability
    constraints, ensuring lag features only use data that would have been available at prediction time.

    Energy consumption has strong temporal patterns: yesterday's peak predicts today's,
    previous hours influence next hour's demand, and energy use lags behind weather changes.
    In production forecasting, you cannot use future data to predict the present.

    For each lag, the transform:
    - Shifts timestamps forward (e.g., -2h lag moves 10:00 data to 12:00)
    - Preserves availability constraints (data available at 15:00 stays available at 15:00)
    - Creates new feature columns (e.g., 'load' becomes 'load_lag_-PT2H')
    - Maintains the versioned structure so multiple data versions are preserved independently

    In versioned datasets with different availability times, this allows automatic selection
    of appropriate data versions:
    - **Short lags + long lead times**: Use high-quality data (available later)
    - **Long lags + short lead times**: Use lower-quality data (available sooner)

    Example:
        Create lag features for energy forecasting:

        >>> from datetime import timedelta
        >>> import pandas as pd
        >>> from openstef_core.datasets.versioned_timeseries.dataset import VersionedTimeSeriesDataset

        >>> # Create sample energy data
        >>> data = pd.DataFrame({
        ...     'timestamp': pd.date_range('2025-01-01 10:00', periods=4, freq='h'),
        ...     'available_at': pd.date_range('2025-01-01 10:00', periods=4, freq='h'),
        ...     'load': [100.0, 110.0, 120.0, 130.0]
        ... })
        >>> dataset = VersionedTimeSeriesDataset.from_dataframe(data, timedelta(hours=1))

        >>> # Create 1-hour and 2-hour lag features
        >>> transform = VersionedLagsAdder(
        ...     feature='load',
        ...     lags=[timedelta(hours=-1), timedelta(hours=-2)]
        ... )
        >>> result = transform.transform(dataset)
        >>> snapshot = result.select_version()

        >>> # Check lag feature names
        >>> lag_features = [col for col in snapshot.feature_names if 'lag' in col]
        >>> sorted(lag_features)
        ['load_lag_-PT1H', 'load_lag_-PT2H']

        >>> # Verify lag values (1-hour lag shifts 100->11:00, 110->12:00, etc.)
        >>> snapshot.data['load_lag_-PT1H'].dropna().tolist()
        [100.0, 110.0, 120.0]

    Note:
        Lag features are constrained to the original dataset's time range. A dataset covering
        10:00-13:00 with a -2h lag will have features available only from 12:00-13:00, not extending
        to 15:00. This prevents creating timepoints outside the forecasting range.
    """

    feature: str = Field(
        default=...,
        description="The name of the column to apply the lags on.",
    )
    lags: list[timedelta] = Field(
        default=[],
        description="List of lags to apply to the time series data. Negative values indicate look back. "
        "For example timedelta(hours=-2) uses data from 2 hours ago.",
        min_length=1,
    )

    @property
    @override
    def is_fitted(self) -> bool:
        return True  # Stateless transform, always considered fitted

    @override
    def fit(self, data: VersionedTimeSeriesDataset) -> None:
        pass

    @override
    def transform(self, data: VersionedTimeSeriesDataset) -> VersionedTimeSeriesDataset:
        if self.feature not in data.feature_names:
            raise MissingColumnsError(missing_columns=[self.feature])

        source_part = next(part for part in data.data_parts if self.feature in part.feature_names)
        lag_parts: list[TimeSeriesDataset] = [
            _transform_to_lag(data=source_part, feature=self.feature, lag=lag) for lag in self.lags
        ]

        return VersionedTimeSeriesDataset(
            data_parts=[
                *data.data_parts,
                *lag_parts,
            ]
        )

    @override
    def to_state(self) -> State:
        return self.model_dump(mode="json")

    @override
    def from_state(self, state: State) -> Self:
        return self.model_validate(state)


def _transform_to_lag(data: TimeSeriesDataset, feature: str, lag: timedelta) -> TimeSeriesDataset:
    # Shift timestamps forward by the lag duration
    data_df = data.data.rename(columns={feature: f"{feature}_lag_{timedelta_to_isoformat(lag)}"})
    data_df.index = cast(pd.Series, data_df.index) - lag  # pyright: ignore[reportArgumentType]

    # Lagging adds a lot of new timepoints outside the original range.
    # We filter to only keep timepoints within the original timestamp range.
    mask: pd.Series = (data_df.index >= data.data.index.min()) & (data_df.index <= data.data.index.max())
    data_df = data_df[mask]

    return TimeSeriesDataset(
        data=data_df,
        sample_interval=data.sample_interval,
        available_at_column=data.available_at_column,
    )
