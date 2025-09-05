# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Lag feature transforms for versioned time series data.

This module provides transforms for creating lag features while preserving
data availability constraints in versioned datasets. Essential for energy
forecasting where temporal dependencies matter but data availability varies.
"""

from datetime import timedelta
from typing import override

from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import VersionedTimeSeriesPart
from openstef_core.datasets.transforms import VersionedTimeSeriesTransform
from openstef_core.datasets.versioned_timeseries.dataset import VersionedTimeSeriesDataset
from openstef_core.exceptions import MissingColumnsError
from openstef_core.utils import timedelta_to_isoformat


class VersionedLagTransform(BaseConfig, VersionedTimeSeriesTransform):
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

    Args:
        column: Name of the column to create lag features from.
        lags: List of lag periods. Negative values look backward in time (typical use).
              Example: timedelta(hours=-2) uses data from 2 hours ago.

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
        >>> transform = VersionedLagTransform(
        ...     column='load',
        ...     lags=[timedelta(hours=-1), timedelta(hours=-2)]
        ... )
        >>> result = transform.transform(dataset)
        >>> snapshot = result.select_version()

        >>> # Check lag feature names
        >>> lag_features = [col for col in snapshot.data.columns if 'lag' in col]
        >>> sorted(lag_features)
        ['load_lag_-PT1H', 'load_lag_-PT2H']

        >>> # Verify lag values (1-hour lag shifts 100->11:00, 110->12:00, etc.)
        >>> snapshot.data['load_lag_-PT1H'].dropna().tolist()
        [100.0, 110.0, 120.0, 130.0]

    Note:
        Lag features extend the dataset's time range. A dataset covering 10:00-13:00
        with a -2h lag will have features available from 12:00-15:00.
    """

    column: str = Field(
        default=...,
        description="The name of the column to apply the lags on.",
    )
    lags: list[timedelta] = Field(
        default=[],
        description="List of lags to apply to the time series data. Negative values indicate look back.",
        min_length=1,
    )

    @override
    def transform(self, data: VersionedTimeSeriesDataset) -> VersionedTimeSeriesDataset:
        if self.column not in data.feature_names:
            raise MissingColumnsError(missing_columns=[self.column])

        source_part = next(part for part in data.data_parts if self.column in part.feature_names)
        lag_parts: list[VersionedTimeSeriesPart] = [
            _transform_to_lag(data=source_part, column=self.column, lag=lag) for lag in self.lags
        ]

        return VersionedTimeSeriesDataset(
            data_parts=[
                *data.data_parts,
                *lag_parts,
            ]
        )


def _transform_to_lag(data: VersionedTimeSeriesPart, column: str, lag: timedelta) -> VersionedTimeSeriesPart:
    data_df = data.data.rename(columns={column: f"{column}_lag_{timedelta_to_isoformat(lag)}"})
    data_df[data.timestamp_column] = data_df[data.timestamp_column].sub(lag)  # pyright: ignore[reportArgumentType]
    return VersionedTimeSeriesPart(
        data=data_df,
        timestamp_column=data.timestamp_column,
        available_at_column=data.available_at_column,
        sample_interval=data.sample_interval,
    )
