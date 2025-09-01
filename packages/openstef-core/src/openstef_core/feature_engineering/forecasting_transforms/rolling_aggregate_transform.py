# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for extracting trend features from time series data.

This module provides functionality to compute trend-based features that capture
long-term patterns and movements in time series data, helping improve forecasting
accuracy by identifying underlying trends.
"""

from datetime import timedelta
from typing import Literal, cast, override

import pandas as pd
from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.transforms import TimeSeriesTransform
from openstef_core.datasets.validation import validate_required_columns
from openstef_core.utils import timedelta_to_isoformat

type AggregationFunction = Literal["mean", "median", "max", "min"]


class RollingAggregateTransform(BaseConfig, TimeSeriesTransform):
    """Transform that adds rolling aggregate features to time series data.

    This transform computes rolling aggregate statistics (e.g., mean, median, min, max)
    over a specified rolling window and adds these as new features to the dataset.
    It is useful for capturing recent trends and patterns in the data.

    The rolling aggregates are computed on the specified columns of the dataset.

    Example:
        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> from openstef_core.feature_engineering.forecasting_transforms import RollingAggregateTransform
        >>>
        >>> # Create sample dataset
        >>> data = pd.DataFrame({
        ...     'load': [100, 120, 110, 130, 125],
        ...     'temperature': [20, 22, 21, 23, 24]
        ... }, index=pd.date_range('2025-01-01', periods=5, freq='1h'))
        >>> dataset = TimeSeriesDataset(data, timedelta(hours=1))
        >>>
        >>> # Initialize and apply transform
        >>> transform = RollingAggregateFeaturesTransform(
        ...     columns=['load', 'temperature'],
        ...     rolling_window_size=timedelta(hours=2),
        ...     aggregation_functions=["mean", "max"]
        ... )
        >>> transformed_dataset = transform.transform(dataset)
        >>> 'rolling_mean_load_PT2H' in transformed_dataset.data.columns
        True
        >>> 'rolling_max_temperature_PT2H' in transformed_dataset.data.columns
        True
    """

    columns: list[str] = Field(
        description="Columns to compute rolling aggregates for.",
    )
    rolling_window_size: timedelta = Field(
        default=timedelta(hours=24),
        description="Rolling window size for the aggregation.",
    )
    aggregation_functions: list[AggregationFunction] = Field(
        default_factory=lambda: ["median", "min", "max"],
        description="List of aggregation functions to compute over the rolling window. ",
    )

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        validate_required_columns(dataset=data, required_columns=self.columns)

        agg_series: list[pd.DataFrame] = [data.data]
        for column in self.columns:
            # Compute rolling aggregations (pandas handles NaNs automatically)
            rolling_window_column = data.data[column].rolling(window=self.rolling_window_size)

            # Compute aggregations
            aggregated_data: pd.DataFrame = cast(
                pd.DataFrame,
                rolling_window_column.agg(self.aggregation_functions),  # type: ignore[misc]
            ).rename(
                columns={
                    func: f"rolling_{func}_{column}_{timedelta_to_isoformat(td=self.rolling_window_size)}"
                    for func in self.aggregation_functions
                }
            )
            agg_series.append(aggregated_data)

        return TimeSeriesDataset(
            data=pd.concat(agg_series, axis=1),
            sample_interval=data.sample_interval,
        )


__all__ = ["RollingAggregateTransform"]
