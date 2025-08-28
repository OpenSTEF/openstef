# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for extracting trend features from time series data.

This module provides functionality to compute trend-based features that capture
long-term patterns and movements in time series data, helping improve forecasting
accuracy by identifying underlying trends.
"""

from datetime import timedelta
from enum import StrEnum

import pandas as pd
from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.transforms import TimeSeriesTransform
from openstef_core.utils import timedelta_to_isoformat


class AggregationFunction(StrEnum):
    """Enum of supported aggregation functions for rolling aggregates."""

    MEAN = "mean"
    MEDIAN = "median"
    MAX = "max"
    MIN = "min"


class RollingAggregateFeaturesConfig(BaseConfig):
    """Configuration for the RollingAggregateFeatures transform."""

    rolling_window_size: timedelta = Field(
        default=timedelta(hours=24),
        description="Rolling window size for the aggregation.",
    )
    aggregation_functions: list[AggregationFunction] = Field(
        default_factory=lambda: [AggregationFunction.MEDIAN, AggregationFunction.MIN, AggregationFunction.MAX],
        description="List of aggregation functions to compute over the rolling window. ",
    )


class RollingAggregateFeatures(TimeSeriesTransform):
    """Transform that adds rolling aggregate features to time series data.

    This transform computes rolling aggregate statistics (e.g., mean, median, min, max)
    over a specified rolling window and adds these as new features to the dataset.
    It is useful for capturing recent trends and patterns in the data.

    The rolling aggregates are computed on the 'load' column of the dataset.

    Example:
        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> from openstef_core.feature_engineering.forecasting_transforms.rolling_aggregate_features import (
        ...     RollingAggregateFeatures,
        ...     RollingAggregateFeaturesConfig,
        ... )
        >>>
        >>> # Create sample dataset
        >>> data = pd.DataFrame({
        ...     'load': [100, 120, 110, 130, 125]
        ... }, index=pd.date_range('2025-01-01', periods=5, freq='1h'))
        >>> dataset = TimeSeriesDataset(data, timedelta(hours=1))
        >>>
        >>> # Initialize and apply transform
        >>> config = RollingAggregateFeaturesConfig(
        ...     rolling_window_size=timedelta(hours=2),
        ...     aggregation_functions=[AggregationFunction.MEAN, AggregationFunction.MAX]
        ... )
        >>> transform = RollingAggregateFeatures(config=config)
        >>> transformed_dataset = transform.fit_transform(dataset)
        >>> 'rolling_mean_load_PT2H' in transformed_dataset.data.columns
        True
        >>> 'rolling_max_load_PT2H' in transformed_dataset.data.columns
        True
    """

    def __init__(self, config: RollingAggregateFeaturesConfig):
        """Initialize the RollingAggregateFeatures transform with a configuration."""
        self.config = config
        self.rolling_aggregate_features: pd.DataFrame = pd.DataFrame()

    def _get_rolling_aggregate_feature_name(self, aggregation_function: AggregationFunction) -> str:
        """Generate the feature name for the rolling aggregate based on config.

        Args:
            aggregation_function: The aggregation function used (e.g., mean, median).

        Returns:
            The generated feature name as a string.
        """
        return f"rolling_{aggregation_function.value}_load_{timedelta_to_isoformat(td=self.config.rolling_window_size)}"

    def fit(self, data: TimeSeriesDataset) -> None:
        """Fit the transform to the input time series `load` column.

        This method computes the rolling aggregate features based on the specified
        rolling window and aggregation functions.

        Args:
            data: Time series dataset with DatetimeIndex.

        Raises:
            ValueError: If the 'load' column is not present in the dataset.
        """
        if "load" not in data.data.columns:
            raise ValueError("The DataFrame must contain a 'load' column.")

        rolling_window_load = data.data["load"].dropna().rolling(window=self.config.rolling_window_size)

        rolling_aggregate_features: dict[str, pd.Series] = {
            self._get_rolling_aggregate_feature_name(func): rolling_window_load.aggregate(func.value)  # type: ignore[reportUnknownMemberType]
            for func in self.config.aggregation_functions
        }

        self.rolling_aggregate_features = pd.DataFrame(
            data=rolling_aggregate_features,
        )

    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        """Transform the input time series data by adding rolling aggregate features.

        This method adds the precomputed rolling aggregate features to the input dataset.
        Missing values are forward-filled to account for lags in the data and preventing
        NaNs in the output.

        Args:
            data: The input time series data to be transformed.

        Returns:
            A new instance of TimeSeriesDataset containing the original and new rolling aggregate features.
        """
        aligned_features = self.rolling_aggregate_features.reindex(data.data.index).ffill()

        return TimeSeriesDataset(
            data=pd.concat(
                [data.data, aligned_features],
                axis=1,
            ),
            sample_interval=data.sample_interval,
        )
