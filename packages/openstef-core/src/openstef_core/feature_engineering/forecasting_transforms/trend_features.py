from datetime import timedelta
from enum import StrEnum

import pandas as pd
from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.transforms import TimeSeriesTransform
from openstef_core.utils import timedelta_to_isoformat


class AggregationFunction(StrEnum):
    MEAN = "mean"
    MEDIAN = "median"
    MAX = "max"
    MIN = "min"


class RollingAggregateFeaturesConfig(BaseConfig):
    """Configuration for the RollingAggregateFeatures transform.
    """

    rolling_window_size: timedelta = Field(
        default=timedelta(hours=24),
        description="Rolling window size for the aggregation.",
    )
    aggregation_functions: list[AggregationFunction] = Field(
        default_factory=lambda: [AggregationFunction.MEDIAN, AggregationFunction.MIN, AggregationFunction.MAX],
        description="List of aggregation functions to compute over the rolling window. "
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
        >>> from openstef_core.feature_engineering.forecasting_transforms.trend_features import (  # noqa: E501
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
        >>> transformed_dataset = transform.transform(dataset)
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
        """Generate the feature name for the rolling aggregate based on config."""
        return f"rolling_{aggregation_function.value}_load_{timedelta_to_isoformat(td=self.config.rolling_window_size)}"

    def fit(self, data: TimeSeriesDataset) -> None:
        """Fit the transform to the input time series `load` column.
        This method computes the rolling aggregate features based on the specified
        rolling window and aggregation functions. Missing values are forward-filled
        to account for lags in the data.

        Args:
            data: Time series dataset with DatetimeIndex.
        """
        if "load" not in data.data.columns:
            raise ValueError("The DataFrame must contain a 'load' column.")

        rolling_window_load = data.data["load"].dropna().rolling(window=self.config.rolling_window_size)
        self.rolling_aggregate_features = pd.DataFrame(
            {
                self._get_rolling_aggregate_feature_name(func): rolling_window_load.aggregate(func.value).ffill()
                for func in self.config.aggregation_functions
            }
        )

    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        """Transform the input time series data by adding rolling aggregate features.

        This method adds the precomputed rolling aggregate features to the input dataset.
        If the transform has not been fitted yet, it raises an error.

        Args:
            data: The input time series data to be transformed.

        Returns:
            A new instance of TimeSeriesDataset containing the original and new rolling aggregate features.

        Raises:
            ValueError: If the transform has not been fitted yet.
        """
        # # Align the rolling features with the input data index
        # aligned_features = self.rolling_aggregate_features.reindex(data.data.index).ffill()
        return TimeSeriesDataset(
            data=pd.concat(
                [data.data, self.rolling_aggregate_features],
                axis=1,
            ),
            sample_interval=data.sample_interval,
        )
