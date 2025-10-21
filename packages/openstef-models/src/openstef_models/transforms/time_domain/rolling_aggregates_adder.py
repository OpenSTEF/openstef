# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for extracting trend features from time series data.

This module provides functionality to compute trend-based features that capture
long-term patterns and movements in time series data, helping improve forecasting
accuracy by identifying underlying trends.
"""

from datetime import timedelta
from typing import Literal, Self, cast, override

import pandas as pd
from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.validation import validate_required_columns
from openstef_core.mixins import State
from openstef_core.transforms import TimeSeriesTransform
from openstef_core.utils import timedelta_to_isoformat

type AggregationFunction = Literal["mean", "median", "max", "min"]


class RollingAggregatesAdder(BaseConfig, TimeSeriesTransform):
    """Transform that adds rolling aggregate features to time series data.

    This transform computes rolling aggregate statistics (e.g., mean, median, min, max)
    over a specified rolling window and adds these as new features to the dataset.
    It is useful for capturing recent trends and patterns in the data.

    The rolling aggregates are computed on the specified columns of the dataset.

    Example:
        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> from openstef_models.transforms.time_domain import RollingAggregatesAdder
        >>>
        >>> # Create sample dataset
        >>> data = pd.DataFrame({
        ...     'load': [100, 120, 110, 130, 125],
        ...     'temperature': [20, 22, 21, 23, 24]
        ... }, index=pd.date_range('2025-01-01', periods=5, freq='1h'))
        >>> dataset = TimeSeriesDataset(data, timedelta(hours=1))
        >>>
        >>> # Initialize and apply transform
        >>> transform = RollingAggregatesAdder(
        ...     feature='load',
        ...     rolling_window_size=timedelta(hours=2),
        ...     aggregation_functions=["mean", "max"]
        ... )
        >>> transformed_dataset = transform.transform(dataset)
        >>> result = transformed_dataset.data[['rolling_mean_load_PT2H', 'rolling_max_load_PT2H']]
        >>> print(result.round(1).head(3))
                             rolling_mean_load_PT2H  rolling_max_load_PT2H
        2025-01-01 00:00:00                   100.0                  100.0
        2025-01-01 01:00:00                   110.0                  120.0
        2025-01-01 02:00:00                   115.0                  120.0
    """

    feature: str = Field(
        description="Feature to compute rolling aggregates for.",
    )
    rolling_window_size: timedelta = Field(
        default=timedelta(hours=24),
        description="Rolling window size for the aggregation.",
    )
    aggregation_functions: list[AggregationFunction] = Field(
        default_factory=lambda: ["median", "min", "max"],
        description="List of aggregation functions to compute over the rolling window. ",
    )

    def _transform_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        rolling_df = cast(
            pd.DataFrame,
            df[self.feature].rolling(window=self.rolling_window_size).agg(self.aggregation_functions),  # type: ignore
        )
        suffix = timedelta_to_isoformat(td=self.rolling_window_size)
        rolling_df = rolling_df.rename(
            columns={func: f"rolling_{func}_{self.feature}_{suffix}" for func in self.aggregation_functions}
        )

        return pd.concat([df, rolling_df], axis=1)

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        validate_required_columns(df=data.data, required_columns=[self.feature])
        return data.pipe_pandas(self._transform_pandas)

    @override
    def to_state(self) -> State:
        return self.model_dump(mode="json")

    @override
    def from_state(self, state: State) -> Self:
        return self.model_validate(state)

    @override
    def features_added(self) -> list[str]:
        return [
            f"rolling_{func}_{self.feature}_{timedelta_to_isoformat(self.rolling_window_size)}"
            for func in self.aggregation_functions
        ]


__all__ = ["RollingAggregatesAdder"]
