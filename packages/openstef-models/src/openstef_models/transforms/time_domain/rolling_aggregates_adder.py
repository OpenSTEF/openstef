# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for adding rolling aggregate features to time series data."""

import logging
from datetime import timedelta
from typing import Any, Literal, override

import pandas as pd
from pydantic import Field, PrivateAttr

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.validation import validate_required_columns
from openstef_core.exceptions import NotFittedError
from openstef_core.transforms import TimeSeriesTransform
from openstef_core.types import LeadTime
from openstef_core.utils import timedelta_to_isoformat

type AggregationFunction = Literal["mean", "median", "max", "min"]


class RollingAggregatesAdder(BaseConfig, TimeSeriesTransform):
    """Transform that adds rolling aggregate features to time series data.

    Computes rolling aggregate statistics (e.g., mean, median, min, max)
    over a specified rolling window and adds these as new features to the dataset.
    It is useful for capturing recent trends and patterns in the data. Handles
    missing target data during inference via a fallback strategy:

    1. Forward-fill from last computed aggregate
    2. Use last valid aggregate from training

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
        ...     aggregation_functions=["mean", "max"],
        ...     horizons=[LeadTime.from_string("PT36H")],
        ... )
        >>> transform.fit(dataset)
        >>> transformed_dataset = transform.transform(dataset)
        >>> result = transformed_dataset.data[['rolling_mean_load_PT2H', 'rolling_max_load_PT2H']]
        >>> print(result.round(1).head(3))
                             rolling_mean_load_PT2H  rolling_max_load_PT2H
        timestamp
        2025-01-01 00:00:00                   100.0                  100.0
        2025-01-01 01:00:00                   110.0                  120.0
        2025-01-01 02:00:00                   115.0                  120.0
    """

    feature: str = Field(description="Feature to compute rolling aggregates for.")
    horizons: list[LeadTime] = Field(description="List of forecast horizons.", min_length=1)
    rolling_window_size: timedelta = Field(
        default=timedelta(hours=24),
        description="Rolling window size for the aggregation.",
    )
    aggregation_functions: list[AggregationFunction] = Field(
        default_factory=lambda: ["median", "min", "max"],
        description="Aggregation functions to compute over the rolling window.",
    )

    _logger: logging.Logger = PrivateAttr(default_factory=lambda: logging.getLogger(__name__))
    _last_valid_aggregates: dict[str, float] = PrivateAttr(default_factory=dict[str, float])
    _is_fitted: bool = PrivateAttr(default=False)

    def _make_column_name(self, func: AggregationFunction) -> str:
        suffix = timedelta_to_isoformat(td=self.rolling_window_size)
        return f"rolling_{func}_{self.feature}_{suffix}"

    def _compute_rolling_aggregates(self, series: pd.Series) -> pd.DataFrame:
        return series.dropna().rolling(window=self.rolling_window_size).agg(self.aggregation_functions)  # type: ignore[return-value]

    @override
    def fit(self, data: TimeSeriesDataset) -> None:
        """Compute and store last valid aggregates from training data for fallback."""
        validate_required_columns(df=data.data, required_columns=[self.feature])

        rolling_df = self._compute_rolling_aggregates(data.data[self.feature])

        for func in self.aggregation_functions:
            valid_rows = rolling_df[func].dropna()
            if not valid_rows.empty:
                self._last_valid_aggregates[self._make_column_name(func)] = float(valid_rows.iloc[-1])

        self._is_fitted = True

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        """Add rolling aggregate features, using fallbacks for missing values.

        Returns:
            Dataset with rolling aggregate feature columns added.

        Raises:
            NotFittedError: If fit() has not been called.
        """
        if not self.aggregation_functions:
            self._logger.warning("No aggregation functions specified. Returning original data.")
            return data

        if len(self.horizons) > 1:
            self._logger.warning("Multiple horizons not yet supported. Returning original data.")
            return data

        validate_required_columns(df=data.data, required_columns=[self.feature])

        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)

        # Compute rolling aggregates and apply fallback for missing values
        result_df = self._compute_and_apply_fallback(data.data)
        return data.copy_with(result_df)

    def _compute_and_apply_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        rolling_df = self._compute_rolling_aggregates(df[self.feature])
        rolling_df = rolling_df.reindex(df.index).ffill()

        # Rename columns and apply last valid fallback
        column_mapping = {func: self._make_column_name(func) for func in self.aggregation_functions}
        rolling_df = rolling_df.rename(columns=column_mapping)

        for col in column_mapping.values():
            if col in self._last_valid_aggregates:
                rolling_df[col] = rolling_df[col].fillna(self._last_valid_aggregates[col])  # pyright: ignore[reportUnknownMemberType]

            if rolling_df[col].isna().any():
                self._logger.warning("Column '%s' has NaN values after fallback.", col)

        return pd.concat([df, rolling_df], axis=1)

    @override
    def features_added(self) -> list[str]:
        return [self._make_column_name(func) for func in self.aggregation_functions]

    @override
    def __setstate__(self, state: Any) -> None:  # TODO(#799): delete after stable release
        if "_last_valid_aggregates" not in state["__pydantic_private__"]:
            state["__pydantic_private__"]["_last_valid_aggregates"] = {}
        if "_is_fitted" not in state["__pydantic_private__"]:
            state["__pydantic_private__"]["_is_fitted"] = True
        return super().__setstate__(state)


__all__ = ["RollingAggregatesAdder"]
