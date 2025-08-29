# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Flatliner check transform for time series datasets.

This module provides functionality for detecting flatliner patterns in time series load data.
A flatliner is defined as a period where the load remains constant for a specified duration, which can indicate sensor
malfunction, data transmission errors, or other anomalies in energy forecasting datasets.
"""

from typing import Any

import pandas as pd
from pydantic import Field, PrivateAttr

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.transforms import TimeSeriesTransform


class CompletenessCheckTransform(TimeSeriesTransform, BaseConfig):
    """Transformer to detect flatliner patterns in time series load data.

    A flatliner is a period where the load remains constant for a specified duration.
    This class can detect both zero and non-zero flatliners, depending on configuration.

    Example:
    >>> from datetime import datetime, timedelta
    >>> import numpy as np
    >>> import pandas as pd
    >>> from openstef_core.datasets import TimeSeriesDataset
    >>> from openstef_core.feature_engineering.validation_transforms.completeness_check import (
    ...     CompletenessCheckTransform,
    ... )
    >>> data = pd.DataFrame({
    ...     'radiation': [100, 110, 110, np.nan],
    ...     'temperature': [20, np.nan, 24, 21],
    ...     'wind_speed': [5, 6, np.nan, 3],
    ... },
    ... index=pd.date_range("2025-01-01", periods=4, freq="15min"))
    >>> dataset = TimeSeriesDataset(data, timedelta(minutes=15))
    >>> transform = CompletenessCheckTransform()
    >>> transform.fit(dataset)
    >>> transform.completeness
    0.75
    """

    columns: list[str] | None = Field(
        default=None,
        description="List of columns to check for completeness. If None, all columns are checked.",
    )
    weights: dict[str, float] | None = Field(
        default=None,
        description="Weights for each column to adjust their importance in the completeness calculation.",
    )
    completeness_threshold: float = Field(
        default=0.5,
        description="Threshold for completeness below which the data is considered insufficiently complete.",
    )
    _completeness: float = PrivateAttr(default=False)
    _sufficiently_complete: bool = PrivateAttr(default=False)

    @property
    def completeness(self) -> float:
        """Indicates how complete the data is."""
        return self._completeness

    @property
    def sufficiently_complete(self) -> bool:
        """Indicates whether the data is sufficiently complete."""
        return self._sufficiently_complete

    def __init__(self, **data: Any):
        """Initializes the CompletenessCheckTransform with the given configuration."""
        super().__init__(**data)

    def _calculate_sufficiently_complete(self, data: pd.DataFrame) -> bool:
        """Check if the DataFrame is sufficiently complete.

        Args:
            data: The input DataFrame to check.

        Returns:
            True if the DataFrame is sufficiently complete, False otherwise.
        """
        completeness = self._calculate_completeness(data)
        return completeness >= self.completeness_threshold

    def _calculate_completeness(self, data: pd.DataFrame) -> float:
        """Calculate how complete the input DataFrame is.

        Args:
            data: The input DataFrame to check.

        Returns:
            A float indicating the completeness of the DataFrame.
        """
        if self.columns:
            data = data[self.columns]

        if not self.weights:
            self.weights = dict.fromkeys(data.columns, 1.0)

        weighted_completeness = 0.0
        total_weight = 0.0

        for col in data.columns:
            weight = self.weights.get(col, 1.0)
            col_completeness = 1.0 - (data[col].isna().sum() / len(data)) if len(data) > 0 else 0.0
            weighted_completeness += weight * col_completeness
            total_weight += weight

        return float(weighted_completeness / total_weight if total_weight > 0 else 0.0)

    def fit(self, data: TimeSeriesDataset) -> None:
        """Calculates and stores the completeness metrics for the provided time series dataset.

        Args:
            data: The dataset containing time series data to evaluate.
        """
        self._completeness = self._calculate_completeness(
            data=data.data,
        )
        self._sufficiently_complete = self._calculate_sufficiently_complete(
            data=data.data,
        )

    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:  # noqa: PLR6301
        """This method returns the input data unchanged.

        Args:
            data: The input time series dataset to be transformed.

        Returns:
            The unmodified input TimeSeriesDataset.
        """
        return data
