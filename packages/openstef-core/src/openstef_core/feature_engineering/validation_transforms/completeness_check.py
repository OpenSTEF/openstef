# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Completeness check transform for time series datasets.

This module provides functionality for checking the completeness of time series load data.
Completeness is defined as the ratio of non-missing values to the total number of values in a given time series.
"""

from typing import override
import pandas as pd
from pydantic import Field, PrivateAttr

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.transforms import TimeSeriesTransform
from openstef_core.exceptions import InsufficientlyCompleteError


class CompletenessCheckTransform(TimeSeriesTransform, BaseConfig):
    """Transformer to check the completeness of time series load data.

    Completeness is defined as the ratio of non-missing values to the total number of values in a given time series.
    This class can be configured to check specific columns and apply weights to their importance in the
    completeness calculation.

    Example:
    >>> from datetime import datetime, timedelta
    >>> import numpy as np
    >>> import pandas as pd
    >>> from openstef_core.datasets import TimeSeriesDataset
    >>> from openstef_core.feature_engineering.validation_transforms.completeness_check import (
    ...     CompletenessCheckTransform,
    ... )
    >>> data = pd.DataFrame({
    ...     'radiation': [100, np.nan, np.nan, np.nan],
    ...     'temperature': [20, np.nan, 24, np.nan],
    ...     'wind_speed': [np.nan, np.nan, np.nan, np.nan],
    ... },
    ... index=pd.date_range("2025-01-01", periods=4, freq="15min"))
    >>> dataset = TimeSeriesDataset(data, timedelta(minutes=15))
    >>> transform = CompletenessCheckTransform()
    >>> try:
    ...     transform.fit(dataset)
    ... except NotSufficientlyCompleteError:
    ...     pass
    >>> transform.completeness
    0.25
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
    error_on_insufficient_completeness: bool = Field(
        default=True,
        description="Whether to raise an error when the data is not sufficiently complete.",
    )
    check_on_transform: bool = Field(
        default=False,
        description="Whether to check completeness during the transform phase.",
    )
    _completeness: float = PrivateAttr(default=False)
    _is_sufficiently_complete: bool = PrivateAttr(default=False)

    @property
    def completeness(self) -> float:
        """Indicates how complete the data is."""
        return self._completeness

    @property
    def is_sufficiently_complete(self) -> bool:
        """Indicates whether the data is sufficiently complete."""
        return self._is_sufficiently_complete

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

    @override
    def fit(self, data: TimeSeriesDataset) -> None:
        """Calculates and stores the completeness metrics for the provided time series dataset.

        Args:
            data: The dataset containing time series data to evaluate.

        Raises:
            InsufficientlyCompleteError: If the dataset is not sufficiently complete.
        """
        self._completeness = self._calculate_completeness(
            data=data.data,
        )
        self._is_sufficiently_complete = self._calculate_sufficiently_complete(
            data=data.data,
        )
        if not self._is_sufficiently_complete and self.error_on_insufficient_completeness:
            raise InsufficientlyCompleteError("The dataset is not sufficiently complete.")

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        """Returns the input data unchanged, optionally checking for completeness.

        This method can optionally run completeness checks when `check_on_transform=True`.

        Args:
            data: The input time series dataset to be transformed.

        Returns:
            The unmodified input TimeSeriesDataset.

        Raises:
            InsufficientlyCompleteError: If the dataset is not sufficiently complete and `error_on_insufficient_completeness` is True.
        """
        if not self.check_on_transform:
            return data

        self._completeness = self._calculate_completeness(
            data=data.data,
        )
        self._is_sufficiently_complete = self._calculate_sufficiently_complete(
            data=data.data,
        )
        if not self._is_sufficiently_complete:
            raise InsufficientlyCompleteError("The dataset is not sufficiently complete.")

        return data
