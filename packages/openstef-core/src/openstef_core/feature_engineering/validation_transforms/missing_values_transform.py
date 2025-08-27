# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Missing values imputation and data quality transform for time series datasets.

This module provides functionality for handling missing values in time series data
through various imputation strategies and data cleaning operations.
"""

import logging
from enum import Enum
from typing import Any, cast

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field, model_validator
from sklearn.impute import SimpleImputer

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.transforms import TimeSeriesTransform

_logger = logging.getLogger(__name__)


class ImputationStrategy(Enum):
    """Enumeration of available imputation strategies for missing values."""

    MEAN = "mean"
    MEDIAN = "median"
    MOST_FREQUENT = "most_frequent"
    CONSTANT = "constant"


class MissingValuesTransform(TimeSeriesTransform, BaseConfig):
    """Transform that checks for, imputes and drops missing values in time series data.

    This transform applies imputation strategies to handle missing values in the dataset.
    For features specified in no_fill_future_values_features, it removes rows with
    trailing NaN values instead of imputing them, as future values for these features
    cannot be reasonably estimated.

    Example:
    >>> from datetime import timedelta
    >>> import numpy as np
    >>> import pandas as pd
    >>> from openstef_core.datasets import TimeSeriesDataset
    >>> from openstef_core.feature_engineering.validation_transforms.missing_values_transform import (
    ...     ImputationStrategy,
    ...     MissingValuesTransform,
    ... )
    >>> data = pd.DataFrame(
    ...     {
    ...         "radiation": [100, np.nan, 110, np.nan],
    ...         "temperature": [20, np.nan, 24, 21],
    ...         "wind_speed": [5, 6, np.nan, np.nan],
    ...         "empty_feature": [np.nan, np.nan, np.nan, np.nan]
    ...     },
    ...     index=pd.date_range("2025-01-01", periods=4, freq="1h"),
    ... )
    >>> dataset = TimeSeriesDataset(data, timedelta(hours=1))
    >>> transform = MissingValuesTransform(
    ...     imputation_strategy=ImputationStrategy.MEAN,
    ...     missing_value=np.nan,
    ...     no_fill_future_values_features=["radiation"]
    ... )
    >>> transform.fit(dataset)
    >>> result = transform.transform(dataset)
    >>> int(result.data.isna().sum().sum()) == 0
    True
    >>> result.data.shape[0] == 3 # Last row dropped
    True
    >>> result.data.shape[1] == 3 # Empty feature column dropped
    True
    """

    imputation_strategy: ImputationStrategy = Field(
        ...,
        description="The strategy to use for imputation",
    )
    missing_value: float = Field(
        default_factory=lambda: np.nan,
        description="The placeholder for missing values that should be imputed",
    )
    fill_value: float | str | None = Field(
        default=None,
        description="Value to use when imputation_strategy is CONSTANT",
    )
    no_fill_future_values_features: list[str] = Field(
        default_factory=list,
        description="List of feature names for which trailing NaN values should not be filled",
    )

    imputer: SimpleImputer = Field(exclude=True, default_factory=SimpleImputer)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_fill_value_with_strategy(self) -> "MissingValuesTransform":
        """Validate that fill_value is provided when strategy is CONSTANT.

        Returns:
            The validated model instance.

        Raises:
            ValueError: If imputation_strategy is CONSTANT but fill_value is None.
        """
        if self.imputation_strategy == ImputationStrategy.CONSTANT and self.fill_value is None:
            raise ValueError("fill_value must be provided when imputation_strategy is CONSTANT")
        return self

    def __init__(self, **data: Any):
        """Initialize the MissingValuesTransform with the given configuration."""
        super().__init__(**data)

        self.imputer = SimpleImputer(
            strategy=self.imputation_strategy.value,
            fill_value=self.fill_value,
            missing_values=self.missing_value,
            keep_empty_features=False,
        )
        self.imputer.set_output(transform="pandas")

    def _drop_empty_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Drop features that contain only missing values.

        Args:
            data: The input DataFrame.

        Returns:
            A DataFrame with empty features removed.
        """
        all_missing_mask = data.isna().all() if np.isnan(self.missing_value) else (data == self.missing_value).all()

        for col in data.columns[all_missing_mask]:
            _logger.warning("Dropped column '%s' from dataset because it contains only missing values.", col)

        non_empty_columns = data.columns[~all_missing_mask]
        return data[non_empty_columns]

    def _remove_trailing_null_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove trailing rows that contain null values in specified features.

        This method removes rows from the end of the dataset where specified features
        (defined in no_fill_future_values_features) contain null values. This
        is useful for cleaning datasets where future values should not be filled and
        trailing nulls indicate incomplete data.

        Args:
            data: The input pandas DataFrame containing the data
                to be processed.

        Returns:
            A pandas DataFrame with trailing null rows removed for the specified
                features. If no features are configured for checking or none of the
                configured features exist in the data, returns the original data unchanged.
        """
        if not self.no_fill_future_values_features:
            return data

        features_to_check = [f for f in self.no_fill_future_values_features if f in data.columns]
        for feature in self.no_fill_future_values_features:
            if feature not in data.columns:
                _logger.warning("Feature '%s' not found in dataset columns.", feature)

        if not features_to_check:
            return data

        subset_df = data[features_to_check]
        mask = ~subset_df.bfill().isna().any(axis="columns")
        return data.loc[mask]

    def fit(self, data: TimeSeriesDataset) -> None:
        """Fit the missing values transformer to the provided time series data.

        This method prepares the imputer by fitting it to the training data after
        removing any trailing null rows. The fitted imputer can then be used to
        transform datasets with missing values.

        Args:
            data: The time series dataset to fit the transformer on.
                Trailing null rows will be automatically removed before fitting.
        """
        fit_data = self._drop_empty_features(data.data)
        fit_data = self._remove_trailing_null_rows(fit_data)
        self.imputer.fit(fit_data)

    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        """Transform the input dataset by removing trailing null rows and imputing missing values.

        This method performs a two-step transformation process:
        1. Removes trailing rows that contain null values from the dataset
        2. Applies the configured imputer to fill remaining missing values in the data

        Args:
            data: The input time series dataset containing data to be transformed.
                The dataset's data attribute should be a pandas DataFrame or similar structure.

        Returns:
            The transformed TimeSeriesDataset with trailing null rows removed and
                missing values imputed. The original dataset structure is preserved with
                only the data attribute modified.
        """
        data_cleaned = self._drop_empty_features(data.data)
        data_cleaned = self._remove_trailing_null_rows(data_cleaned)
        data_transformed = cast(pd.DataFrame, self.imputer.transform(data_cleaned))

        return TimeSeriesDataset(data=data_transformed, sample_interval=data.sample_interval)
