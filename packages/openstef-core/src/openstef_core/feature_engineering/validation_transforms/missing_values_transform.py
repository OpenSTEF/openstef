# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Missing values imputation and data quality transform for time series datasets.

This module provides functionality for handling missing values in time series data
through various imputation strategies and data cleaning operations.
"""

from enum import Enum

import numpy as np
import pandas as pd
from pydantic import Field
from sklearn.impute import SimpleImputer

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.transforms import TimeSeriesTransform


class ImputationStrategy(Enum):
    """Enumeration of available imputation strategies for missing values."""

    MEAN = "mean"
    MEDIAN = "median"
    MOST_FREQUENT = "most_frequent"
    CONSTANT = "constant"


class MissingValuesTransformConfig(BaseConfig):
    """Configuration class for MissingValuesTransform."""

    missing_value: float = Field(
        default=np.nan, description="The placeholder for missing values that should be imputed."
    )
    imputation_strategy: ImputationStrategy = Field(description="The strategy to use for imputation.")
    fill_value: int | float | str | None = Field(
        default=None, description="Value to use when imputation_strategy is CONSTANT."
    )
    no_fill_future_values_features: list[str] = Field(
        default_factory=list, description="List of feature names for which trailing NaN values should not be filled."
    )


class MissingValuesTransform(TimeSeriesTransform):
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
    ...     MissingValuesTransformConfig,
    ... )
    >>> data = pd.DataFrame(
    ...     {
    ...         "load": [100, np.nan, 110, np.nan],
    ...         "temperature": [20, np.nan, 24, 21],
    ...         "wind_speed": [5, 6, np.nan, np.nan],
    ...         "empty_feature": [np.nan, np.nan, np.nan, np.nan]
    ...     },
    ...     index=pd.date_range("2025-01-01", periods=4, freq="1h"),
    ... )
    >>> dataset = TimeSeriesDataset(data, timedelta(hours=1))
    >>> config = MissingValuesTransformConfig(
    ...     imputation_strategy=ImputationStrategy.MEAN,
    ...     missing_value=np.nan,
    ...     no_fill_future_values_features=["load"]
    ... )
    >>> transform = MissingValuesTransform(config)
    >>> transform.fit(dataset)
    >>> result = transform.transform(dataset)
    >>> int(result.data.isna().sum().sum()) == 0
    True
    >>> result.data.shape[0] == 3 # Last row dropped
    True
    >>> result.data.shape[1] == 3 # Empty feature column dropped
    True
    """

    def __init__(self, config: MissingValuesTransformConfig):
        """Initialize the MissingValuesTransform with the given configuration."""
        self.config = config

        self.imputer_: SimpleImputer = SimpleImputer(
            strategy=self.config.imputation_strategy.value,
            fill_value=self.config.fill_value,
            missing_values=self.config.missing_value,
            keep_empty_features=False,
        )
        self.imputer_.set_output(transform="pandas")

    @staticmethod
    def _determine_trailing_null_rows(x: pd.DataFrame) -> pd.Series:
        """Determine rows with trailing null values in a DataFrame.

        Args:
            x: DataFrame to check for trailing null values

        Returns:
            Boolean Series where True indicates rows to keep (non-trailing null rows)
        """
        return ~x.bfill().isna().any(axis="columns")

    def _remove_trailing_null_rows(self, data: TimeSeriesDataset) -> pd.DataFrame:
        """Remove trailing rows that contain null values in specified features.

        This method removes rows from the end of the dataset where specified features
        (defined in config.no_fill_future_values_features) contain null values. This
        is useful for cleaning datasets where future values should not be filled and
        trailing nulls indicate incomplete data.

        Args:
            data (TimeSeriesDataset): The input time series dataset containing the data
                to be processed.

        Returns:
            pd.DataFrame: A DataFrame with trailing null rows removed for the specified
                features. If no features are configured for checking or none of the
                configured features exist in the data, returns the original data unchanged.
        """
        if not self.config.no_fill_future_values_features:
            return data.data

        features_to_check = [f for f in self.config.no_fill_future_values_features if f in data.data.columns]
        if not features_to_check:
            return data.data

        subset_df = data.data[features_to_check]
        non_trailing_null_rows = self._determine_trailing_null_rows(subset_df)
        return data.data.loc[non_trailing_null_rows]

    def fit(self, data: TimeSeriesDataset) -> None:
        """Fit the missing values transformer to the provided time series data.

        This method prepares the imputer by fitting it to the training data after
        removing any trailing null rows. The fitted imputer can then be used to
        transform datasets with missing values.

        Args:
            data (TimeSeriesDataset): The time series dataset to fit the transformer on.
                Trailing null rows will be automatically removed before fitting.
        """
        fit_data = self._remove_trailing_null_rows(data)
        self.imputer_.fit(fit_data)

    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        """Transform the input dataset by removing trailing null rows and imputing missing values.

        This method performs a two-step transformation process:
        1. Removes trailing rows that contain null values from the dataset
        2. Applies the configured imputer to fill remaining missing values in the data

        Args:
            data (TimeSeriesDataset): The input time series dataset containing data to be transformed.
                The dataset's data attribute should be a pandas DataFrame or similar structure.

        Returns:
            TimeSeriesDataset: The transformed dataset with trailing null rows removed and
                missing values imputed. The original dataset structure is preserved with
                only the data attribute modified.
        """
        data.data = self._remove_trailing_null_rows(data)
        data.data = self.imputer_.transform(data.data)
        return data
