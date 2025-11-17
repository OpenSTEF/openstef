# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Missing values imputation transform for time series datasets.

This module provides functionality for handling missing values in time series data
through various imputation strategies.
"""

import logging
from typing import Any, Literal, cast, override

import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr, model_validator

# This imputer is still experimental for now:
# default parameters or details of behaviour might change without any deprecation cycle.
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 # type: ignore
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.transforms import TimeSeriesTransform
from openstef_models.utils.feature_selection import FeatureSelection

type ImputationStrategy = Literal["mean", "median", "most_frequent", "constant", "iterative"]


def _check_for_empty_columns(data: pd.DataFrame, missing_value: float) -> set[str]:
    """Check for completely empty columns and return their names.

    Args:
        data: DataFrame to check for empty columns
        missing_value: The placeholder for missing values

    Returns:
        Set of column names that are completely empty

    Raises:
        ValueError: If any columns are completely empty
    """
    empty_mask = data.isna().all() if np.isnan(missing_value) else (data == missing_value).all()
    empty_columns = set(data.columns[empty_mask])

    if empty_columns:
        msg = (
            f"Cannot impute completely empty columns: {empty_columns}. "
            "Use EmptyFeatureRemover first to remove these columns."
        )
        raise ValueError(msg)

    return empty_columns


class Imputer(BaseConfig, TimeSeriesTransform):
    """Transform that imputes missing values in specified columns of time series data.

    This transform applies imputation strategies to handle missing values in the dataset.
    It focuses solely on missing value imputation and avoids filling future values by
    preserving trailing NaNs after the last valid value in each column.

    The transform works by:
    1. Validating that selected columns are not completely empty
    2. Applying imputation to the specified columns
    3. Restoring trailing NaNs to preserve time series integrity

    Imputation Strategies:
        - Simple strategies (mean, median, most_frequent, constant): Use statistics
          computed during fit()
        - Iterative strategy: Multivariate imputation (default: BayesianRidge()).
          Leverages relations between features but can be slower and needs parameter tuning.

    Note: If you have completely empty columns, use EmptyFeatureRemover first
    to remove them before applying imputation.

    Example:
    >>> from datetime import timedelta
    >>> import numpy as np
    >>> import pandas as pd
    >>> from openstef_core.datasets import TimeSeriesDataset
    >>> from openstef_models.transforms.general import (
    ...     Imputer,
    ... )
    >>> data = pd.DataFrame(
    ...     {
    ...         "radiation": [100, np.nan, 110, np.nan],
    ...         "temperature": [20, np.nan, 24, 21],
    ...         "wind_speed": [5, 6, np.nan, np.nan]
    ...     },
    ...     index=pd.date_range("2025-01-01", periods=4, freq="1h"),
    ... )
    >>> dataset = TimeSeriesDataset(data, timedelta(hours=1))
    >>> # Apply imputation to all columns (default behavior)
    >>> transform_all = Imputer(imputation_strategy="mean")
    >>> transform_all.fit(dataset)
    >>> result_all = transform_all.transform(dataset)
    >>> # Apply imputation only to specific columns
    >>> from openstef_models.utils.feature_selection import FeatureSelection
    >>> transform_selective = Imputer(
    ...     imputation_strategy="mean",
    ...     selection=FeatureSelection(include={"temperature", "wind_speed"})
    ... )
    >>> transform_selective.fit(dataset)
    >>> result_selective = transform_selective.transform(dataset)
    >>> result_selective.data["temperature"].isna().sum() == 0  # Temperature NaNs filled
    np.True_
    >>> result_selective.data["radiation"].isna().sum() == 2  # Radiation NaNs preserved
    np.True_
    >>> # Use iterative imputation with custom estimator
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> transform_iterative = Imputer(
    ...     imputation_strategy="iterative",
    ...     selection=FeatureSelection(include={"temperature", "radiation"}),
    ...     impute_estimator=RandomForestRegressor(
    ...         n_estimators=2,  # not many trees for test speed
    ...         max_depth=3,  # shallow tree for test speed
    ...         bootstrap=True,
    ...         max_samples=0.5,
    ...         n_jobs=1,
    ...         random_state=0,
    ...     ),
    ...     max_iterations=20,
    ...     tolerance=0.1
    ... )
    >>> transform_iterative.fit(dataset)
    >>> result_iterative = transform_iterative.transform(dataset)
    >>> result_iterative.data["temperature"].isna().sum() == 0  # Temperature NaNs filled
    np.True_
    >>> np.isnan(result_iterative.data["radiation"].iloc[1])  # Radiation first NaN replaced
    np.False_
    >>> np.isnan(result_iterative.data["radiation"].iloc[3]) # Check if trailing NaN is preserved
    np.True_
    >>> result_iterative.data["wind_speed"].isna().sum() == 2  # Wind speed NaNs preserved
    np.True_
    """

    imputation_strategy: ImputationStrategy = Field(
        ...,
        description="The strategy to use for imputation",
    )
    missing_value: float = Field(
        default=np.nan,
        description="The placeholder for missing values that should be imputed",
    )
    fill_value: float | str | None = Field(
        default=None,
        description="Value to use when imputation_strategy is CONSTANT",
    )
    impute_estimator: Any = Field(
        default_factory=BayesianRidge,
        description="Estimator to use for IterativeImputer. Defaults to BayesianRidge.",
    )
    initial_strategy: Literal["mean", "median", "most_frequent", "constant"] = Field(
        default="mean",
        description="Initial imputation strategy for IterativeImputer",
    )
    tolerance: float = Field(
        default=1e-3,
        description="Threshold for differences between consecutive iterations for IterativeImputer convergence",
    )
    max_iterations: int = Field(
        default=40,
        description="Maximum iterations for IterativeImputer",
    )
    selection: FeatureSelection = Field(
        default=FeatureSelection.ALL,
        description=(
            "Features to impute. If strategy is 'iterative', these features are also used as predictors for imputation."
        ),
    )
    fill_future_values: FeatureSelection = Field(
        default=FeatureSelection.NONE,
        description=(
            "Features for which to fill future missing values. "
            "This transform does not fill future missing values by default to preserve time series integrity."
        ),
    )

    _imputer: SimpleImputer | IterativeImputer = PrivateAttr()
    _is_fitted: bool = PrivateAttr(default=False)
    _logger: logging.Logger = PrivateAttr(default=logging.getLogger(__name__))

    @model_validator(mode="after")
    def validate_fill_value_with_strategy(self) -> "Imputer":
        """Validate that fill_value is provided when strategy is CONSTANT.

        Returns:
            The validated model instance.

        Raises:
            ValueError: If imputation_strategy is CONSTANT but fill_value is None.
        """
        if self.imputation_strategy == "constant" and self.fill_value is None:
            raise ValueError("fill_value must be provided when imputation_strategy is CONSTANT")
        return self

    @override
    def model_post_init(self, context: Any) -> None:
        if self.imputation_strategy == "iterative":
            self._imputer = IterativeImputer(
                random_state=0,
                estimator=self.impute_estimator,
                initial_strategy=self.initial_strategy,
                max_iter=self.max_iterations,
                tol=self.tolerance,
            )
        else:
            self._imputer = SimpleImputer(
                strategy=self.imputation_strategy,
                fill_value=self.fill_value,
                missing_values=self.missing_value,
                keep_empty_features=False,
            )
        self._imputer.set_output(transform="pandas")

    @property
    @override
    def is_fitted(self) -> bool:
        return self._is_fitted

    @override
    def fit(self, data: TimeSeriesDataset) -> None:
        features = self.selection.resolve(data.feature_names)
        if not features:
            self._is_fitted = True
            return

        data_subset = data.data[features]

        # Check for completely empty columns and raise an error
        _check_for_empty_columns(data_subset, self.missing_value)

        # Fit sklearn imputer
        self._imputer.fit(data_subset)
        self._is_fitted = True

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)

        features = self.selection.resolve(data.feature_names)
        if not features:
            return data

        # Warn if using iterative imputation with only one feature
        if self.imputation_strategy == "iterative" and len(features) < 2:  # noqa: PLR2004
            self._logger.warning(
                "Iterative imputer with only one feature will fall back to initial_strategy. "
                "Using '%s' for imputation.",
                self.initial_strategy,
            )

        data_subset = data.data[features]
        data_transformed = cast(pd.DataFrame, self._imputer.transform(data_subset))

        # Set imputed trailing NaNs back to NaN since they cannot be reasonably imputed
        fill_future_features = self.fill_future_values.resolve(features)
        no_fill_future_features = set(features) - set(fill_future_features)
        for col in no_fill_future_features:
            last_valid = data_subset[col].last_valid_index()
            data_transformed.loc[data_transformed.index > (last_valid or data_transformed.index[0]), col] = np.nan

        # Recombine transformed columns with untouched columns
        result_data = data.data.copy()
        result_data[features] = data_transformed

        return TimeSeriesDataset(data=result_data, sample_interval=data.sample_interval)

    @override
    def features_added(self) -> list[str]:
        return []
