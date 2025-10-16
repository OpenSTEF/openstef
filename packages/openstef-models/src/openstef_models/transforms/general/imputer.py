# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Missing values imputation transform for time series datasets.

This module provides functionality for handling missing values in time series data
through various imputation strategies.
"""

from typing import Any, Literal, Self, cast, override

import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr, model_validator
from sklearn.impute import SimpleImputer

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import TransformNotFittedError
from openstef_core.mixins import State
from openstef_core.transforms import TimeSeriesTransform

type ImputationStrategy = Literal["mean", "median", "most_frequent", "constant"]


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
    >>> transform_selective = Imputer(
    ...     imputation_strategy="mean",
    ...     columns={"temperature", "wind_speed"}
    ... )
    >>> transform_selective.fit(dataset)
    >>> result_selective = transform_selective.transform(dataset)
    >>> result_selective.data["temperature"].isna().sum() == 0  # Temperature NaNs filled
    np.True_
    >>> result_selective.data["radiation"].isna().sum() == 2  # Radiation NaNs preserved
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
    columns: set[str] | None = Field(
        default=None,
        description="List of column names to apply imputation to. If None, applies to all columns.",
    )

    _imputer: SimpleImputer = PrivateAttr()
    _is_fitted: bool = PrivateAttr(default=False)
    _transform_columns: set[str] = PrivateAttr()

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
        transform_columns = (self.columns or set(data.data.columns)).intersection(data.data.columns)

        if not transform_columns:
            self._is_fitted = True
            self._transform_columns = set()
            return

        data_subset = data.data[list(transform_columns)]

        # Check for completely empty columns and raise an error
        _check_for_empty_columns(data_subset, self.missing_value)

        # Fit sklearn imputer
        self._imputer.fit(data_subset)
        self._is_fitted = True
        self._transform_columns = transform_columns

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        if not self._is_fitted:
            raise TransformNotFittedError(self.__class__.__name__)

        transform_columns = self._transform_columns.intersection(data.data.columns)
        if not transform_columns:
            return data

        data_subset = data.data[list(transform_columns)]
        data_transformed = cast(pd.DataFrame, self._imputer.transform(data_subset))

        # Set imputed trailing NaNs back to NaN since they cannot be reasonably imputed
        for col in data_transformed.columns:
            last_valid = data_subset[col].last_valid_index()
            data_transformed.loc[data_transformed.index > (last_valid or data_transformed.index[0]), col] = np.nan

        # Recombine transformed columns with untouched columns
        result_data = data.data.copy()
        result_data[list(transform_columns)] = data_transformed

        return TimeSeriesDataset(data=result_data, sample_interval=data.sample_interval)

    @override
    def to_state(self) -> State:
        return cast(
            State,
            {
                "config": self.model_dump(mode="json"),
                "imputer": self._imputer.__getstate__(),  # pyright: ignore[reportUnknownMemberType]
                "is_fitted": self._is_fitted,
                "transform_columns": list(self._transform_columns),
            },
        )

    @override
    def from_state(self, state: State) -> Self:
        state = cast(dict[str, Any], state)
        instance = self.model_validate(state["config"])
        instance._imputer.__setstate__(state["imputer"])  # pyright: ignore[reportUnknownMemberType]  # noqa: SLF001
        instance._is_fitted = state["is_fitted"]  # noqa: SLF001
        instance._transform_columns = set(state["transform_columns"])  # noqa: SLF001
        return instance
