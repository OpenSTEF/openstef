# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for removing completely empty columns from time series datasets.

This module provides functionality for identifying and removing columns that contain
only missing values from time series data.
"""

import logging
from typing import Any, Self, cast, override

import numpy as np
from pydantic import Field, PrivateAttr

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import TransformNotFittedError
from openstef_core.mixins import State
from openstef_core.transforms import TimeSeriesTransform
from openstef_models.utils.feature_selection import FeatureSelection

_logger = logging.getLogger(__name__)


class EmptyFeatureRemover(BaseConfig, TimeSeriesTransform):
    """Transform that removes columns which are completely empty (all values are missing).

    This transform identifies columns that contain only missing values and removes them
    from the dataset. It respects both NaN values and custom missing value placeholders.

    Args:
        columns: Set of column names to check for emptiness. If None, checks all columns.
        missing_value: The placeholder for missing values. Defaults to np.nan.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> from openstef_models.transforms.general import (
        ...     EmptyFeatureRemover,
        ... )
        >>> from openstef_models.utils.feature_selection import FeatureSelection
        >>> # Create dataset with some empty columns
        >>> data = pd.DataFrame(
        ...     {
        ...         "radiation": [100.0, 110.0, 120.0],
        ...         "temperature": [20.0, 21.0, 22.0],
        ...         "empty_col1": [np.nan, np.nan, np.nan],
        ...         "empty_col2": [np.nan, np.nan, np.nan],
        ...     },
        ...     index=pd.date_range("2025-01-01", periods=3, freq="1h"),
        ... )
        >>> dataset = TimeSeriesDataset(data, timedelta(hours=1))
        >>> # Remove all empty columns
        >>> transform = EmptyFeatureRemover()
        >>> transform.fit(dataset)
        >>> result = transform.transform(dataset)
        >>> list(result.data.columns)
        ['radiation', 'temperature']
        >>> # Only check specific columns
        >>> transform_selective = EmptyFeatureRemover(
        ...     selection=FeatureSelection(include={"empty_col1", "radiation"})
        ... )
        >>> transform_selective.fit(dataset)
        >>> result_selective = transform_selective.transform(dataset)
        >>> "empty_col1" in result_selective.data.columns
        False
        >>> "empty_col2" in result_selective.data.columns  # Not checked, so not removed
        True
    """

    selection: FeatureSelection = Field(
        default=FeatureSelection.ALL,
        description="Features to check for emptiness.",
    )
    missing_value: float = Field(
        default=np.nan,
        description="The placeholder for missing values that should be considered empty",
    )

    _remove_columns: set[str] = PrivateAttr(default_factory=set)  # pyright: ignore[reportUnknownVariableType]
    _is_fitted: bool = PrivateAttr(default=False)

    @property
    @override
    def is_fitted(self) -> bool:
        return self._is_fitted

    @override
    def fit(self, data: TimeSeriesDataset) -> None:
        features = self.selection.resolve(data.feature_names)
        data_subset = data.data[features]

        empty_mask = (
            data_subset.isna().all() if np.isnan(self.missing_value) else (data_subset == self.missing_value).all()
        )

        self._remove_columns = set(data_subset.columns[empty_mask])
        if self._remove_columns:
            _logger.warning(
                "Dropped columns from dataset because they contain only missing values: %s", self._remove_columns
            )

        self._is_fitted = True

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        if not self._is_fitted:
            raise TransformNotFittedError(self.__class__.__name__)

        # Remove empty columns
        result_data = data.data.loc[:, ~data.data.columns.isin(self._remove_columns)]  # pyright: ignore[reportUnknownMemberType]

        return TimeSeriesDataset(data=result_data, sample_interval=data.sample_interval)

    @override
    def to_state(self) -> State:
        return {
            "config": self.model_dump(mode="json"),
            "remove_columns": list(self._remove_columns),
            "is_fitted": self._is_fitted,
        }

    @override
    def from_state(self, state: State) -> Self:
        state = cast(dict[str, Any], state)
        instance = self.model_validate(state["config"])
        instance._remove_columns = set(state["remove_columns"])  # noqa: SLF001
        instance._is_fitted = state["is_fitted"]  # noqa: SLF001
        return instance

    @override
    def features_added(self) -> list[str]:
        return []
