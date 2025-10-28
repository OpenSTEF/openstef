# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Input consistency validation for time series transforms."""

import logging
from typing import Any, Self, cast, override

from pydantic import PrivateAttr

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.validation import validate_required_columns
from openstef_core.exceptions import NotFittedError
from openstef_core.transforms import TimeSeriesTransform


class InputConsistencyChecker(BaseConfig, TimeSeriesTransform):
    """Validates input data consistency during transform operations.

    Ensures that input features match those seen during fitting and maintains
    consistent column ordering. Logs warnings and removes extra columns.

    Invariants:
        - Must be fitted before transform() can be called
        - Validates presence of all features seen during fitting
        - Logs warnings for extra columns not seen during fitting
        - Removes extra columns from output
        - Maintains consistent column ordering in output
    """

    _feature_names: list[str] = PrivateAttr(default_factory=list[str])
    _is_fitted: bool = PrivateAttr(default=False)
    _logger: logging.Logger = PrivateAttr(default=logging.getLogger(__name__))

    @property
    @override
    def is_fitted(self) -> bool:
        return self._is_fitted

    @override
    def fit(self, data: TimeSeriesDataset) -> None:
        self._feature_names = list(data.feature_names)
        self._is_fitted = True

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        validate_required_columns(data.data, self._feature_names)

        extra_columns = set(data.feature_names) - set(self._feature_names)
        if extra_columns:
            self._logger.warning("Input data contains extra columns not seen during fitting: %s", extra_columns)

        # Ensure features are ordered.
        internal_columns = [col for col in data.data.columns if col not in data.feature_names]
        columns_ordered = [*internal_columns, *self._feature_names]

        df = data.data[columns_ordered]
        return data.copy_with(data=df)

    @override
    def to_state(self) -> object:
        return {
            "feature_names": self._feature_names,
            "is_fitted": self._is_fitted,
        }

    @override
    def from_state(self, state: object) -> Self:
        state = cast(dict[str, Any], state)
        self._feature_names = state["feature_names"]
        self._is_fitted = state["is_fitted"]
        return self

    @override
    def features_added(self) -> list[str]:
        return []
