# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Confidence interval generation for probabilistic forecasts.

This module provides transforms for adding quantile predictions to forecasts based on
learned hour-specific uncertainty patterns from validation data.
"""

from typing import override

from pydantic import BaseModel, Field, PrivateAttr

from openstef_core.datasets import ForecastDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.mixins import Transform


class MinMaxClipper(BaseModel, Transform[ForecastDataset, ForecastDataset]):
    """Uses Bayes Rule to estimate the probability that the prediciton for a given quantile is correct."""

    _is_fitted: bool = PrivateAttr(default=False)
    _min: float = PrivateAttr()
    _max: float = PrivateAttr()
    threshold: float = Field(default=0.10, description="Threshold for clipping predictions outside the min-max range.")

    @property
    @override
    def is_fitted(self) -> bool:
        return self._is_fitted

    @override
    def fit(self, data: ForecastDataset) -> None:
        if data.target_series is None:
            raise ValueError("Input data must contain target series for error computation.")
        _min = data.target_series.min()
        _max = data.target_series.max()
        range_span = _max - _min
        self._min = _min - self.threshold * range_span
        self._max = _max + self.threshold * range_span
        self._is_fitted = True

    @override
    def transform(self, data: ForecastDataset) -> ForecastDataset:
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)

        quantile_columns = [quantile.format() for quantile in data.quantiles]

        clipped_data = data.data.copy()
        clipped_data[quantile_columns] = clipped_data[quantile_columns].clip(lower=self._min, upper=self._max)

        return data._copy_with_data(data=clipped_data)  # noqa: SLF001 - safe - invariant is preserved


__all__ = ["MinMaxClipper"]
