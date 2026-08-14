# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Asymmetric conformal quantile calibration without external dependencies."""

from typing import override

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr, field_validator

from openstef_core.datasets import ForecastDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.mixins import Transform
from openstef_core.types import Quantile

MEDIAN_QUANTILE = 0.5


def _conformal_quantile(scores: np.ndarray, level: float) -> float:
    """Return the finite-sample conformal quantile of one-sided scores."""
    n_samples = scores.size
    if n_samples == 0:
        return 0.0
    adjusted_level = min(level * (n_samples + 1) / n_samples, 1.0)
    return float(np.quantile(scores, adjusted_level, method="higher"))


class ConformalizedQuantileCalibrator(BaseModel, Transform[ForecastDataset, ForecastDataset]):
    """Apply asymmetric split-conformal corrections to forecast quantiles.

    Lower quantiles are corrected using lower-tail scores and upper quantiles
    using upper-tail scores. The median is left unchanged by default.

    Args:
        quantiles: Quantiles to calibrate. If None, all input quantiles are used.
        conformalize_median: Whether to apply the upper-tail correction to P50.
    """

    quantiles: list[Quantile] | None = Field(default=None)
    conformalize_median: bool = Field(default=False)
    min_calibration_samples: int = Field(default=100, ge=1)

    _corrections: dict[str, float] = PrivateAttr(default_factory=dict)
    _is_fitted: bool = PrivateAttr(default=False)

    @field_validator("quantiles")
    @classmethod
    def _validate_configured_quantiles(cls, quantiles: list[Quantile] | None) -> list[Quantile] | None:
        """Validate explicitly configured quantiles when the model is created."""
        if quantiles is not None:
            cls._validate_quantiles(quantiles)
        return quantiles

    @property
    @override
    def is_fitted(self) -> bool:
        """Return whether calibration corrections have been fitted."""
        return self._is_fitted

    @override
    def fit(self, data: ForecastDataset) -> None:
        """Estimate one-sided conformal corrections from forecast errors."""
        if data.target_series is None:
            raise ValueError("Input data must contain target series for calibration.")

        quantiles_to_fit = self.quantiles if self.quantiles is not None else data.quantiles
        quantile_columns = self._validate_quantiles(quantiles_to_fit)
        missing_columns = [column for column in quantile_columns if column not in data.data.columns]
        if missing_columns:
            missing_columns_message = f"Quantile columns not found in data: {missing_columns}."
            raise ValueError(missing_columns_message)

        actuals = data.target_series.to_numpy()
        self._corrections = {}
        if np.count_nonzero(~np.isnan(actuals)) < self.min_calibration_samples:
            self._is_fitted = True
            return

        for column in quantile_columns:
            quantile = float(Quantile.parse(column))
            if quantile == MEDIAN_QUANTILE and not self.conformalize_median:
                continue

            predictions = data.data[column].to_numpy()
            valid = ~(np.isnan(predictions) | np.isnan(actuals))
            predictions_valid = predictions[valid]
            actuals_valid = actuals[valid]
            if predictions_valid.size == 0:
                no_data_message = f"No valid data points for quantile {column}."
                raise ValueError(no_data_message)

            if quantile < MEDIAN_QUANTILE:
                scores = predictions_valid - actuals_valid
                self._corrections[column] = -_conformal_quantile(scores, level=1.0 - quantile)
            else:
                scores = actuals_valid - predictions_valid
                self._corrections[column] = _conformal_quantile(scores, level=quantile)

        self._is_fitted = True

    @staticmethod
    def _validate_quantiles(quantiles: list[Quantile]) -> list[str]:
        """Validate quantile levels and return canonical column names."""
        values = sorted(float(quantile) for quantile in quantiles)
        if not values or any(value <= 0 or value >= 1 for value in values) or len(set(values)) != len(values):
            raise ValueError("Conformal calibration requires unique quantiles strictly between 0 and 1.")
        return [Quantile(value).format() for value in values]

    @override
    def transform(self, data: ForecastDataset) -> ForecastDataset:
        """Apply fitted corrections without changing quantile ordering."""
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)

        predictions = data.data.copy(deep=False)
        for column, correction in self._corrections.items():
            if column in predictions.columns:
                predictions[column] = predictions[column] + correction
        return data._copy_with_data(data=predictions)  # noqa: SLF001 - safe transform copy


__all__ = ["ConformalizedQuantileCalibrator"]
