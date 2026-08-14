# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Asymmetric conformal quantile calibration without external dependencies."""

from typing import override

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr

from openstef_core.datasets import ForecastDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.mixins import Transform
from openstef_core.types import Quantile

MEDIAN_QUANTILE = Quantile(0.5)


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
        min_calibration_samples: Minimum number of valid calibration pairs required
            before fitting a correction for a quantile. Quantiles with fewer valid
            pairs are left unchanged; if all quantiles are skipped, fitting becomes
            a no-op calibrator.
    """

    quantiles: list[Quantile] | None = Field(default=None)
    conformalize_median: bool = Field(default=False)
    min_calibration_samples: int = Field(default=100, ge=1)

    _corrections: dict[str, float] = PrivateAttr(default_factory=dict)
    _is_fitted: bool = PrivateAttr(default=False)

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
        if not quantiles_to_fit:
            raise ValueError("No quantiles found to calibrate.")

        self._is_fitted = False
        actuals = data.target_series.to_numpy()
        self._corrections = {}
        if np.count_nonzero(~np.isnan(actuals)) < self.min_calibration_samples:
            self._is_fitted = True
            return

        for quantile in quantiles_to_fit:
            column = quantile.format()
            if column not in data.data.columns:
                continue
            if quantile == MEDIAN_QUANTILE and not self.conformalize_median:
                continue

            predictions = data.data[column].to_numpy()
            valid = ~(np.isnan(predictions) | np.isnan(actuals))
            predictions_valid = predictions[valid]
            actuals_valid = actuals[valid]
            if predictions_valid.size < self.min_calibration_samples:
                continue

            if quantile < MEDIAN_QUANTILE:
                scores = predictions_valid - actuals_valid
                self._corrections[column] = -_conformal_quantile(scores, level=1.0 - quantile)
            else:
                scores = actuals_valid - predictions_valid
                self._corrections[column] = _conformal_quantile(scores, level=quantile)

        self._is_fitted = True


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
