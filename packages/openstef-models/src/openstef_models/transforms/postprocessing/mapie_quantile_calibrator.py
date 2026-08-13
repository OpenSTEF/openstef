# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Conformal quantile calibration for probabilistic forecasts."""

from typing import Any, override

import numpy as np
import pandas as pd
from mapie.conformity_scores import BaseRegressionScore
from mapie.regression import SplitConformalRegressor
from pydantic import BaseModel, Field, PrivateAttr
from sklearn.base import BaseEstimator, RegressorMixin

from openstef_core.datasets import ForecastDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.mixins import Transform
from openstef_core.types import Quantile


class _StoredPredictionEstimator(BaseEstimator, RegressorMixin):
    """Expose a stored forecast column through MAPIE's estimator interface."""

    def __init__(self, prediction_column: str) -> None:
        self.prediction_column = prediction_column
        self.fitted_ = True

    def fit(self, x: object, y: object) -> "_StoredPredictionEstimator":
        del x, y
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.prediction_column].to_numpy()


class _SignedResidualScore(BaseRegressionScore):
    """Use signed residuals to calibrate one quantile at a time."""

    def __init__(self) -> None:
        super().__init__(sym=False)

    @override
    def get_signed_conformity_scores(self, y: np.ndarray, y_pred: np.ndarray, **kwargs: Any) -> np.ndarray:
        del kwargs
        return y - y_pred

    @override
    def get_estimation_distribution(
        self, y_pred: np.ndarray, conformity_scores: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        del kwargs
        return y_pred + conformity_scores


class MapieQuantileCalibrator(BaseModel, Transform[ForecastDataset, ForecastDataset]):
    """Conformalize forecast predictions independently for each quantile with MAPIE.

    For each requested quantile, MAPIE computes signed calibration residuals. The
    quantile-specific conformal correction is then added to that forecast column.
    This deliberately calibrates individual quantiles rather than a joint interval.

    Args:
        quantiles: Individual quantiles to calibrate. If None, uses all quantiles
            found in the calibration data.

    Example:
        >>> calibrator = MapieQuantileCalibrator(
        ...     quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)]
        ... )
        >>> calibrator.fit(validation_data)  # doctest: +SKIP
        >>> calibrated = calibrator.transform(new_predictions)  # doctest: +SKIP
    """

    quantiles: list[Quantile] | None = Field(default=None)

    _calibrators: dict[str, SplitConformalRegressor] = PrivateAttr(default_factory=dict)
    _prediction_estimators: dict[str, _StoredPredictionEstimator] = PrivateAttr(default_factory=dict)
    _quantile_levels: dict[str, float] = PrivateAttr(default_factory=dict)
    _corrections: dict[str, float] = PrivateAttr(default_factory=dict)
    _quantile_columns: list[str] = PrivateAttr(default_factory=list[str])
    _is_fitted: bool = PrivateAttr(default=False)

    @property
    @override
    def is_fitted(self) -> bool:
        return self._is_fitted

    @override
    def fit(self, data: ForecastDataset) -> None:
        if data.target_series is None:
            raise ValueError("Input data must contain target series for calibration.")

        quantiles_to_fit = self.quantiles if self.quantiles is not None else data.quantiles
        quantile_columns = self._validate_quantiles(quantiles_to_fit)
        missing_columns = [column for column in quantile_columns if column not in data.data.columns]
        if missing_columns:
            missing_columns_message = f"Quantile columns not found in data: {missing_columns}."
            raise ValueError(missing_columns_message)

        calibration_data = data.data.loc[:, [*quantile_columns, data.target_column]]
        calibration_data = calibration_data.dropna()
        if calibration_data.empty:
            raise ValueError("No valid data points for calibration.")

        actuals = calibration_data[data.target_column].to_numpy()
        quantile_levels = {Quantile(float(quantile)).format(): float(quantile) for quantile in quantiles_to_fit}
        self._calibrators = {}
        self._prediction_estimators = {}
        self._quantile_levels = quantile_levels
        self._corrections = {}
        for column in quantile_columns:
            estimator = _StoredPredictionEstimator(column)
            calibrator = SplitConformalRegressor(
                estimator=estimator,
                confidence_level=1 - quantile_levels[column],
                conformity_score=_SignedResidualScore(),
                prefit=True,
            )
            calibrator.conformalize(calibration_data, actuals)
            self._calibrators[column] = calibrator
            self._prediction_estimators[column] = estimator
            n_samples = len(calibrator.conformity_scores)
            correction_level = min(quantile_levels[column] * (n_samples + 1) / n_samples, 1.0)
            self._corrections[column] = float(
                np.quantile(calibrator.conformity_scores, correction_level, method="higher")
            )
        self._quantile_columns = quantile_columns
        self._is_fitted = True

    @staticmethod
    def _validate_quantiles(quantiles: list[Quantile]) -> list[str]:
        values = sorted(float(quantile) for quantile in quantiles)
        if not values or any(value <= 0 or value >= 1 for value in values) or len(set(values)) != len(values):
            raise ValueError("MAPIE requires one or more unique quantiles strictly between 0 and 1.")
        return [Quantile(value).format() for value in values]

    @override
    def transform(self, data: ForecastDataset) -> ForecastDataset:
        if not self._is_fitted or not self._calibrators:
            raise NotFittedError(self.__class__.__name__)

        predictions = data.data.copy(deep=False)
        for column in self._calibrators:
            prediction = self._prediction_estimators[column].predict(data.data)
            predictions[column] = prediction + self._corrections[column]

        # Independent corrections can introduce quantile crossings. Restore the
        # invariant expected from a quantile forecast after applying them.
        predictions[self._quantile_columns] = np.sort(
            predictions[self._quantile_columns].to_numpy(),
            axis=1,
        )
        return data._copy_with_data(data=predictions)  # noqa: SLF001 - safe - invariant preserved


__all__ = ["MapieQuantileCalibrator"]
