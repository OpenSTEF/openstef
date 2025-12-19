# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Isotonic quantile calibration for probabilistic forecasts.

This module provides transforms for calibrating quantile predictions using isotonic
regression to ensure predicted quantiles match observed quantile levels.
"""

from typing import Literal, override

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr
from sklearn.isotonic import IsotonicRegression

from openstef_core.datasets import ForecastDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.mixins import Transform
from openstef_core.types import Quantile


class IsotonicQuantileCalibrator(BaseModel, Transform[ForecastDataset, ForecastDataset]):
    """Calibrate quantile predictions using isotonic regression.

    This transform learns a monotonic mapping from predicted quantile values to
    calibrated values, ensuring that predicted quantiles match observed empirical
    quantiles in the validation data.

    How it works:
        1. **Learning phase (fit)**:
           - For each quantile (e.g., P10, P50, P90):
             * Sorts predictions and actual values together
             * Optionally estimates local quantiles using adaptive windowing
             * Fits isotonic regression that maps predictions to actuals
             * Enforces monotonicity: higher predictions → higher calibrated values

        2. **Prediction phase (transform)**:
           - Applies learned isotonic mapping to each quantile
           - Out-of-bounds predictions are handled per configuration
           - Preserves monotonic ordering across quantiles

    Args:
        quantiles: Quantiles to calibrate (e.g., [0.1, 0.5, 0.9]).
            If None, calibrates all quantiles found in the validation data.
        y_min: Minimum bound for calibrated predictions.
            If specified, all calibrated values will be at least y_min.
        y_max: Maximum bound for calibrated predictions.
            If specified, all calibrated values will be at most y_max.
        out_of_bounds: How to handle predictions outside training range:
            - 'clip' (default): Extend to closest training value
            - 'nan': Return NaN for out-of-bounds predictions
            - 'raise': Raise error for out-of-bounds predictions
        use_local_quantile_estimation: Whether to estimate target quantiles from
            local neighborhoods before fitting isotonic regression. Useful when
            bias varies across prediction range.
        window_size: Window size for local quantile estimation. If None, uses
            adaptive sizing (max(5, n/10)). Only used when
            use_local_quantile_estimation=True.

    Invariants:
        - Validation data must contain both predictions and target values
        - Each quantile must have sufficient valid (non-NaN) data points
        - Calibrated quantiles maintain monotonic ordering if input quantiles do

    Example:
        >>> from openstef_core.types import Quantile
        >>> # Basic calibration
        >>> calibrator = IsotonicQuantileCalibrator(
        ...     quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)]
        ... )
        >>> calibrator.fit(validation_data)  # doctest: +SKIP
        >>> calibrated = calibrator.transform(new_predictions)  # doctest: +SKIP
        >>>
        >>> # With local quantile estimation and bounds
        >>> calibrator = IsotonicQuantileCalibrator(
        ...     quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
        ...     use_local_quantile_estimation=True,
        ...     y_min=0.0,  # Ensure non-negative predictions
        ...     out_of_bounds='clip'
        ... )
        >>> calibrator.fit(validation_data)  # doctest: +SKIP
        >>> calibrated = calibrator.transform(new_predictions)  # doctest: +SKIP

    """

    quantiles: list[Quantile] | None = Field(default=None)
    y_min: float | None = Field(default=None)
    y_max: float | None = Field(default=None)
    out_of_bounds: Literal["nan", "clip", "raise"] = Field(default="clip")

    use_local_quantile_estimation: bool = Field(
        default=False,
        description="First estimate quantile from neighbours within window before applying isotonic regression. \
        Useful when bias varies over prediction range.",
    )
    window_size: int | None = Field(
        default=None,
        description="Window size for local estimation. If None, uses adaptive sizing (n/10).",
    )

    _calibrators: dict[str, IsotonicRegression] = PrivateAttr(default_factory=dict[str, IsotonicRegression])
    _is_fitted: bool = PrivateAttr(default=False)

    @property
    @override
    def is_fitted(self) -> bool:
        return self._is_fitted

    @override
    def fit(self, data: ForecastDataset) -> None:
        if data.target_series is None:
            raise ValueError("Input data must contain target series for calibration.")

        # Determine which quantiles to calibrate
        quantiles_to_fit = self.quantiles if self.quantiles is not None else data.quantiles

        if not quantiles_to_fit:
            raise ValueError("No quantiles found to calibrate.")

        # Fit isotonic regression for each quantile
        for quantile in quantiles_to_fit:
            column_name = quantile.format()
            if column_name not in data.data.columns:
                missing_column_message = f"Quantile column '{column_name}' not found in data."
                raise ValueError(missing_column_message)

            # Get predictions and actuals
            predictions = data.data[column_name].to_numpy()  # type: ignore
            actuals = data.target_series.to_numpy()  # type: ignore

            # Remove NaN values (required for IsotonicRegression)
            mask = ~(np.isnan(predictions) | np.isnan(actuals))
            predictions_clean = predictions[mask]
            actuals_clean = actuals[mask]

            if len(predictions_clean) == 0:
                no_data_available_error = f"No valid data points for quantile {column_name}"
                raise ValueError(no_data_available_error)

            if self.use_local_quantile_estimation:
                actuals_clean = self._estimate_local_quantiles(predictions_clean, actuals_clean, float(quantile))

            # Fit isotonic regression
            calibrator = IsotonicRegression(
                y_min=self.y_min,
                y_max=self.y_max,
                increasing=True,
                out_of_bounds=self.out_of_bounds,
            )
            calibrator.fit(predictions_clean, actuals_clean)
            self._calibrators[column_name] = calibrator

        self._is_fitted = True

    def _estimate_local_quantiles(
        self, predictions: np.ndarray, actuals: np.ndarray, quantile_level: float
    ) -> np.ndarray:
        """Estimate local quantiles using adaptive windowing.

        Returns:
            Array of locally estimated quantile values matching the input order.
        """
        # Sort by predicted values
        sort_indices = np.argsort(predictions)
        sorted_predictions = predictions[sort_indices]
        sorted_actuals = actuals[sort_indices]

        # Adaptive window size
        n_samples = len(sorted_predictions)
        window_size = self.window_size if self.window_size is not None else max(5, n_samples // 10)

        # Compute local quantiles
        calibrated_values = np.zeros(n_samples)
        for j in range(n_samples):
            start_idx = max(0, j - window_size // 2)
            end_idx = min(n_samples, j + window_size // 2 + 1)
            window_actuals = sorted_actuals[start_idx:end_idx]
            calibrated_values[j] = np.quantile(window_actuals, quantile_level)

        # Unsort to match original order
        unsort_indices = np.argsort(sort_indices)
        return calibrated_values[unsort_indices]

    @override
    def transform(self, data: ForecastDataset) -> ForecastDataset:
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)

        # Apply calibration to each quantile
        predictions = data.data.copy(deep=False)
        for column_name, calibrator in self._calibrators.items():
            if column_name in predictions.columns:
                predictions[column_name] = calibrator.predict(predictions[column_name].to_numpy())  # type: ignore

        return data._copy_with_data(data=predictions)  # noqa: SLF001 - safe - invariant preserved


__all__ = ["IsotonicQuantileCalibrator"]
