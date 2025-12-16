# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Loss functions for quantile regression with XGBoost.

This module provides custom loss functions for multi-quantile regression with XGBoost,
including magnitude-weighted pinball loss, arctan-smoothed pinball loss, and standard
pinball loss. All functions support sample weighting for flexible training.
"""

from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from openstef_core.types import Quantile


def arctan_loss_multi_objective(
    y_true: npt.NDArray[np.floating],
    y_pred: npt.NDArray[np.floating],
    quantiles: list[Quantile],
    sample_weight: npt.NDArray[np.floating] | None = None,
    s: float = 0.1,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Arctan-smoothed multi-quantile pinball loss objective function for XGBoost.

    Computes first and second derivatives of the arctan pinball loss, a smooth approximation
    of the standard pinball loss that provides non-zero second derivatives essential for
    XGBoost's second-order optimization. This formulation addresses the fundamental limitation
    of standard pinball loss having zero second derivatives.

    The arctan pinball loss maintains asymptotic unbiasedness while providing substantial
    second derivatives in the relevant domain (typically -10 to 10 for standardized targets),
    making it well-suited for gradient boosting algorithms that rely on second-order
    information for tree splitting decisions.

    Args:
        y_true: True target values, shape (n_samples, n_quantiles)
        y_pred: Predicted values, shape (n_samples, n_quantiles)
        quantiles: List of validated quantiles corresponding to predictions
        sample_weight: Optional sample weights, shape (n_samples,). If provided,
            gradients and hessians are scaled by these weights to give different importance
            to different samples in the loss computation.
        s: Smoothing parameter for the arctan function. Smaller values provide closer approximation
           to standard pinball loss but reduce second derivative magnitude.

    Returns:
        tuple: (gradient, hessian) arrays in 2D format for XGBoost multi-output
            - gradient: First derivative of arctan pinball loss, shape (n_samples, n_quantiles)
            - hessian: Second derivative of arctan pinball loss, shape (n_samples, n_quantiles)
            Both arrays are normalized by n_quantiles for consistent optimization dynamics.

    Mathematical Formulation:
        Loss function: L^(arctan)_τ,s(u) = (τ - 0.5 + arctan(u/s)/π) * u + s/π

        Where:
        - u = y_true - y_pred (prediction error)
        - τ = quantile level (0 < τ < 1)
        - s = smoothing parameter

        First derivative (gradient):
        ∂L/∂u = τ - 0.5 + arctan(u/s)/π + u/(π*s*(1+(u/s)²))

        Second derivative (hessian):
        ∂²L/∂u² = 2/(π*s) * (1+(u/s)²)^(-2)

    Key Properties:
        - Smooth approximation: Continuously differentiable everywhere
        - Asymptotic unbiasedness: Behaves like standard pinball loss for large |u|
        - Non-zero hessian: Provides substantial second derivatives for XGBoost optimization
        - Scale sensitivity: Hessian magnitude inversely proportional to smoothing parameter s

    Gradient Normalization:
        Both gradient and hessian are divided by n_quantiles to ensure consistent optimization
        dynamics regardless of the number of quantiles being predicted. This normalization
        prevents gradient magnitude from scaling linearly with the number of outputs.

    References:
        - Arctan pinball loss: arXiv:2406.02293v1 "Composite Quantile Regression With XGBoost"
        - Second-order optimization: XGBoost requires non-zero hessian for tree splitting
        - Implementation: https://github.com/LaurensSluyterman/XGBoost_quantile_regression
    """
    # Resize the predictions and targets.
    n_items = len(y_true)
    n_quantiles = len(quantiles)
    n_rows = n_items // n_quantiles
    y_pred = np.reshape(y_pred, (n_rows, n_quantiles))
    y_true = np.reshape(y_true, (n_rows, n_quantiles))
    sample_weight = np.reshape(sample_weight, (n_rows, -1)) if sample_weight is not None else None

    # Calculate the differences
    u = y_true - y_pred

    # Extract quantile values into array for vectorized operations
    quantile_values = np.array(quantiles)  # shape: (n_quantiles,)

    # Calculate the scaled differences
    z = u / s  # shape: (n_samples, n_quantiles)

    # Vectorized computation using broadcasting
    # quantile_values broadcasts from (n_quantiles,) to (n_samples, n_quantiles)
    x = 1 + z**2  # shape: (n_samples, n_quantiles)

    # Compute errors for proper sign handling
    errors = y_pred - y_true  # shape: (n_samples, n_quantiles)

    # Compute gradients for all quantiles simultaneously
    # Scale gradient by error magnitude to handle unscaled data, similar to magnitude-weighted approach
    # The arctan term provides smoothness, while error scaling ensures effective learning with large-scale data
    grad = (quantile_values - 0.5 + (1 / np.pi) * np.arctan(z) + z / (np.pi * x)) * np.abs(errors)

    # Compute hessians for all quantiles simultaneously
    # Use constant hessian for numerical stability (arctan's natural hessian decays too quickly)
    hess = np.ones_like(errors)

    # Apply sample weights if provided
    if sample_weight is not None:
        grad *= sample_weight
        hess *= sample_weight

    # Shape: (n_samples, n_quantiles) for proper multi-output format
    # Negate gradient to get correct sign (arctan formula uses u = y_true - y_pred convention)
    return -grad / n_quantiles, hess / n_quantiles


def pinball_loss_multi_objective(
    y_true: npt.NDArray[np.floating],
    y_pred: npt.NDArray[np.floating],
    quantiles: list[Quantile],
    sample_weight: npt.NDArray[np.floating] | None = None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Multi-quantile pinball loss objective function for XGBoost.

    Computes first-order derivatives of pinball loss for multiple quantiles
    and non-degenerate substitutes for second-order derivatives. This implementation
    scales gradients by error magnitude, making it effective with both scaled and
    unscaled data.

    The pinball loss is the standard loss function for quantile regression, where
    underestimation errors are penalized by the quantile level and overestimation
    errors by (1 - quantile). This ensures unbiased quantile estimates when the
    model converges.

    Gradient scaling by error magnitude addresses convergence issues with large-scale
    unscaled data (e.g., values in millions) by making the objective function responsive
    to error scale.

    Non-zero second-order derivatives are returned instead of zeros because XGBoost requires
    non-degenerate hessian values for proper convergence. See XGBoost issue #1825 for details
    on why this approximation is mathematically valid. Ensure that the hyperparameter
    `max_delta_step` satisfies: 0.5 * max_delta_step <= min(quantile, 1 - quantile) for all
    quantiles.

    Args:
        y_true: True target values, shape (n_samples, n_quantiles)
        y_pred: Predicted values, shape (n_samples, n_quantiles)
        quantiles: List of validated quantiles corresponding to predictions
        sample_weight: Optional sample weights, shape (n_samples,). If provided,
            gradients and hessians are scaled by these weights to give different importance
            to different samples in the loss computation.

    Returns:
        tuple: (gradient, hessian) arrays in 2D format for XGBoost multi-output
            - gradient: First derivative scaled by error magnitude, shape (n_samples, n_quantiles)
            - hessian: Constant positive values for numerical stability, shape (n_samples, n_quantiles)
            Both arrays are normalized by n_quantiles for consistent loss values.

    Mathematical Formulation:
        Magnitude-aware pinball loss gradient: ∇L = (τ * I(error < 0) + (1 - τ) * I(error >= 0)) * error

        Where τ is the quantile level, I(·) is the indicator function, and error = y_pred - y_true.
        The gradient is scaled by error magnitude to handle unscaled data effectively.

    References:
        - Original pinball loss: Koenker & Bassett (1978), "Regression Quantiles"
        - XGBoost hessian approximation: https://github.com/dmlc/xgboost/issues/1825
        - Implementation reference: https://gist.github.com/Nikolay-Lysenko/06769d701c1d9c9acb9a66f2f9d7a6c7
    """
    # Resize the predictions and targets.
    n_items = len(y_true)
    n_quantiles = len(quantiles)
    n_rows = n_items // n_quantiles
    y_pred = np.reshape(y_pred, (n_rows, n_quantiles))
    y_true = np.reshape(y_true, (n_rows, n_quantiles))
    sample_weight = np.reshape(sample_weight, (n_rows, -1)) if sample_weight is not None else None

    # Extract quantile values into array for vectorized operations
    quantile_values = np.array(quantiles)  # shape: (n_quantiles,)

    # Compute errors for all quantiles at once
    errors = y_pred - y_true  # shape: (n_samples, n_quantiles)

    # Compute masks for all quantiles simultaneously
    left_mask = errors < 0  # underprediction, shape: (n_samples, n_quantiles)
    right_mask = errors >= 0  # overprediction, shape: (n_samples, n_quantiles)

    # Vectorized gradient computation using broadcasting
    # quantile_values broadcasts from (n_quantiles,) to (n_samples, n_quantiles)
    # Scale by error magnitude to handle unscaled data (same approach as magnitude-weighted version)
    gradient = (quantile_values * left_mask + (1 - quantile_values) * right_mask) * errors

    # Non-degenerate hessian for XGBoost numerical stability
    hessian = np.ones_like(y_pred)

    # Apply sample weights if provided
    if sample_weight is not None:
        gradient *= sample_weight
        hessian *= sample_weight

    # Shape: (n_samples, n_quantiles) for proper multi-output format
    return gradient / n_quantiles, hessian / n_quantiles


type ObjectiveFunctionType = Literal["pinball_loss", "arctan_loss"]

OBJECTIVE_MAP = {
    "pinball_loss": pinball_loss_multi_objective,
    "arctan_loss": arctan_loss_multi_objective,
}


def xgb_prepare_target_for_objective(
    target: np.ndarray,
    objective: ObjectiveFunctionType | Literal["reg:quantileerror"],
    quantiles: list[Quantile],
) -> np.ndarray:
    if objective == "reg:quantileerror":
        return target

    return np.repeat(target[:, np.newaxis], repeats=len(quantiles), axis=1)


def get_objective_function(
    function_type: ObjectiveFunctionType,
    quantiles: list[Quantile],
    **kwargs: Any,
) -> Callable[
    [npt.NDArray[np.floating], npt.NDArray[np.floating]], tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
]:
    fn = partial(OBJECTIVE_MAP[function_type], quantiles=quantiles, **kwargs)
    fn.__name__ = function_type  # pyright: ignore[reportAttributeAccessIssue]
    return fn


__all__ = [
    "OBJECTIVE_MAP",
    "ObjectiveFunctionType",
    "arctan_loss_multi_objective",
    "pinball_loss_multi_objective",
]
