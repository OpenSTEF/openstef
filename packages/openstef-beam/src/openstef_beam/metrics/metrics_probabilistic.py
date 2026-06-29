# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Metrics for forecasts that predict probability distributions instead of single values.

Unlike deterministic forecasts that predict one value (e.g., "load will be 100 MW"),
probabilistic forecasts predict a range of possible outcomes with their likelihoods
(e.g., "80% chance load will be between 90-110 MW"). These metrics evaluate both
how accurate these probability estimates are and how well-calibrated they are.

Key concepts:

    - Calibration: Do 90% prediction intervals actually contain the true value 90% of the time?
    - Sharpness: How narrow are the prediction intervals (more precise is better)?
    - Proper scoring: Metrics that reward honest probability estimates over gaming the system.
"""

from typing import Literal

import numpy as np
import numpy.typing as npt

from openstef_beam.metrics.metrics_deterministic import pinball_loss
from openstef_beam.metrics.metrics_helpers import represented_interval_weights
from openstef_core.types import Quantile

_Q_05 = Quantile(0.05)
_Q_95 = Quantile(0.95)

type QuantileWeightingMethod = Literal["interval", "uniform"]


def crps(
    y_true: npt.NDArray[np.floating],
    y_pred: npt.NDArray[np.floating],
    quantiles: list[Quantile],
    sample_weights: npt.NDArray[np.floating] | None = None,
    method: QuantileWeightingMethod = "interval",
) -> float:
    """Calculate the Continuous Ranked Probability Score (CRPS) for probabilistic forecasts.

    CRPS is a proper scoring rule that measures the quality of probabilistic forecasts.
    It generalizes the absolute error to distributional forecasts and is expressed
    in the same units as the forecast variable.

    Args:
        y_true: Observed values with shape (num_samples,).
        y_pred: Predicted quantiles with shape (num_samples, num_quantiles).
            Each row contains quantile predictions for the corresponding observation.
        quantiles: Quantile levels with shape (num_quantiles,).
            Must be sorted in ascending order and contain values in [0, 1].
        sample_weights: Optional weights for each sample with shape (num_samples,).
            If None, all samples are weighted equally.
        method: Quantile weighting scheme to use. "interval" (default) weights each
            quantile by the probability interval it represents on [0, 1], giving more
            stability across quantile sets with different spacing. "uniform"
            weights every quantile equally.

    Returns:
        The weighted average CRPS across all samples. Lower values indicate
        better forecast quality.

    Example:
        Evaluate quantile forecasts for energy load

        >>> import numpy as np
        >>> y_true = np.array([100, 120, 110])
        >>> quantiles = np.array([0.1, 0.5, 0.9])
        >>> y_pred = np.array([[95, 100, 105],    # Quantiles for first observation
        ...                    [115, 120, 125],   # Quantiles for second observation
        ...                    [105, 110, 115]])  # Quantiles for third observation
        >>> score = crps(y_true, y_pred, quantiles)
        >>> isinstance(score, float)
        True

    Note:
        CRPS reduces to the absolute error when comparing point forecasts
        (single quantile). For well-calibrated forecasts, CRPS approximately
        equals half the expected absolute error of random forecasts.

        The factor 2.0 converts the (weighted) average pinball loss into CRPS:
        CRPS equals twice the integral of the pinball loss over all quantile
        levels in [0, 1]. The (weighted) average of the per-quantile pinball losses
        approximates that integral, and multiplying by 2 recovers the CRPS scale.
    """
    per_quantile_loss = np.array(
        [
            pinball_loss(y_true, y_pred[:, i], quantile=quantile, sample_weights=sample_weights)
            for i, quantile in enumerate(quantiles)
        ]
    )

    if method == "interval":
        quantile_weights = represented_interval_weights(quantiles)
    else:
        quantile_weights = np.full(len(quantiles), 1.0 / len(quantiles))

    return float(2.0 * (quantile_weights @ per_quantile_loss))


def rcrps(
    y_true: npt.NDArray[np.floating],
    y_pred: npt.NDArray[np.floating],
    quantiles: list[Quantile],
    lower_quantile: Quantile = _Q_05,
    upper_quantile: Quantile = _Q_95,
    sample_weights: npt.NDArray[np.floating] | None = None,
    method: QuantileWeightingMethod = "interval",
) -> float:
    """Calculate the relative Continuous Ranked Probability Score (rCRPS).

    The rCRPS normalizes the CRPS by the range of observed values, making it
    scale-invariant and suitable for comparing forecast quality across different
    datasets or time periods with varying magnitudes.

    Args:
        y_true: Observed values with shape (num_samples,).
        y_pred: Predicted quantiles with shape (num_samples, num_quantiles).
        quantiles: Quantile levels with shape (num_quantiles,). Must be sorted
            in ascending order and contain values in [0, 1].
        lower_quantile: Lower quantile for range calculation. Must be in [0, 1].
        upper_quantile: Upper quantile for range calculation. Must be in [0, 1]
            and greater than lower_quantile.
        sample_weights: Optional weights for each sample with shape (num_samples,).
        method: Quantile weighting scheme to use, either "interval" (default) or "uniform".

    Returns:
        The relative CRPS as a float. Returns NaN if the range between
        quantiles is zero.

    Example:
        Compare forecast quality across different scales

        >>> import numpy as np
        >>> # High load period
        >>> y_true_high = np.array([1000, 1200, 1100])
        >>> quantiles = np.array([0.1, 0.5, 0.9])
        >>> y_pred_high = np.array([[950, 1000, 1050],
        ...                         [1150, 1200, 1250],
        ...                         [1050, 1100, 1150]])
        >>> rcrps_high = rcrps(y_true_high, y_pred_high, quantiles)
        >>> isinstance(rcrps_high, float)
        True

    Note:
        rCRPS allows fair comparison of forecast quality between periods with
        different load levels, such as summer vs. winter energy demand.
    """
    y_range = np.quantile(y_true, q=upper_quantile) - np.quantile(y_true, q=lower_quantile)
    if y_range == 0:
        return float("NaN")

    return float(crps(y_true, y_pred, quantiles, sample_weights, method=method) / y_range)


def observed_probability(
    y_true: npt.NDArray[np.floating],
    y_pred: npt.NDArray[np.floating],
) -> float:
    """Calculate the observed probability (empirical quantile) of predicted values.

    This function determines what quantile the predicted values correspond to
    based on the observed outcomes. For well-calibrated forecasts, a prediction
    at the p-th quantile should have approximately p fraction of observations below it.

    Args:
        y_true: Observed values with shape (num_samples,).
        y_pred: Predicted values with shape (num_samples,). These are typically
            predictions from a specific quantile level.

    Returns:
        The empirical quantile level as a float in [0, 1]. This represents
        the fraction of observations that fall below the predicted values.

    Example:
        Check calibration of median forecasts

        >>> import numpy as np
        >>> y_true = np.array([95, 105, 100, 110, 90])
        >>> y_pred = np.array([100, 100, 100, 100, 100])  # Median predictions
        >>> obs_prob = observed_probability(y_true, y_pred)
        >>> round(obs_prob, 1)  # Should be close to 0.5 for well-calibrated median
        0.4

    Note:
        This metric is fundamental for evaluating forecast calibration.
        Systematic deviations from expected quantile levels indicate
        overconfident or underconfident uncertainty estimates.
    """
    probability = np.mean(y_true < y_pred)
    return float(probability) if not np.isnan(probability) else 0.0


def mean_absolute_calibration_error(
    y_true: npt.NDArray[np.floating],
    y_pred: npt.NDArray[np.floating],
    quantiles: list[Quantile],
) -> float:
    """Calculate the Mean Absolute Calibration Error (MACE) for probabilistic forecasts.

    MACE measures how well the predicted quantiles match their nominal levels
    by comparing observed probabilities to expected quantile levels. Perfect
    calibration yields MACE = 0.

    Args:
        y_true: Observed values with shape (num_samples,).
        y_pred: Predicted quantiles with shape (num_samples, num_quantiles).
            Each column represents predictions for a specific quantile level.
        quantiles: Nominal quantile levels with shape (num_quantiles,).
            Must be sorted in ascending order and contain values in [0, 1].

    Returns:
        The mean absolute calibration error as a float in [0, 0.5].
        Values closer to 0 indicate better calibration.

    Example:
        Evaluate calibration of quantile forecasts

        >>> import numpy as np
        >>> y_true = np.array([95, 105, 100, 110, 90, 115, 85, 120])
        >>> quantiles = np.array([0.1, 0.5, 0.9])
        >>> # Well-calibrated forecasts
        >>> y_pred = np.array([[90, 95, 100],    # 10%, 50%, 90% quantiles
        ...                    [100, 105, 110],
        ...                    [95, 100, 105],
        ...                    [105, 110, 115],
        ...                    [85, 90, 95],
        ...                    [110, 115, 120],
        ...                    [80, 85, 90],
        ...                    [115, 120, 125]])
        >>> mace = mean_absolute_calibration_error(y_true, y_pred, quantiles)
        >>> round(mace, 2)
        0.23

    Note:
        MACE is a key diagnostic for probabilistic forecasts. High MACE values
        indicate that the forecast confidence intervals are either too wide
        (overconfident) or too narrow (underconfident).
    """
    observed_probs = np.array([observed_probability(y_true, y_pred[:, i]) for i in range(len(quantiles))])
    return float(np.mean(np.abs(observed_probs - quantiles)))


def mean_pinball_loss(
    y_true: npt.NDArray[np.floating],
    y_pred: npt.NDArray[np.floating],
    quantiles: list[Quantile],
    sample_weight: npt.NDArray[np.floating] | None = None,
) -> float:
    """Calculate the Mean Pinball Loss for quantile forecasts.

    The Pinball Loss is a proper scoring rule for evaluating quantile forecasts.
    It penalizes under- and over-predictions differently based on the quantile level.

    Args:
        y_true: Observed values with shape (num_samples,) or (num_samples, num_quantiles).
        y_pred: Predicted quantiles with shape (num_samples, num_quantiles).
            Each column corresponds to predictions for a specific quantile level.
        quantiles: Quantile levels with shape (num_quantiles,).
            Must be sorted in ascending order and contain values in [0, 1].
        sample_weight: Optional weights for each sample with shape (num_samples,).

    Returns:
        The weighted average Pinball Loss across all samples and quantiles. Lower values indicate better
        forecast quality.
    """
    # Reshape predictions and targets in case they arrive flattened (e.g. from an XGBoost eval callback,
    # which repeats the observed value across quantile columns). Collapse the targets back to one value per sample.
    y_pred = np.reshape(y_pred, [-1, len(quantiles)])
    n_rows = y_pred.shape[0]
    y_true = np.reshape(y_true, [n_rows, -1])[:, 0]

    # Average the (sample-weighted) pinball loss across all quantiles.
    return float(
        np.mean(
            [
                pinball_loss(y_true, y_pred[:, i], quantile=quantile, sample_weights=sample_weight)
                for i, quantile in enumerate(quantiles)
            ]
        )
    )
