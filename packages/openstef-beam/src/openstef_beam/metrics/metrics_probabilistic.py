# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Metrics for forecasts that predict probability distributions instead of single values.

Unlike deterministic forecasts that predict one value (e.g., "consumption will be 100 MW"),
probabilistic forecasts predict a range of possible outcomes with their likelihoods
(e.g., "80% chance consumption will be between 90-110 MW"). These metrics evaluate both
how accurate these probability estimates are and how well-calibrated they are.

Key concepts:
    - Calibration: Do 90% prediction intervals actually contain the true value 90% of the time?
    - Sharpness: How narrow are the prediction intervals (more precise is better)?
    - Proper scoring: Metrics that reward honest probability estimates over gaming the system.
"""

import numpy as np
import numpy.typing as npt


def crps(
    y_true: npt.NDArray[np.floating],
    y_pred: npt.NDArray[np.floating],
    quantiles: npt.NDArray[np.floating],
    sample_weights: npt.NDArray[np.floating] | None = None,
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

    Returns:
        The weighted average CRPS across all samples. Lower values indicate
        better forecast quality.

    Example:
        Evaluate quantile forecasts for energy consumption:

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
    """
    import scoringrules as sr  # noqa: PLC0415 - import is quite slow, so we delay it until this function is called

    return float(np.average(sr.crps_quantile(y_true, y_pred, quantiles), weights=sample_weights))


def rcrps(
    y_true: npt.NDArray[np.floating],
    y_pred: npt.NDArray[np.floating],
    quantiles: npt.NDArray[np.floating],
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    sample_weights: npt.NDArray[np.floating] | None = None,
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

    Returns:
        The relative CRPS as a float. Returns NaN if the range between
        quantiles is zero.

    Example:
        Compare forecast quality across different scales:

        >>> import numpy as np
        >>> # High consumption period
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
        different consumption levels, such as summer vs. winter energy demand.
    """
    y_range = np.quantile(y_true, q=upper_quantile) - np.quantile(y_true, q=lower_quantile)
    if y_range == 0:
        return float("NaN")

    return float(crps(y_true, y_pred, quantiles, sample_weights) / y_range)


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
        Check calibration of median forecasts:

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
    quantiles: npt.NDArray[np.floating],
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
        Evaluate calibration of quantile forecasts:

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
