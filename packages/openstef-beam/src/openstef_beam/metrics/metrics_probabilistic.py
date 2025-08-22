# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import numpy.typing as npt


def crps(
    y_true: npt.NDArray[np.floating],
    y_pred: npt.NDArray[np.floating],
    quantiles: npt.NDArray[np.floating],
    sample_weights: npt.NDArray[np.floating] | None = None,
) -> float:
    """Calculate the Continuous Ranked Probability Score (CRPS) for a given set of true and predicted values.

    Parameters:
        y_true: True values, 1D array of shape (num_samples,).
        y_pred: Predicted values, 2D array of shape (num_samples, num_quantiles).
        quantiles: Quantiles used for prediction, 1D array of shape (num_quantiles,).
        sample_weights: Optional sample weights, 1D array of shape (num_samples,).

    Returns:
        float: The Continuous Ranked Probability Score.
    """
    import scoringrules as sr

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

    This function normalizes the CRPS by the range of true values
    calculated using specified quantiles.

    Parameters:
        y_true: True values, 1D array of shape (num_samples,).
        y_pred: Predicted values, 2D array of shape (num_samples, num_quantiles).
        quantiles: Quantiles used for prediction, 1D array of shape (num_quantiles,).
        lower_quantile: Lower quantile for range calculation (default: 5th quantile).
        upper_quantile: Upper quantile for range calculation (default: 95th quantile).
        sample_weights: Optional sample weights, 1D array of shape (num_samples,).

    Returns:
       float: The relative Continuous Ranked Probability Score.
    """
    y_range = np.quantile(y_true, q=upper_quantile) - np.quantile(y_true, q=lower_quantile)
    if y_range == 0:
        return float("NaN")

    return float(crps(y_true, y_pred, quantiles, sample_weights) / y_range)


def observed_probability(
    y_true: npt.NDArray[np.floating],
    y_pred: npt.NDArray[np.floating],
) -> float:
    """Calculate the observed probability (or the quantile) of the predicted values.

    Parameters:
        y_true: True values, 1D array of shape (num_samples,).
        y_pred: Predicted values, 1D array of shape (num_samples,).

    Returns:
        float: The observed probability / quantile.
    """
    probability = np.mean(y_true < y_pred)
    return float(probability) if not np.isnan(probability) else 0.0


def mean_absolute_calibration_error(
    y_true: npt.NDArray[np.floating],
    y_pred: npt.NDArray[np.floating],
    quantiles: npt.NDArray[np.floating],
) -> float:
    """Calculate the Mean Absolute Calibration Error (MACE).

    This function computes the mean absolute difference between the observed probabilities
    and the predicted quantiles.

    Parameters:
        y_true: True values, 1D array of shape (num_samples,).
        y_pred: Predicted quantiles, 2D array of shape (num_samples, num_quantiles).
        quantiles: Quantiles used for prediction, 1D array of shape (num_quantiles,).

    Returns:
        float: The mean absolute calibration error.
    """
    observed_probs = np.array([observed_probability(y_true, y_pred[:, i]) for i in range(len(quantiles))])
    return float(np.mean(np.abs(observed_probs - quantiles)))
