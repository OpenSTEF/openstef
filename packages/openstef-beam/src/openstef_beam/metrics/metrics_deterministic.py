# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from typing import NamedTuple

import numpy as np
import numpy.typing as npt


def rmae(
    y_true: npt.NDArray[np.floating],
    y_pred: npt.NDArray[np.floating],
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    sample_weights: npt.NDArray[np.floating] | None = None,
) -> float:
    """Calculate the relative Mean Absolute Error (rMAE) using percentiles for range calculation.

    Parameters:
        y_true: True values, 1D array of shape (num_samples,).
        y_pred: Predicted values, 1D array of shape (num_samples,).
        lower_quantile: Lower quantile for range calculation (default: 5th percentile).
        upper_quantile: Upper quantile for range calculation (default: 95th percentile).
        sample_weights: Optional sample weights, 1D array of shape (num_samples,).

    Returns:
        The relative Mean Absolute Error (rMAE).
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate MAE
    mae = np.average(np.abs(y_true - y_pred), weights=sample_weights)

    # Calculate range using quantile
    y_range = np.quantile(y_true, q=upper_quantile) - np.quantile(y_true, q=lower_quantile)

    # Avoid division by zero if range is zero
    if y_range == 0:
        return float("NaN")

    # Calculate rMAE
    rmae = mae / y_range

    return float(rmae)


def mape(
    y_true: npt.NDArray[np.floating],
    y_pred: npt.NDArray[np.floating],
) -> float:
    """Calculate the Mean Absolute Percentage Error (MAPE).

    Parameters:
        y_true: True values, 1D array of shape (num_samples,).
        y_pred: Predicted values, 1D array of shape (num_samples,).

    Returns:
        The Mean Absolute Percentage Error (MAPE).
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate MAPE
    mape_value = np.mean(np.abs((y_true - y_pred) / y_true))

    return float(mape_value)


class ConfusionMatrix(NamedTuple):
    """Named tuple representing a confusion matrix for peak detection evaluations.

    Attributes:
        true_positives: Boolean array indicating correctly predicted peaks,
                       shape (num_samples,).
        true_negatives: Boolean array indicating correctly predicted non-peaks,
                       shape (num_samples,).
        false_positives: Boolean array indicating incorrectly predicted peaks,
                        shape (num_samples,).
        false_negatives: Boolean array indicating incorrectly missed peaks,
                        shape (num_samples,).
        effective_true_positives: Boolean array indicating true positives that are effective
                                 (peak is correctly predicted and is also high enough),
                                 shape (num_samples,).
        ineffective_true_positives: Boolean array indicating true positives that are ineffective
                                   (peak is correctly predicted, but is not high enough),
                                   shape (num_samples,).
    """

    true_positives: npt.NDArray[np.bool_]
    true_negatives: npt.NDArray[np.bool_]
    false_positives: npt.NDArray[np.bool_]
    false_negatives: npt.NDArray[np.bool_]
    effective_true_positives: npt.NDArray[np.bool_]
    ineffective_true_positives: npt.NDArray[np.bool_]


def confusion_matrix(
    y_true: npt.NDArray[np.floating],
    y_pred: npt.NDArray[np.floating],
    *,
    limit_pos: float,
    limit_neg: float,
) -> ConfusionMatrix:
    """Calculate the confusion matrix for peak detection.

    A peak is defined as a value that is either above the positive limit
    or below the negative limit.

    Parameters:
        y_true: True values, 1D array of shape (num_samples,).
        y_pred: Predicted values, 1D array of shape (num_samples,).
        limit_pos: Positive threshold to define peaks.
        limit_neg: Negative threshold to define peaks.

    Returns:
        ConfusionMatrix: A NamedTuple containing the components of the confusion matrix.
                        All boolean arrays have shape (num_samples,).
    """
    true_peaks_y: npt.NDArray[np.bool_] = (y_true >= limit_pos) | (y_true <= limit_neg)
    pred_peaks_y: npt.NDArray[np.bool_] = (y_pred >= limit_pos) | (y_pred <= limit_neg)

    error = y_pred - y_true
    effective_true_positives = ((y_true >= limit_pos) & (y_pred >= limit_pos) & (error > 0)) | (
        (y_true <= limit_neg) & (y_pred <= limit_neg) & (error < 0)
    )
    ineffective_true_positives = true_peaks_y & pred_peaks_y & ~effective_true_positives

    return ConfusionMatrix(
        true_positives=true_peaks_y & pred_peaks_y,
        true_negatives=~true_peaks_y & ~pred_peaks_y,
        false_positives=~true_peaks_y & pred_peaks_y,
        false_negatives=true_peaks_y & ~pred_peaks_y,
        effective_true_positives=effective_true_positives,
        ineffective_true_positives=ineffective_true_positives,
    )


class PrecisionRecall(NamedTuple):
    precision: float
    recall: float


def precision_recall(
    cm: ConfusionMatrix,
    effective: bool = False,
) -> PrecisionRecall:
    """Calculate precision and recall metrics from a confusion matrix.

    Parameters:
        cm: Confusion matrix calculated with the confusion_matrix function.
        effective: If True, use effective true positives for precision and recall.

    Returns:
        PrecisionRecall: A NamedTuple containing precision and recall metrics.
    """
    relevant = np.sum(cm.true_positives) + np.sum(cm.false_negatives)
    retrieved = np.sum(cm.true_positives) + np.sum(cm.false_positives)

    if effective:
        precision = np.sum(cm.effective_true_positives) / retrieved if retrieved > 0 else 0
        recall = np.sum(cm.effective_true_positives) / relevant if relevant > 0 else 0
    else:
        precision = np.sum(cm.true_positives) / retrieved if retrieved > 0 else 0
        recall = np.sum(cm.true_positives) / relevant if relevant > 0 else 0

    return PrecisionRecall(
        precision=precision,
        recall=recall,
    )


def fbeta(
    precision_recall: PrecisionRecall,
    beta: float = 2.0,
) -> float:
    """Calculate the F-beta score from precision and recall metrics.
    The F-beta score is a weighted harmonic mean of precision and recall.

    Parameters:
        precision_recall: A NamedTuple containing precision and recall metrics.
        beta: The weight of recall in the F-beta score.

    Returns:
        float: The F-beta score.
    """
    precision = precision_recall.precision
    recall = precision_recall.recall

    if precision + recall == 0:
        return 0.0

    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
