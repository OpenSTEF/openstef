# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Metrics for forecasts that predict single values instead of probability distributions.

Deterministic forecasts predict one specific value (e.g., "load will be 100 MW").
These metrics measure how close predicted values are to actual values, with special
attention to peak load events that are critical for energy system operations.

Key focus areas:
    - Scale-invariant errors: Compare accuracy across different load levels
    - Peak detection: Identify when load will exceed operational thresholds
    - Operational effectiveness: Ensure predictions support actionable decisions
"""

from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from sklearn.metrics import r2_score


def rmae(
    y_true: npt.NDArray[np.floating],
    y_pred: npt.NDArray[np.floating],
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    sample_weights: npt.NDArray[np.floating] | None = None,
) -> float:
    """Calculate the relative Mean Absolute Error (rMAE) using percentiles for range calculation.

    The rMAE normalizes the Mean Absolute Error by the range of true values,
    making it scale-invariant and suitable for comparing errors across different
    datasets or time periods.

    Args:
        y_true: Ground truth values with shape (num_samples,).
        y_pred: Predicted values with shape (num_samples,).
        lower_quantile: Lower quantile for range calculation. Must be in [0, 1].
        upper_quantile: Upper quantile for range calculation. Must be in [0, 1]
            and greater than lower_quantile.
        sample_weights: Optional weights for each sample with shape (num_samples,).
            If None, all samples are weighted equally.

    Returns:
        The relative Mean Absolute Error as a float. Returns NaN if the range
        between quantiles is zero.

    Example:
        Basic usage with energy load data:

        >>> import numpy as np
        >>> y_true = np.array([100, 120, 110, 130, 105])
        >>> y_pred = np.array([98, 122, 108, 135, 107])
        >>> error = rmae(y_true, y_pred)
        >>> round(error, 3)
        0.096

        With custom quantiles and weights:

        >>> weights = np.array([1, 2, 1, 2, 1])
        >>> error = rmae(y_true, y_pred, lower_quantile=0.1,
        ...               upper_quantile=0.9, sample_weights=weights)
        >>> isinstance(error, float)
        True
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

    MAPE measures the average magnitude of errors in percentage terms,
    making it scale-independent and easily interpretable. However, it
    can be undefined or inflated when true values are near zero.

    Args:
        y_true: Ground truth values with shape (num_samples,). Should not contain
            values close to zero to avoid division issues.
        y_pred: Predicted values with shape (num_samples,).

    Returns:
        The Mean Absolute Percentage Error as a float. May return inf or
        extremely large values if y_true contains values close to zero.

    Example:
        Basic usage with energy load data:

        >>> import numpy as np
        >>> y_true = np.array([100, 120, 110, 130, 105])
        >>> y_pred = np.array([98, 122, 108, 135, 107])
        >>> error = mape(y_true, y_pred)
        >>> round(error, 4)
        0.0225

        With perfect predictions:

        >>> perfect_pred = np.array([100, 120, 110, 130, 105])
        >>> mape(y_true, perfect_pred)
        0.0
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate MAPE
    mape_value = np.mean(np.abs((y_true - y_pred) / y_true))

    return float(mape_value)


class ConfusionMatrix(NamedTuple):
    """Confusion matrix components for peak detection in energy forecasting.

    This class represents the results of classifying energy load peaks
    versus non-peaks, with additional effectiveness metrics to account for
    the direction and magnitude of prediction errors.

    Attributes:
        true_positives: Boolean array indicating correctly predicted peaks.
        true_negatives: Boolean array indicating correctly predicted non-peaks.
        false_positives: Boolean array indicating incorrectly predicted peaks.
        false_negatives: Boolean array indicating missed peaks.
        effective_true_positives: Boolean array indicating true positives that
            are effective (peak correctly predicted with appropriate magnitude/direction).
        ineffective_true_positives: Boolean array indicating true positives that
            are ineffective (peak correctly predicted but with wrong magnitude/direction).

    Note:
        All arrays have shape (num_samples,) and correspond to the same time points
        in the original forecast evaluation.
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
    """Calculate confusion matrix for peak detection in energy load.

    A peak is defined as a value that exceeds the positive limit or falls below
    the negative limit. This function evaluates both the accuracy of peak detection
    and the effectiveness of predictions based on error direction.

    Args:
        y_true: Ground truth energy load values with shape (num_samples,).
        y_pred: Predicted energy load values with shape (num_samples,).
        limit_pos: Positive threshold defining high load peaks.
            Values >= limit_pos are considered positive peaks.
        limit_neg: Negative threshold defining low load peaks.
            Values <= limit_neg are considered negative peaks.

    Returns:
        ConfusionMatrix containing boolean arrays for all classification outcomes
        and effectiveness metrics.

    Example:
        Peak detection for energy load data:

        >>> import numpy as np
        >>> y_true = np.array([100, 150, 80, 200, 90])  # 150 and 200 are peaks
        >>> y_pred = np.array([105, 145, 85, 195, 95])
        >>> cm = confusion_matrix(y_true, y_pred, limit_pos=120, limit_neg=85)
        >>> int(cm.true_positives.sum())  # Successfully detected peaks
        3
        >>> int(cm.false_positives.sum())  # Incorrectly predicted peaks
        0

    Note:
        Effective true positives require that high peaks are predicted even higher
        (positive error) and low peaks are predicted even lower (negative error).
        This captures whether the forecast provides actionable information for
        energy system operators.
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
    """Container for precision and recall metrics.

    This class holds the fundamental classification metrics used to evaluate
    peak detection performance in energy forecasting applications.

    Attributes:
        precision: The fraction of predicted peaks that were actual peaks.
            Range [0, 1] where 1.0 is perfect precision.
        recall: The fraction of actual peaks that were correctly predicted.
            Range [0, 1] where 1.0 is perfect recall.

    Note:
        High precision means few false alarms, while high recall means
        few missed peaks. There is often a trade-off between these metrics.
    """

    precision: float
    recall: float


def precision_recall(
    cm: ConfusionMatrix,
    *,
    effective: bool = False,
) -> PrecisionRecall:
    """Calculate precision and recall metrics from a confusion matrix.

    These metrics evaluate the quality of peak detection by measuring
    how many predicted peaks were correct (precision) and how many
    actual peaks were detected (recall).

    Args:
        cm: Confusion matrix from the confusion_matrix function containing
            all classification outcomes.
        effective: If True, uses effective true positives which account for
            prediction direction and magnitude. If False, uses standard
            true positives for calculation.

    Returns:
        PrecisionRecall containing precision and recall values, each in range [0, 1].

    Example:
        Calculate standard precision and recall:

        >>> import numpy as np
        >>> y_true = np.array([100, 150, 80, 200, 90])
        >>> y_pred = np.array([105, 145, 85, 195, 95])
        >>> cm = confusion_matrix(y_true, y_pred, limit_pos=120, limit_neg=85)
        >>> pr = precision_recall(cm)
        >>> float(pr.precision)
        1.0
        >>> float(pr.recall)
        1.0

    Note:
        When effective=True, the metrics focus on predictions that provide
        actionable information (correct magnitude and direction) rather than
        just correct classification.
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

    The F-beta score is a weighted harmonic mean of precision and recall,
    allowing for different emphasis on recall versus precision based on
    the beta parameter.

    Args:
        precision_recall: Container with precision and recall values from
            the precision_recall function.
        beta: Weight parameter controlling the relative importance of recall.
            Values > 1 favor recall, values < 1 favor precision.
            Common values: 0.5 (favor precision), 1.0 (balanced F1), 2.0 (favor recall).

    Returns:
        The F-beta score as a float in range [0, 1]. Returns 0.0 if both
        precision and recall are zero.

    Example:
        Calculate F2 score (favoring recall):

        >>> pr = PrecisionRecall(precision=0.8, recall=0.9)
        >>> score = fbeta(pr, beta=2.0)
        >>> round(score, 3)
        0.878

        Calculate F1 score (balanced):

        >>> score = fbeta(pr, beta=1.0)
        >>> round(score, 3)
        0.847

    Note:
        F-beta scores are particularly useful for imbalanced datasets where
        the cost of false positives and false negatives differs significantly.
        In energy forecasting, beta > 1 is often preferred to minimize missed peaks.
    """
    precision = precision_recall.precision
    recall = precision_recall.recall

    if precision + recall == 0:
        return 0.0

    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)


def riqd(
    y_true: npt.NDArray[np.floating],
    y_pred_lower_q: npt.NDArray[np.floating],
    y_pred_upper_q: npt.NDArray[np.floating],
    *,
    measurement_range_lower_q: float = 0.05,
    measurement_range_upper_q: float = 0.95,
) -> float:
    """Calculate the relative Inter Quantile Distance (rIQD).

    rIQD measures the average distance between two quantiles, normalized by the measurement range.

    Args:
        y_true: Ground truth values with shape (num_samples,).
        y_pred_lower_q: Predicted values of lower quantile with shape (num_samples,).
        y_pred_upper_q: Predicted values of upper quantile with shape (num_samples,).
        measurement_range_lower_q: Lower quantile for range calculation. Must be in [0, 1].
        measurement_range_upper_q: Upper quantile for range calculation. Must be in [0, 1]
            and greater than measurement_range_lower_q.

    Returns:
        The relative Inter Quantile Distance (rIQD) as a float. Returns NaN if the measurement
            range is zero.

    Example:
        Basic usage with energy load data:

        >>> import numpy as np
        >>> y_true = np.array([100, 120, 110, 130, 105])
        >>> y_pred_lower_q = np.array([90, 100, 105, 95, 85])
        >>> y_pred_upper_q = np.array([110, 125, 140, 135, 90])
        >>> riqd = riqd(y_true, y_pred_lower_q, y_pred_upper_q)
        >>> round(riqd, 4)
        0.9259

    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred_lower_q = np.array(y_pred_lower_q)
    y_pred_upper_q = np.array(y_pred_upper_q)

    y_range = np.quantile(y_true, q=measurement_range_upper_q) - np.quantile(y_true, q=measurement_range_lower_q)

    # Calculate IQD
    iqd = np.mean(y_pred_upper_q - y_pred_lower_q)

    # Avoid division by zero if range is zero
    if y_range == 0:
        return float("NaN")

    # Calculate rIQD
    riqd = iqd / y_range

    return float(riqd)


def r2(
    y_true: npt.NDArray[np.floating],
    y_pred: npt.NDArray[np.floating],
    *,
    sample_weights: npt.NDArray[np.floating] | None = None,
) -> float:
    """Calculate the R² (coefficient of determination) score.

    R² represents the proportion of variance in the dependent variable that is
    predictable from the independent variable(s). It provides a measure of how
    well observed outcomes are replicated by the model, based on the proportion
    of total variation of outcomes explained by the model.

    Args:
        y_true: Ground truth values with shape (num_samples,).
        y_pred: Predicted values with shape (num_samples,).
        sample_weights: Optional weights for each sample with shape (num_samples,).
            If None, all samples are weighted equally.

    Returns:
        The R² score as a float. Best possible score is 1.0, and it can be negative
        (because the model can be arbitrarily worse). A constant model that always
        predicts the mean of y_true would get an R² score of 0.0.

    Example:
        Basic usage with energy load data:

        >>> import numpy as np
        >>> y_true = np.array([100, 120, 110, 130, 105])
        >>> y_pred = np.array([98, 122, 108, 135, 107])
        >>> score = r2(y_true, y_pred)
        >>> round(score, 3)
        0.929

        Perfect predictions give R² = 1.0:

        >>> perfect_pred = np.array([100, 120, 110, 130, 105])
        >>> r2(y_true, perfect_pred)
        1.0

        With sample weights:

        >>> weights = np.array([1, 2, 1, 2, 1])
        >>> score = r2(y_true, y_pred, sample_weights=weights)
        >>> isinstance(score, float)
        True
    """
    return float(r2_score(y_true, y_pred, sample_weight=sample_weights))


def relative_pinball_loss(
    y_true: npt.NDArray[np.floating],
    y_pred: npt.NDArray[np.floating],
    *,
    quantile: float,
    measurement_range_lower_q: float = 0.05,
    measurement_range_upper_q: float = 0.95,
    sample_weights: npt.NDArray[np.floating] | None = None,
) -> float:
    """Calculate the relative Pinball Loss (also known as relative Quantile Loss).

    The relative pinball loss normalizes the pinball loss by the range of true values,
    making it scale-invariant and suitable for comparing quantile prediction errors
    across different datasets or time periods. The pinball loss can be used to quantify
    the accuracy of a single quantile.

    Args:
        y_true: Ground truth values with shape (num_samples,).
        y_pred: Predicted quantile values with shape (num_samples,).
        quantile: The quantile level being predicted (e.g., 0.1, 0.5, 0.9).
            Must be in [0, 1].
        measurement_range_lower_q: Lower quantile for range calculation. Must be in [0, 1].
        measurement_range_upper_q: Upper quantile for range calculation. Must be in [0, 1]
            and greater than measurement_range_lower_q.
        sample_weights: Optional weights for each sample with shape (num_samples,).
            If None, all samples are weighted equally.

    Returns:
        The relative Pinball Loss as a float. Returns NaN if the measurement range
        is zero.

    Example:
        Basic usage for 10th percentile predictions:

        >>> import numpy as np
        >>> y_true = np.array([100, 120, 110, 130, 105])
        >>> y_pred = np.array([95, 115, 105, 125, 100])  # 10th percentile predictions
        >>> rpbl = relative_pinball_loss(y_true, y_pred, quantile=0.1, measurement_range_lower_q=0.0,
        ...                               measurement_range_upper_q=1.0)
        >>> round(rpbl, 4)
        0.0167
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate pinball loss for each sample
    errors = y_true - y_pred
    pinball_losses = np.where(
        errors >= 0,
        quantile * errors,  # Under-prediction
        (quantile - 1) * errors,  # Over-prediction
    )

    # Calculate mean pinball loss (weighted if weights provided)
    mean_pinball_loss = np.average(pinball_losses, weights=sample_weights)

    # Calculate measurement range for normalization
    y_range = np.quantile(y_true, q=measurement_range_upper_q) - np.quantile(y_true, q=measurement_range_lower_q)

    # Avoid division by zero if range is zero
    if y_range == 0:
        return float("NaN")

    # Calculate relative pinball loss
    relative_pinball_loss = mean_pinball_loss / y_range

    return float(relative_pinball_loss)
