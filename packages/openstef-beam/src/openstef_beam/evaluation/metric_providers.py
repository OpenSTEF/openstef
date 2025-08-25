# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from typing import cast, override

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field

from openstef_beam.evaluation.models import EvaluationSubset
from openstef_beam.evaluation.models.subset import MetricsDict, QuantileMetricsDict
from openstef_beam.metrics import (
    confusion_matrix,
    fbeta,
    mape,
    mean_absolute_calibration_error,
    observed_probability,
    precision_recall,
    rcrps,
    rmae,
)
from openstef_core.base_model import BaseConfig
from openstef_core.types import Quantile


class MetricProvider(BaseConfig):
    """Base class for forecast metric computation.

    Handles processing of probabilistic forecasts across multiple quantiles.
    Subclasses should implement compute_deterministic to provide specific metrics.

    If quantiles are specified, metrics are computed only for those quantiles.
    """

    quantiles: list[Quantile] | None = Field(
        default=None,
        description="List of quantiles to compute metrics for. If None, all quantiles are processed.",
    )

    def __call__(self, subset: EvaluationSubset) -> QuantileMetricsDict:
        """Process an evaluation subset and return metrics.

        Extracts predictions and ground truth from the subset, then computes
        metrics for all relevant quantiles.
        """
        quantiles = np.array([Quantile.parse(quantile) for quantile in subset.predictions.feature_names])
        y_true: npt.NDArray[np.floating] = cast(pd.Series, subset.ground_truth.data.squeeze()).to_numpy()
        y_pred: npt.NDArray[np.floating] = subset.predictions.data.to_numpy()

        return self.compute_probabilistic(y_true, y_pred, quantiles)

    def compute_probabilistic(
        self,
        y_true: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        quantiles: npt.NDArray[np.floating],
    ) -> QuantileMetricsDict:
        """Compute probabilistic metrics computed on multiple quantile data.

        Default behaviour is to call compute_deterministic for each quantile and returns the metrics prefixed by
        the quantile value.

        Parameters:
            y_true: True values, 1D array of shape (num_samples,).
            y_pred: Predicted values, 2D array of shape (num_samples, num_quantiles).
            quantiles: Quantiles used for prediction, 1D array of shape (num_quantiles,).
        """
        metrics: QuantileMetricsDict = {}
        for i, quantile in enumerate(quantiles.tolist()):
            if self.quantiles is not None and quantile not in self.quantiles:
                continue

            metrics[quantile] = self.compute_deterministic(y_true=y_true, y_pred=y_pred[:, i], quantile=float(quantile))

        return metrics

    def compute_deterministic(
        self,
        y_true: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        quantile: float,
    ) -> MetricsDict:
        """Compute metrics for a single quantile prediction.

        Must be implemented by subclasses that provide deterministic metrics (per quantile).

        Parameters:
            y_true: True values, 1D array of shape (num_samples,).
            y_pred: Predicted values, 1D array of shape (num_samples,).
            quantile: Quantile value for the prediction.
        """
        msg = "Subclasses should implement this method."
        raise NotImplementedError(msg)


class PeakMetricProvider(MetricProvider):
    """Provides metrics for peak detection performance.

    Computes precision, recall, and F-beta score for both standard and
    effective cases. Uses confusion matrix based on specified thresholds.
    """

    limit_pos: float = Field(
        default=0.5,
        description="Positive peak detection threshold. Predictions above this value are considered a peak.",
    )
    limit_neg: float = Field(
        default=-0.5,
        description="Negative peak detection threshold. Predictions below this value are considered a peak.",
    )
    beta: float = Field(
        default=1.0,
        description="Beta parameter for F-beta score. Controls the balance between precision and recall.",
    )

    @override
    def compute_deterministic(
        self,
        y_true: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        quantile: float,
    ) -> MetricsDict:
        cm = confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            limit_pos=self.limit_pos,
            limit_neg=self.limit_neg,
        )

        metrics: MetricsDict = {}
        metrics["num_predicted_peaks"] = cm.true_positives.sum() + cm.false_positives.sum()
        metrics["num_true_peaks"] = cm.true_positives.sum() + cm.false_negatives.sum()
        peak_pr = precision_recall(cm, effective=False)
        effective_pr = precision_recall(cm, effective=True)
        metrics["precision"], metrics["recall"] = peak_pr
        metrics["effective_precision"], metrics["effective_recall"] = effective_pr
        metrics[f"F{self.beta}"] = fbeta(peak_pr, beta=self.beta)
        metrics[f"effective_F{self.beta}"] = fbeta(effective_pr, beta=self.beta)

        return metrics


class RCRPSProvider(MetricProvider):
    """Provides the Relative Continuous Ranked Probability Score.

    Computes rCRPS directly from the full probabilistic forecast without
    processing individual quantiles.
    """

    lower_quantile: float = Field(
        default=0.01,
        description="Lower quantile bound for rCRPS normalization.",
    )
    upper_quantile: float = Field(
        default=0.99,
        description="Upper quantile bound for rCRPS normalization.",
    )

    def compute_probabilistic(
        self,
        y_true: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        quantiles: npt.NDArray[np.floating],
    ) -> QuantileMetricsDict:
        """Parameters:
        y_true: True values, 1D array of shape (num_samples,).
        y_pred: Predicted values, 2D array of shape (num_samples, num_quantiles).
        quantiles: Quantiles used for prediction, 1D array of shape (num_quantiles,).
        """
        return {
            "global": {
                "rCRPS": rcrps(
                    y_true=y_true,
                    y_pred=y_pred,
                    quantiles=quantiles,
                    lower_quantile=self.lower_quantile,
                    upper_quantile=self.upper_quantile,
                )
            }
        }


class RMAEProvider(MetricProvider):
    """Provides Relative Mean Absolute Error metrics.

    Normalizes MAE using specified quantile bounds to make errors
    comparable across different scales.
    """

    lower_quantile: float = Field(
        default=0.01,
        description="Lower quantile bound for rMAE normalization.",
    )
    upper_quantile: float = Field(
        default=0.99,
        description="Upper quantile bound for rMAE normalization.",
    )

    @override
    def compute_deterministic(
        self,
        y_true: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        quantile: float,
    ) -> MetricsDict:
        return {
            "rMAE": rmae(
                y_true=y_true,
                y_pred=y_pred,
                lower_quantile=self.lower_quantile,
                upper_quantile=self.upper_quantile,
            )
        }


class RMAEPeakHoursProvider(MetricProvider):
    """Provides Relative Mean Absolute Error metrics for peak hours only (8:00-20:00).

    Normalizes MAE using specified quantile bounds but only considers
    data points between 8:00 and 20:00 hours.
    """

    lower_quantile: float = Field(
        default=0.01,
        description="Lower quantile bound for rMAE normalization.",
    )
    upper_quantile: float = Field(
        default=0.99,
        description="Upper quantile bound for rMAE normalization.",
    )
    start_peak_hours: int = Field(
        default=7,
        description="Start hour for peak hours (default: 7 AM).",
    )
    end_peak_hours: int = Field(
        default=20,
        description="End hour for peak hours (default: 8 PM).",
    )

    @override
    def __call__(self, subset: EvaluationSubset) -> QuantileMetricsDict:
        """Process an evaluation subset and return metrics.

        Extracts predictions and ground truth from the subset, then computes
        metrics for all relevant quantiles.
        """
        quantiles = np.array([Quantile.parse(quantile) for quantile in subset.predictions.feature_names])
        y_true: npt.NDArray[np.floating] = cast(pd.Series, subset.ground_truth.data.squeeze()).to_numpy()
        y_pred: npt.NDArray[np.floating] = subset.predictions.data.to_numpy()

        hours = subset.index.hour
        peak_hours_mask = (hours >= self.start_peak_hours) & (hours < self.end_peak_hours)

        y_true = y_true[peak_hours_mask]
        y_pred = y_pred[peak_hours_mask]

        return self.compute_probabilistic(y_true, y_pred, quantiles)

    @override
    def compute_deterministic(
        self,
        y_true: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        quantile: float,
    ) -> MetricsDict:
        return {
            "rMAE_peak_hours": rmae(
                y_true=y_true,
                y_pred=y_pred,
                lower_quantile=self.lower_quantile,
                upper_quantile=self.upper_quantile,
            )
        }


class MAPEProvider(MetricProvider):
    """Provides Mean Absolute Percentage Error metrics.

    Computes relative errors as percentages, suitable for comparing
    errors across different scales.
    """

    @override
    def compute_deterministic(
        self,
        y_true: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        quantile: float,
    ) -> MetricsDict:
        return {"MAPE": mape(y_true=y_true, y_pred=y_pred)}


class ObservedProbabilityProvider(MetricProvider):
    """Provides observed probability metrics.

    Measures how often the actual value falls below the predicted value,
    which should match the quantile level for well-calibrated forecasts.

    This metric is only useful as a global metric and not windowed.
    """

    @override
    def compute_deterministic(
        self,
        y_true: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        quantile: float,
    ) -> MetricsDict:
        return {"observed_probability": observed_probability(y_true=y_true, y_pred=y_pred)}


class MeanAbsoluteCalibrationErrorProvider(MetricProvider):
    """Provides quantile calibration metrics.

    Computes the observed probability for each quantile, which should
    match the quantile level for well-calibrated forecasts. The metric
    quantifies this by computing the mean absolute error between
    observed probabilities and predicted quantiles across all samples.
    """

    @override
    def compute_probabilistic(
        self,
        y_true: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        quantiles: npt.NDArray[np.floating],
    ) -> QuantileMetricsDict:
        """Parameters:
        y_true: True values, 1D array of shape (num_samples,).
        y_pred: Predicted values, 2D array of shape (num_samples, num_quantiles).
        quantiles: Quantiles used for prediction, 1D array of shape (num_quantiles,).
        """
        return {
            "global": {
                "mean_absolute_calibration_error": mean_absolute_calibration_error(
                    y_true=y_true,
                    y_pred=y_pred,
                    quantiles=quantiles,
                )
            }
        }
