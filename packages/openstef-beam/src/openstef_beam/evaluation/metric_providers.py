# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Metric computation providers for forecast evaluation.

Implements various metric providers that compute performance measures
for probabilistic forecasts. Each provider handles specific metric types
and can be configured to work with specific quantiles or all available quantiles.
"""

from typing import Literal, cast, override

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
    r2,
    rcrps,
    relative_pinball_loss,
    riqd,
    rmae,
)
from openstef_core.base_model import BaseConfig
from openstef_core.types import Quantile

type MetricDirection = Literal["higher_is_better", "lower_is_better"]


class MetricProvider(BaseConfig):
    """Base class for forecast metric computation.

    Provides a standardized interface for computing performance metrics on probabilistic
    forecasts. Handles processing across multiple quantiles and allows filtering to
    specific quantiles of interest.

    Subclasses implement compute_deterministic() to provide specific metric calculations
    for individual quantiles. The base class handles the iteration and organization.

    Example:
        Creating a custom metric provider:

        >>> from openstef_beam.evaluation.metric_providers import MetricProvider
        >>> from openstef_core.types import Quantile
        >>> import numpy as np
        >>>
        >>> class CustomMaeProvider(MetricProvider):
        ...     def compute_deterministic(self, y_true, y_pred, quantile):
        ...         return {"mae": float(np.mean(np.abs(y_true - y_pred)))}
        >>>
        >>> # Use with specific quantiles only
        >>> provider = CustomMaeProvider(quantiles=[Quantile(0.1), Quantile(0.9)])
        >>>
        >>> # Or process all available quantiles
        >>> provider_all = CustomMaeProvider()

    Implementation guide:
        Subclasses should override compute_deterministic() to return a dictionary
        mapping metric names to computed values for a single quantile.

        For global metrics that don't depend on individual quantiles, override
        compute_probabilistic() instead to process all quantiles together.
    """

    quantiles: list[Quantile] | None = Field(
        default=None,
        description="List of quantiles to compute metrics for. If None, all quantiles are processed.",
    )

    def __call__(self, subset: EvaluationSubset) -> QuantileMetricsDict:
        """Process an evaluation subset and return metrics.

        Extracts predictions and ground truth from the subset, then computes
        metrics for all relevant quantiles.

        Args:
            subset: Evaluation subset containing predictions and ground truth data.

        Returns:
            QuantileMetricsDict mapping quantile keys to computed metric values.
        """
        quantiles = np.array(subset.predictions.quantiles)
        y_true: npt.NDArray[np.floating] = subset.ground_truth.target_series.to_numpy()  # type: ignore
        y_pred: npt.NDArray[np.floating] = subset.predictions.quantiles_data.to_numpy()

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

        Args:
            y_true: True values, 1D array of shape (num_samples,).
            y_pred: Predicted values, 2D array of shape (num_samples, num_quantiles).
            quantiles: Quantiles used for prediction, 1D array of shape (num_quantiles,).

        Returns:
            QuantileMetricsDict mapping quantile-prefixed metric names to computed values.
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
        """Compute rCRPS metric for probabilistic forecasts.

        Args:
            y_true: True values, 1D array of shape (num_samples,).
            y_pred: Predicted values, 2D array of shape (num_samples, num_quantiles).
            quantiles: Quantiles used for prediction, 1D array of shape (num_quantiles,).

        Returns:
            QuantileMetricsDict containing global rCRPS metric value.
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

        Args:
            subset: Evaluation subset containing predictions and ground truth data.

        Returns:
            QuantileMetricsDict mapping peak/off-peak periods to computed metric values.
        """
        quantiles = np.array(subset.predictions.quantiles)
        y_true: npt.NDArray[np.floating] = cast(pd.Series, subset.ground_truth.data.squeeze()).to_numpy()  # type: ignore
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


class R2Provider(MetricProvider):
    """Provides R² (coefficient of determination) metrics.

    Computes the R² score which represents the proportion of variance
    in the dependent variable that is predictable from the predictions.
    Values range from -∞ to 1.0, where 1.0 is perfect prediction.
    """

    @override
    def compute_deterministic(
        self,
        y_true: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        quantile: float,
    ) -> MetricsDict:
        return {"R2": r2(y_true=y_true, y_pred=y_pred)}


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
        """Compute mean absolute calibration error for probabilistic forecasts.

        Args:
            y_true: True values, 1D array of shape (num_samples,).
            y_pred: Predicted values, 2D array of shape (num_samples, num_quantiles).
            quantiles: Quantiles used for prediction, 1D array of shape (num_quantiles,).

        Returns:
            QuantileMetricsDict containing global mean absolute calibration error metric.
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


class RIQDProvider(MetricProvider):
    """Provides Relative Inter Quantile Distance metrics.

    Measures the average distance between symmetric quantiles (e.g., 0.1 and 0.9),
    normalized by the measurement range. For each quantile, finds
    its symmetric counterpart and computes rIQD between them.
    """

    median_quantile: float = 0.5

    measurement_range_lower_q: float = Field(
        default=0.05,
        description="Lower quantile bound for measurement range normalization.",
    )
    measurement_range_upper_q: float = Field(
        default=0.95,
        description="Upper quantile bound for measurement range normalization.",
    )

    @override
    def compute_probabilistic(
        self,
        y_true: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        quantiles: npt.NDArray[np.floating],
    ) -> QuantileMetricsDict:
        """Compute rIQD for each quantile by finding its symmetric counterpart.

        For each quantile q, finds the symmetric quantile (1-q) and computes
        rIQD between them. Only processes quantiles for which a symmetric
        counterpart is available.

        Args:
            y_true: True values, 1D array of shape (num_samples,).
            y_pred: Predicted values, 2D array of shape (num_samples, num_quantiles).
            quantiles: Quantiles used for prediction, 1D array of shape (num_quantiles,).

        Returns:
            QuantileMetricsDict containing rIQD metrics for each processable quantile.
        """
        metrics: QuantileMetricsDict = {}

        for i, quantile in enumerate(quantiles.tolist()):
            if self.quantiles is not None and quantile not in self.quantiles:
                continue

            symmetric_quantile = 1.0 - quantile

            if np.isclose(quantile, symmetric_quantile, atol=1e-6):
                continue  # skip if same quantile (e.g., 0.5)

            symmetric_indices = np.where(np.isclose(quantiles, symmetric_quantile, atol=1e-6))[0]

            if len(symmetric_indices) == 0:
                continue  # no symmetric quantile found, skip

            symmetric_idx = symmetric_indices[0]

            if quantile < self.median_quantile:
                lower_pred = y_pred[:, i]
                upper_pred = y_pred[:, symmetric_idx]
            else:
                lower_pred = y_pred[:, symmetric_idx]
                upper_pred = y_pred[:, i]

            metrics[quantile] = {
                "rIQD": riqd(
                    y_true=y_true,
                    y_pred_lower_q=lower_pred,
                    y_pred_upper_q=upper_pred,
                    measurement_range_lower_q=self.measurement_range_lower_q,
                    measurement_range_upper_q=self.measurement_range_upper_q,
                )
            }

        return metrics


class RelativePinballLossProvider(MetricProvider):
    """Provides Relative Pinball Loss metrics for quantile predictions.

    Computes the relative pinball loss (also known as relative quantile loss)
    for each quantile, normalized by the measurement range to make it scale-invariant
    and suitable for comparing quantile prediction errors across different datasets.
    """

    measurement_range_lower_q: float = Field(
        default=0.01,
        description="Lower quantile bound for measurement range normalization.",
    )
    measurement_range_upper_q: float = Field(
        default=0.99,
        description="Upper quantile bound for measurement range normalization.",
    )

    @override
    def compute_deterministic(
        self,
        y_true: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        quantile: float,
    ) -> MetricsDict:
        return {
            "relative_pinball_loss": relative_pinball_loss(
                y_true=y_true,
                y_pred=y_pred,
                quantile=quantile,
                measurement_range_lower_q=self.measurement_range_lower_q,
                measurement_range_upper_q=self.measurement_range_upper_q,
            )
        }


__all__ = [
    "MAPEProvider",
    "MeanAbsoluteCalibrationErrorProvider",
    "MetricDirection",
    "MetricProvider",
    "ObservedProbabilityProvider",
    "PeakMetricProvider",
    "R2Provider",
    "RCRPSProvider",
    "RIQDProvider",
    "RMAEPeakHoursProvider",
    "RMAEProvider",
    "RelativePinballLossProvider",
]
