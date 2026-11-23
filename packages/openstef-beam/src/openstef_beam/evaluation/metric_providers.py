# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Metric computation providers for forecast evaluation.

Implements various metric providers that compute performance measures
for probabilistic forecasts. Each provider handles specific metric types
and can be configured to work with specific quantiles or all available quantiles.
"""

from datetime import timedelta
from typing import Literal, cast, override

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field
from typing_extensions import deprecated

from openstef_beam.evaluation.evaluation_helper import (
    ReferenceForecastBuilder,
    compute_symmetric_quantile_metrics,
    rolling_quantile_reference_forecast,
)
from openstef_beam.evaluation.models.subset import MetricsDict, QuantileMetricsDict
from openstef_beam.evaluation.models.window import Filtering
from openstef_beam.metrics import (
    completeness,
    confusion_matrix,
    crpss,
    fbeta,
    mae,
    mape,
    mean_absolute_calibration_error,
    observed_probability,
    precision_recall,
    r2,
    rcrps,
    rcs,
    relative_pinball_loss,
    riqd,
    rmae,
)
from openstef_beam.metrics.metrics_helpers import SampleWeighter
from openstef_core.base_model import BaseConfig
from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.types import AvailableAt, Quantile
from openstef_core.utils.invariants import not_none

type MetricDirection = Literal["higher_is_better", "lower_is_better"]


class MetricProvider(BaseConfig):
    """Base class for forecast metric computation.

    Provides a standardized interface for computing performance metrics on probabilistic
    forecasts. Handles processing across multiple quantiles and allows filtering to
    specific quantiles of interest.

    Subclasses implement compute_deterministic() to provide specific metric calculations
    for individual quantiles. The base class handles the iteration and organization.

    Example:
        Creating a custom metric provider

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

    def __call__(
        self, subset: ForecastDataset, ground_truth: ForecastInputDataset | None = None
    ) -> QuantileMetricsDict:
        """Process an evaluation subset and return metrics.

        Extracts predictions and ground truth from the subset, then computes
        metrics for all relevant quantiles.

        Args:
            subset: Evaluation subset containing predictions and ground truth data.
            ground_truth: Full measurement history; ignored if unused by the provider.

        Returns:
            QuantileMetricsDict mapping quantile keys to computed metric values.
        """
        _ = ground_truth
        y_true: npt.NDArray[np.floating] = cast(
            pd.DataFrame, not_none(subset.target_series)
        ).to_numpy()  # type workaround
        y_pred: npt.NDArray[np.floating] = subset.quantiles_data.to_numpy()

        return self.compute_probabilistic(y_true, y_pred, subset.quantiles)

    def compute_probabilistic(
        self,
        y_true: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        quantiles: list[Quantile],
    ) -> QuantileMetricsDict:
        """Compute probabilistic metrics computed on multiple quantile data.

        Default behaviour is to call compute_deterministic for each quantile and returns the metrics prefixed by
        the quantile value.

        Args:
            y_true: True values, 1D array of shape (num_samples,).
            y_pred: Predicted values, 2D array of shape (num_samples, num_quantiles).
            quantiles: Quantiles used for prediction, sequence of length (num_quantiles,).

        Returns:
            QuantileMetricsDict mapping quantile-prefixed metric names to computed values.
        """
        metrics: QuantileMetricsDict = {}
        for i, quantile in enumerate(quantiles):
            if self.quantiles is not None and quantile not in self.quantiles:
                continue

            metrics[quantile] = self.compute_deterministic(y_true=y_true, y_pred=y_pred[:, i], quantile=quantile)

        return metrics

    @property
    def metric_names(self) -> frozenset[str]:
        """Declared metric names that this provider produces.

        Override in subclasses to enable eager metric-name validation
        (e.g. in the hyperparameter tuner).
        """
        return frozenset[str]()

    def compute_deterministic(
        self,
        y_true: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        quantile: Quantile,
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


class CompletenessProvider(MetricProvider):
    """Provides completeness metric.

    Measures the proportion of non-missing values in the predictions.
    """

    @property
    @override
    def metric_names(self) -> frozenset[str]:
        return frozenset({"completeness"})

    @override
    def compute_deterministic(
        self,
        y_true: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        quantile: Quantile,
    ) -> MetricsDict:
        return {"completeness": completeness(y_pred)}


class PeakMetricProvider(MetricProvider):
    """Provides metrics for peak detection performance.

    Computes precision, recall, and F-beta score for both standard and
    effective cases. Uses confusion matrix based on specified thresholds.
    """

    @property
    @override
    def metric_names(self) -> frozenset[str]:
        return frozenset(
            {
                "num_predicted_peaks",
                "num_true_peaks",
                "precision",
                "recall",
                "effective_precision",
                "effective_recall",
                f"F{self.beta}",
                f"effective_F{self.beta}",
            }
        )

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
        quantile: Quantile,
    ) -> MetricsDict:
        cm = confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            limit_pos=self.limit_pos,
            limit_neg=self.limit_neg,
        )

        metrics: MetricsDict = {}
        metrics["num_predicted_peaks"] = int(cm.true_positives.sum() + cm.false_positives.sum())
        metrics["num_true_peaks"] = int(cm.true_positives.sum() + cm.false_negatives.sum())
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

    @property
    @override
    def metric_names(self) -> frozenset[str]:
        return frozenset({self.metric_name})

    metric_name: str = Field(
        default="rCRPS",
        description="Name used for the rCRPS result.",
    )

    lower_quantile: Quantile = Field(
        default=Quantile(0.01),
        description="Lower quantile bound for rCRPS normalization.",
    )
    upper_quantile: Quantile = Field(
        default=Quantile(0.99),
        description="Upper quantile bound for rCRPS normalization.",
    )
    sample_weighter: SampleWeighter | None = Field(
        default=None,
        description="Optional sample weighting configuration.",
    )

    @override
    def compute_probabilistic(
        self,
        y_true: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        quantiles: list[Quantile],
    ) -> QuantileMetricsDict:
        """Compute rCRPS metric for probabilistic forecasts.

        Args:
            y_true: True values, 1D array of shape (num_samples,).
            y_pred: Predicted values, 2D array of shape (num_samples, num_quantiles).
            quantiles: Quantiles used for prediction, sequence of length (num_quantiles,).

        Returns:
            QuantileMetricsDict containing global rCRPS metric value.
        """
        sample_weights = None
        if self.sample_weighter is not None:
            weight_scale = np.quantile(np.abs(y_true), q=self.upper_quantile)
            sample_weights = self.sample_weighter.calculate_weights(y_true, weight_scale)

        return {
            "global": {
                self.metric_name: rcrps(
                    y_true=y_true,
                    y_pred=y_pred,
                    quantiles=quantiles,
                    lower_quantile=self.lower_quantile,
                    upper_quantile=self.upper_quantile,
                    sample_weights=sample_weights,
                )
            }
        }


@deprecated(
    "RCRPSSampleWeightedProvider is deprecated. Use RCRPSProvider with sample_weighter instead.",
    stacklevel=2,
)
class RCRPSSampleWeightedProvider(MetricProvider):
    """Deprecated compatibility facade for weighted rCRPS."""

    @property
    @override
    def metric_names(self) -> frozenset[str]:
        return frozenset({"rCRPS_sample_weighted"})

    lower_quantile: Quantile = Field(
        default=Quantile(0.01),
        description="Lower quantile bound for rCRPS normalization.",
    )
    upper_quantile: Quantile = Field(
        default=Quantile(0.99),
        description="Upper quantile bound for rCRPS normalization and sample weighting.",
    )
    a_min: float = Field(
        default=0.1,
        description="Minimum sample weight.",
    )
    a_max: float = Field(
        default=1.0,
        description="Maximum sample weight.",
    )
    exponent: float = Field(
        default=1.0,
        description="Exponent for sample weight calculation.",
    )

    @override
    def compute_probabilistic(
        self,
        y_true: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        quantiles: list[Quantile],
    ) -> QuantileMetricsDict:
        """Compute sample weighted rCRPS metric for probabilistic forecasts.

        Args:
            y_true: True values, 1D array of shape (num_samples,).
            y_pred: Predicted values, 2D array of shape (num_samples, num_quantiles).
            quantiles: Quantiles used for prediction, sequence of length (num_quantiles,).

        Returns:
            QuantileMetricsDict containing global sample weighted rCRPS metric value.
        """
        sample_weights = np.clip(
            np.abs(y_true / np.quantile(np.abs(y_true), q=self.upper_quantile)) ** self.exponent,
            a_min=self.a_min,
            a_max=self.a_max,
        )

        return {
            "global": {
                "rCRPS_sample_weighted": rcrps(
                    y_true=y_true,
                    y_pred=y_pred,
                    quantiles=quantiles,
                    lower_quantile=self.lower_quantile,
                    upper_quantile=self.upper_quantile,
                    sample_weights=sample_weights,
                )
            }
        }


class CRPSSProvider(MetricProvider):
    """CRPS skill score relative to a rolling-quantile reference forecast.

    The reference forecast for a timestamp consists of the quantiles of the measurements in the
    ``context_length`` before its cutoff, where the cutoff follows from ``reference_filtering``:
    ``AvailableAt`` gives the same cutoff as ``filter_by_available_at`` (e.g. D-1 06:00) and
    ``LeadTime`` gives a cutoff of ``timestamp - lead_time``. Measurements are considered
    available at their own timestamp.

    Example:
        >>> from openstef_core.types import LeadTime
        >>> provider = CRPSSProvider(reference_filtering=LeadTime.from_string("PT48H"))
    """

    @property
    @override
    def metric_names(self) -> frozenset[str]:
        return frozenset({self.metric_name})

    metric_name: str = Field(
        default="CRPSS",
        description="Name used for the CRPSS result.",
    )

    reference_filtering: Filtering = Field(
        default=AvailableAt.from_string("D-1T0600"),
        description="Cutoff of the measurement history used for the reference forecast.",
    )
    context_length: timedelta = Field(
        default=timedelta(days=14),
        description="Length of the measurement history used for each reference forecast.",
    )
    minimum_coverage: float = Field(
        default=0.9,
        description="Minimum fraction of expected context samples; references with less coverage are NaN.",
    )
    weight_scale_quantile: Quantile = Field(
        default=Quantile(0.99),
        description="Quantile of the absolute reference context used to scale sample weights. "
        "Only used when sample_weighter is set.",
    )
    sample_weighter: SampleWeighter | None = Field(
        default=None,
        description="Optional sample weighting configuration.",
    )
    reference_forecast_builder: ReferenceForecastBuilder = Field(
        default=rolling_quantile_reference_forecast,
        exclude=True,
        description="Callable used to construct the reference forecast.",
    )

    def reference_forecast(
        self,
        measurements: pd.Series,
        index: pd.DatetimeIndex,
        sample_interval: timedelta,
        quantiles: list[Quantile],
    ) -> ForecastDataset:
        """Build the configured reference forecast for the requested timestamps."""
        return self.reference_forecast_builder(
            measurements=measurements,
            index=index,
            sample_interval=sample_interval,
            quantiles=quantiles,
            reference_filtering=self.reference_filtering,
            context_length=self.context_length,
            minimum_coverage=self.minimum_coverage,
        )

    @override
    def __call__(
        self, subset: ForecastDataset, ground_truth: ForecastInputDataset | None = None
    ) -> QuantileMetricsDict:
        """Compute global CRPSS against the reference forecast built from the ground truth."""
        if ground_truth is None:
            return {"global": {self.metric_name: float("NaN")}}

        y_true: npt.NDArray[np.floating] = cast(pd.DataFrame, not_none(subset.target_series)).to_numpy()
        y_pred = subset.quantiles_data.to_numpy()
        y_reference = self.reference_forecast(
            measurements=ground_truth.target_series,
            index=subset.index,
            sample_interval=subset.sample_interval,
            quantiles=subset.quantiles,
        ).quantiles_data.to_numpy()

        sample_weights = None
        if self.sample_weighter is not None:
            weight_scale = self.reference_forecast(
                measurements=ground_truth.target_series.abs(),
                index=subset.index,
                sample_interval=subset.sample_interval,
                quantiles=[self.weight_scale_quantile],
            ).quantiles_data.to_numpy()[:, 0]
            sample_weights = self.sample_weighter.calculate_weights(y_true, weight_scale)

        # Rows lacking measurement history have neither a reference nor a weight scale.
        valid = ~np.isnan(y_reference).any(axis=1)
        if not valid.any():
            return {"global": {self.metric_name: float("NaN")}}

        return {
            "global": {
                self.metric_name: crpss(
                    y_true=y_true[valid],
                    y_pred=y_pred[valid],
                    y_reference=y_reference[valid],
                    quantiles=subset.quantiles,
                    sample_weights=None if sample_weights is None else sample_weights[valid],
                )
            }
        }


class MAEProvider(MetricProvider):
    """Provides Mean Absolute Error metrics.

    Computes the average absolute error between predictions and true values.
    """

    @property
    @override
    def metric_names(self) -> frozenset[str]:
        return frozenset({"MAE"})

    allow_nan: bool = Field(
        default=True,
        description="Whether to allow NaN values in input. If False, NaN value will result in NaN metric output.",
    )

    @override
    def compute_deterministic(
        self,
        y_true: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        quantile: Quantile,
    ) -> MetricsDict:
        return {"MAE": mae(y_true=y_true, y_pred=y_pred, allow_nan=self.allow_nan)}


class RMAEProvider(MetricProvider):
    """Provides Relative Mean Absolute Error metrics.

    Normalizes MAE using specified quantile bounds to make errors
    comparable across different scales.
    """

    @property
    @override
    def metric_names(self) -> frozenset[str]:
        return frozenset({"rMAE"})

    lower_quantile: Quantile = Field(
        default=Quantile(0.01),
        description="Lower quantile bound for rMAE normalization.",
    )
    upper_quantile: Quantile = Field(
        default=Quantile(0.99),
        description="Upper quantile bound for rMAE normalization.",
    )
    norm_value: float | None = Field(
        default=None,
        description="Optional pre-calculated normalization value. If provided, it will be used directly instead "
        "of calculating the range from quantiles.",
    )
    allow_nan: bool = Field(
        default=False,
        description="Whether to allow NaN values in input. If False, NaN value will result in NaN metric output.",
    )

    @override
    def compute_deterministic(
        self,
        y_true: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        quantile: Quantile,
    ) -> MetricsDict:
        return {
            "rMAE": rmae(
                y_true=y_true,
                y_pred=y_pred,
                lower_quantile=self.lower_quantile,
                upper_quantile=self.upper_quantile,
                norm_value=self.norm_value,
                allow_nan=self.allow_nan,
            )
        }


class RMAEPeakHoursProvider(MetricProvider):
    """Provides Relative Mean Absolute Error metrics for peak hours only (8:00-20:00).

    Normalizes MAE using specified quantile bounds but only considers
    data points between 8:00 and 20:00 hours.
    """

    @property
    @override
    def metric_names(self) -> frozenset[str]:
        return frozenset({"rMAE_peak_hours"})

    lower_quantile: Quantile = Field(
        default=Quantile(0.01),
        description="Lower quantile bound for rMAE normalization.",
    )
    upper_quantile: Quantile = Field(
        default=Quantile(0.99),
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
    def __call__(
        self, subset: ForecastDataset, ground_truth: ForecastInputDataset | None = None
    ) -> QuantileMetricsDict:
        """Process an evaluation subset and return metrics.

        Extracts predictions and ground truth from the subset, then computes
        metrics for all relevant quantiles.

        Args:
            subset: Evaluation subset containing predictions and ground truth data.
            ground_truth: Unused by this provider.

        Returns:
            QuantileMetricsDict mapping peak/off-peak periods to computed metric values.
        """
        _ = ground_truth
        y_true: npt.NDArray[np.floating] = cast(pd.DataFrame, not_none(subset.target_series)).to_numpy()
        y_pred: npt.NDArray[np.floating] = subset.quantiles_data.to_numpy()

        hours = subset.index.to_series().dt.hour.to_numpy()
        peak_hours_mask = (hours >= self.start_peak_hours) & (hours < self.end_peak_hours)

        y_true = y_true[peak_hours_mask]
        y_pred = y_pred[peak_hours_mask]

        return self.compute_probabilistic(y_true, y_pred, subset.quantiles)

    @override
    def compute_deterministic(
        self,
        y_true: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        quantile: Quantile,
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

    @property
    @override
    def metric_names(self) -> frozenset[str]:
        return frozenset({"MAPE"})

    @override
    def compute_deterministic(
        self,
        y_true: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        quantile: Quantile,
    ) -> MetricsDict:
        return {"MAPE": mape(y_true=y_true, y_pred=y_pred)}


class R2Provider(MetricProvider):
    """Provides R² (coefficient of determination) metrics.

    Computes the R² score which represents the proportion of variance
    in the dependent variable that is predictable from the predictions.
    Values range from -∞ to 1.0, where 1.0 is perfect prediction.
    """

    @property
    @override
    def metric_names(self) -> frozenset[str]:
        return frozenset({"R2"})

    @override
    def compute_deterministic(
        self,
        y_true: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        quantile: Quantile,
    ) -> MetricsDict:
        return {"R2": r2(y_true=y_true, y_pred=y_pred)}


class ObservedProbabilityProvider(MetricProvider):
    """Provides observed probability metrics.

    Measures how often the actual value falls below the predicted value,
    which should match the quantile level for well-calibrated forecasts.

    This metric is only useful as a global metric and not windowed.
    """

    @property
    @override
    def metric_names(self) -> frozenset[str]:
        return frozenset({"observed_probability"})

    @override
    def compute_deterministic(
        self,
        y_true: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        quantile: Quantile,
    ) -> MetricsDict:
        return {"observed_probability": observed_probability(y_true=y_true, y_pred=y_pred)}


class MeanAbsoluteCalibrationErrorProvider(MetricProvider):
    """Provides quantile calibration metrics.

    Computes the observed probability for each quantile, which should
    match the quantile level for well-calibrated forecasts. The metric
    quantifies this by computing the mean absolute error between
    observed probabilities and predicted quantiles across all samples.
    """

    @property
    @override
    def metric_names(self) -> frozenset[str]:
        return frozenset({"mean_absolute_calibration_error"})

    @override
    def compute_probabilistic(
        self,
        y_true: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        quantiles: list[Quantile],
    ) -> QuantileMetricsDict:
        """Compute mean absolute calibration error for probabilistic forecasts.

        Args:
            y_true: True values, 1D array of shape (num_samples,).
            y_pred: Predicted values, 2D array of shape (num_samples, num_quantiles).
            quantiles: Quantiles used for prediction, sequence of length (num_quantiles,).

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

    @property
    @override
    def metric_names(self) -> frozenset[str]:
        return frozenset({"rIQD"})

    measurement_range_lower_q: Quantile = Field(
        default=Quantile(0.05),
        description="Lower quantile bound for measurement range normalization.",
    )
    measurement_range_upper_q: Quantile = Field(
        default=Quantile(0.95),
        description="Upper quantile bound for measurement range normalization.",
    )

    @override
    def compute_probabilistic(
        self,
        y_true: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        quantiles: list[Quantile],
    ) -> QuantileMetricsDict:
        """Compute rIQD for each quantile by finding its symmetric counterpart.

        For each quantile q, finds the symmetric quantile (1-q) and computes
        rIQD between them. Only processes quantiles for which a symmetric
        counterpart is available.

        Args:
            y_true: True values, 1D array of shape (num_samples,).
            y_pred: Predicted values, 2D array of shape (num_samples, num_quantiles).
            quantiles: Quantiles used for prediction, sequence of length (num_quantiles,).

        Returns:
            QuantileMetricsDict containing rIQD metrics for each processable quantile.
        """
        return compute_symmetric_quantile_metrics(
            y_true=y_true,
            y_pred=y_pred,
            quantiles=quantiles,
            selected_quantiles=self.quantiles,
            metric_name="rIQD",
            metric=riqd,
            measurement_range_lower_q=self.measurement_range_lower_q,
            measurement_range_upper_q=self.measurement_range_upper_q,
        )


class RCSProvider(MetricProvider):
    """Provides Regression Coverage Score metrics.

    Measures the fraction of observed values inside symmetric prediction
    intervals, such as P10-P90. For each quantile, finds its symmetric
    counterpart and computes RCS between them.
    """

    @property
    @override
    def metric_names(self) -> frozenset[str]:
        return frozenset({"RCS"})

    @override
    def compute_probabilistic(
        self,
        y_true: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        quantiles: list[Quantile],
    ) -> QuantileMetricsDict:
        """Compute RCS for each quantile by finding its symmetric counterpart.

        For each quantile q, finds the symmetric quantile (1-q) and computes
        RCS between them. Only processes quantiles for which a symmetric
        counterpart is available.

        Args:
            y_true: True values, 1D array of shape (num_samples,).
            y_pred: Predicted values, 2D array of shape (num_samples, num_quantiles).
            quantiles: Quantiles used for prediction, sequence of length (num_quantiles,).

        Returns:
            QuantileMetricsDict containing RCS metrics for each processable quantile.
        """
        return compute_symmetric_quantile_metrics(
            y_true=y_true,
            y_pred=y_pred,
            quantiles=quantiles,
            selected_quantiles=self.quantiles,
            metric_name="RCS",
            metric=rcs,
        )


class RelativePinballLossProvider(MetricProvider):
    """Provides Relative Pinball Loss metrics for quantile predictions.

    Computes the relative pinball loss (also known as relative quantile loss)
    for each quantile, normalized by the measurement range to make it scale-invariant
    and suitable for comparing quantile prediction errors across different datasets.
    """

    @property
    @override
    def metric_names(self) -> frozenset[str]:
        return frozenset({"relative_pinball_loss"})

    measurement_range_lower_q: Quantile = Field(
        default=Quantile(0.01),
        description="Lower quantile bound for measurement range normalization.",
    )
    measurement_range_upper_q: Quantile = Field(
        default=Quantile(0.99),
        description="Upper quantile bound for measurement range normalization.",
    )

    @override
    def compute_deterministic(
        self,
        y_true: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        quantile: Quantile,
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
    "CRPSSProvider",
    "MAPEProvider",
    "MeanAbsoluteCalibrationErrorProvider",
    "MetricDirection",
    "MetricProvider",
    "ObservedProbabilityProvider",
    "PeakMetricProvider",
    "R2Provider",
    "RCRPSProvider",
    "RCRPSSampleWeightedProvider",
    "RCSProvider",
    "RIQDProvider",
    "RMAEPeakHoursProvider",
    "RMAEProvider",
    "RelativePinballLossProvider",
]
