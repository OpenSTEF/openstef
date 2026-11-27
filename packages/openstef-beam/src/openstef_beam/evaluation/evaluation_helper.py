# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Helper functions shared between evaluation metric providers."""

import math
from collections.abc import Callable, Sequence
from datetime import timedelta
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from openstef_beam.evaluation.models.subset import QuantileMetricsDict
from openstef_beam.evaluation.models.window import Filtering
from openstef_core.datasets import ForecastDataset
from openstef_core.types import AvailableAt, Quantile

type SymmetricQuantileMetric = Callable[..., float]
type ReferenceForecastBuilder = Callable[..., ForecastDataset]


def rolling_quantile_reference_forecast(
    measurements: pd.Series,
    index: pd.DatetimeIndex,
    sample_interval: timedelta,
    quantiles: list[Quantile],
    *,
    reference_filtering: Filtering,
    context_length: timedelta,
    minimum_coverage: float,
) -> ForecastDataset:
    """Build a rolling quantile forecast from sorted, NaN-free measurements."""
    cutoffs = (
        reference_filtering.apply_index(index)
        if isinstance(reference_filtering, AvailableAt)
        else index - reference_filtering.value
    )
    context = measurements.loc[cutoffs.min() - context_length : cutoffs.max()]
    # Rolling windows end at existing index points, so add the cutoffs as NaN points.
    context = context.reindex(context.index.union(cutoffs.unique()))
    rolling = context.rolling(
        window=context_length,
        min_periods=math.ceil(minimum_coverage * context_length / sample_interval),
    )
    reference = pd.DataFrame({quantile.format(): rolling.quantile(quantile) for quantile in quantiles})
    return ForecastDataset(data=reference.loc[cutoffs].set_axis(index), sample_interval=sample_interval)


def compute_symmetric_quantile_metrics(
    y_true: npt.NDArray[np.floating],
    y_pred: npt.NDArray[np.floating],
    quantiles: Sequence[Quantile],
    metric_name: str,
    metric: SymmetricQuantileMetric,
    *,
    selected_quantiles: Sequence[Quantile] | None = None,
    **metric_kwargs: Any,
) -> QuantileMetricsDict:
    """Compute metrics for quantiles that have a complementary quantile.

    For each selected quantile, the helper finds its complementary quantile
    (``1 - q``), orders the corresponding predictions as lower and upper bounds,
    and computes the supplied interval metric. Median quantiles and quantiles
    without a complementary counterpart are skipped.

    Args:
        y_true: True values with shape (num_samples,).
        y_pred: Predicted values with shape (num_samples, num_quantiles).
        quantiles: Quantiles used for prediction, in the same order as y_pred columns.
        metric_name: Name under which to store the computed metric.
        metric: Callable that computes a metric from true values and interval bounds.
        selected_quantiles: Optional subset of quantiles to compute metrics for.
        metric_kwargs: Additional keyword arguments passed to the metric callable.

    Returns:
        QuantileMetricsDict containing metric values for matching quantile pairs.
    """
    quantile_indices = {quantile: index for index, quantile in enumerate(quantiles)}
    metrics: QuantileMetricsDict = {}

    for quantile, quantile_index in quantile_indices.items():
        if selected_quantiles is not None and quantile not in selected_quantiles:
            continue

        complementary_quantile = quantile.complementary()
        if quantile == complementary_quantile:
            continue

        complementary_index = quantile_indices.get(complementary_quantile)
        if complementary_index is None:
            continue

        if quantile < complementary_quantile:
            lower_pred = y_pred[:, quantile_index]
            upper_pred = y_pred[:, complementary_index]
        else:
            lower_pred = y_pred[:, complementary_index]
            upper_pred = y_pred[:, quantile_index]

        metrics[quantile] = {
            metric_name: metric(
                y_true=y_true,
                y_pred_lower_q=lower_pred,
                y_pred_upper_q=upper_pred,
                **metric_kwargs,
            )
        }

    return metrics


__all__ = [
    "ReferenceForecastBuilder",
    "compute_symmetric_quantile_metrics",
    "rolling_quantile_reference_forecast",
]
