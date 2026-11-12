# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Helper functions shared between evaluation metric providers."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from openstef_beam.evaluation.models.subset import QuantileMetricsDict
from openstef_core.types import Quantile

type SymmetricQuantileMetric = Callable[..., float]


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


__all__ = ["compute_symmetric_quantile_metrics"]
