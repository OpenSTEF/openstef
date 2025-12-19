# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Data models for evaluation subsets and metrics.

Provides structures for organizing forecast evaluation data into meaningful subsets
with their corresponding ground truth and predictions. Includes metric containers
that support both quantile-specific and global performance measurements.
"""

from collections import defaultdict
from datetime import datetime
from typing import Literal

import pandas as pd

from openstef_beam.evaluation.models.window import Window
from openstef_core.base_model import BaseModel, FloatOrNan
from openstef_core.types import Quantile, QuantileOrGlobal

MetricsDict = dict[str, FloatOrNan]
QuantileMetricsDict = dict[QuantileOrGlobal, MetricsDict]


class SubsetMetric(BaseModel):
    """Container for evaluation metrics computed on a data subset.

    Stores performance metrics organized by quantile and window, enabling
    detailed analysis of forecast quality across different probability levels
    and temporal periods.
    """

    window: Window | Literal["global"]
    timestamp: datetime
    metrics: QuantileMetricsDict

    def get_quantiles(self) -> list[Quantile]:
        """Return a list of quantiles present in the metrics.

        Returns:
            Sorted list of quantile values (excluding 'global').
        """
        return sorted([q for q in self.metrics if q != "global"])

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the metrics to a pandas DataFrame.

        Returns:
            DataFrame with quantiles as index and metric names as columns.
        """
        return pd.DataFrame(
            data=[{"quantile": quantile, **metric_dict} for quantile, metric_dict in self.metrics.items()]
        )

    def get_metric(self, quantile: QuantileOrGlobal, metric_name: str) -> FloatOrNan | None:
        """Retrieve a specific metric value for a given quantile.

        Args:
            quantile: The quantile level or 'global'.
            metric_name: The name of the metric to retrieve.

        Returns:
            The metric value if it exists, otherwise None.
        """
        return self.metrics.get(quantile, {}).get(metric_name)


def merge_quantile_metrics(metrics_list: list[QuantileMetricsDict]) -> QuantileMetricsDict:
    """Merge multiple quantile metrics dictionaries into a single one.

    Args:
        metrics_list: List of quantile metrics dictionaries to merge.

    Returns:
        Combined dictionary with all metrics from input dictionaries.
    """
    merged_metrics: QuantileMetricsDict = defaultdict(dict)
    for metrics in metrics_list:
        for quantile, metric_dict in metrics.items():
            merged_metrics[quantile].update(metric_dict)

    return merged_metrics
