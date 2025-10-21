# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Data models for evaluation subsets and metrics.

Provides structures for organizing forecast evaluation data into meaningful subsets
with their corresponding ground truth and predictions. Includes metric containers
that support both quantile-specific and global performance measurements.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Self

import pandas as pd

from openstef_beam.evaluation.models.window import Window
from openstef_core.base_model import BaseModel, FloatOrNan
from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.datasets.validation import validate_same_sample_intervals
from openstef_core.exceptions import TimeSeriesValidationError
from openstef_core.types import Quantile, QuantileOrGlobal


class EvaluationSubset:
    """A paired dataset of ground truth and predictions for evaluation.

    Ensures temporal alignment between ground truth measurements and forecast
    predictions to enable accurate performance assessment. Validates index
    consistency and sample intervals for reliable metric computation.
    """

    ground_truth: ForecastInputDataset
    predictions: ForecastDataset

    def __init__(
        self,
        ground_truth: ForecastInputDataset,
        predictions: ForecastDataset,
    ):
        """Initialize evaluation subset with aligned ground truth and predictions.

        Args:
            ground_truth: Historical measurements dataset.
            predictions: Forecast predictions dataset.

        Raises:
            TimeSeriesValidationError: If indices don't match or sample intervals differ.
        """
        super().__init__()
        if not ground_truth.index.equals(predictions.index):  # type: ignore[reportUnknownMemberType]
            raise TimeSeriesValidationError("Ground truth and predictions must have the same index.")

        validate_same_sample_intervals([ground_truth, predictions])

        self.ground_truth = ground_truth
        self.predictions = predictions

    @property
    def index(self) -> pd.DatetimeIndex:
        """Get the common temporal index for both datasets.

        Returns:
            DatetimeIndex shared by ground truth and predictions.
        """
        return self.ground_truth.index

    @property
    def sample_interval(self) -> timedelta:
        """Get the sampling interval of the datasets.

        Returns:
            Temporal resolution of the time series data.
        """
        return self.ground_truth.sample_interval

    @classmethod
    def create(
        cls,
        ground_truth: ForecastInputDataset,
        predictions: ForecastDataset,
        index: pd.DatetimeIndex | None = None,
    ) -> Self:
        """Create an evaluation subset with optional index filtering.

        Args:
            ground_truth: Historical measurements dataset.
            predictions: Forecast predictions dataset.
            index: Optional index to filter both datasets.

        Returns:
            New EvaluationSubset with aligned and optionally filtered data.
        """
        combined_index = ground_truth.index.intersection(predictions.index)
        if index is not None:
            combined_index = combined_index.intersection(index)

        return cls(
            ground_truth=ForecastInputDataset(
                data=ground_truth.data.loc[combined_index],
                sample_interval=ground_truth.sample_interval,
                target_column=ground_truth.target_column,
            ),
            predictions=ForecastDataset(
                data=predictions.data.loc[combined_index],
                sample_interval=predictions.sample_interval,
                forecast_start=None,
            ),
        )

    def to_parquet(self, path: Path):
        """Save the evaluation subset to parquet files.

        Args:
            path: Directory where to save ground truth and predictions data.
        """
        path.mkdir(parents=True, exist_ok=True)
        self.ground_truth.to_parquet(path / "ground_truth.parquet")
        self.predictions.to_parquet(path / "predictions.parquet")

    @classmethod
    def from_parquet(cls, path: Path) -> Self:
        """Load an evaluation subset from parquet files.

        Args:
            path: Directory containing saved ground truth and predictions data.

        Returns:
            Loaded EvaluationSubset instance.
        """
        ground_truth = ForecastInputDataset.read_parquet(path / "ground_truth.parquet")
        predictions = ForecastDataset.read_parquet(path / "predictions.parquet")
        return cls(
            ground_truth=ground_truth,
            predictions=predictions,
        )


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
