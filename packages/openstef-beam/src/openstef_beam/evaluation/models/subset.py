# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Self

import pandas as pd

from openstef_beam.evaluation.models.window import Window
from openstef_core.base_model import BaseModel, FloatOrNan
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.validation import check_sample_intervals
from openstef_core.exceptions import TimeSeriesValidationError
from openstef_core.types import Quantile


class EvaluationSubset:
    ground_truth: TimeSeriesDataset
    predictions: TimeSeriesDataset

    def __init__(
        self,
        ground_truth: TimeSeriesDataset,
        predictions: TimeSeriesDataset,
    ):
        super().__init__()
        if not ground_truth.index.equals(predictions.index):  # type: ignore[reportUnknownMemberType]
            raise TimeSeriesValidationError("Ground truth and predictions must have the same index.")

        check_sample_intervals([ground_truth, predictions])

        self.ground_truth = ground_truth
        self.predictions = predictions

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.ground_truth.index

    @property
    def sample_interval(self) -> timedelta:
        return self.ground_truth.sample_interval

    @classmethod
    def create(
        cls,
        ground_truth: TimeSeriesDataset,
        predictions: TimeSeriesDataset,
        index: pd.DatetimeIndex | None = None,
    ) -> Self:
        combined_index = ground_truth.index.intersection(predictions.index)
        if index is not None:
            combined_index = combined_index.intersection(index)

        return cls(
            ground_truth=TimeSeriesDataset(
                data=ground_truth.data.loc[combined_index],
                sample_interval=ground_truth.sample_interval,
            ),
            predictions=TimeSeriesDataset(
                data=predictions.data.loc[combined_index],
                sample_interval=predictions.sample_interval,
            ),
        )

    def to_parquet(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.ground_truth.to_parquet(path / "ground_truth.parquet")
        self.predictions.to_parquet(path / "predictions.parquet")

    @classmethod
    def from_parquet(cls, path: Path) -> Self:
        ground_truth = TimeSeriesDataset.read_parquet(path / "ground_truth.parquet")
        predictions = TimeSeriesDataset.read_parquet(path / "predictions.parquet")
        return cls(
            ground_truth=ground_truth,
            predictions=predictions,
        )


QuantileOrGlobal = Quantile | Literal["global"]

MetricsDict = dict[str, FloatOrNan]
QuantileMetricsDict = dict[QuantileOrGlobal, MetricsDict]


class SubsetMetric(BaseModel):
    window: Window | Literal["global"]
    timestamp: datetime
    metrics: QuantileMetricsDict

    def get_quantiles(self) -> list[Quantile]:
        """Return a list of quantiles present in the metrics."""
        return sorted([q for q in self.metrics if q != "global"])


def merge_quantile_metrics(metrics_list: list[QuantileMetricsDict]) -> QuantileMetricsDict:
    """Merge multiple quantile metrics dictionaries into a single one."""
    merged_metrics: QuantileMetricsDict = defaultdict(dict)
    for metrics in metrics_list:
        for quantile, metric_dict in metrics.items():
            merged_metrics[quantile].update(metric_dict)

    return merged_metrics
