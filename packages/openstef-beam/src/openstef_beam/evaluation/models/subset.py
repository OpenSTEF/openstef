# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Self, cast

import pandas as pd

from openstef_beam.evaluation.models.window import Window
from openstef_core.base_model import BaseModel, FloatOrNan
from openstef_core.types import Quantile
from openstef_core.utils import timedelta_from_isoformat, timedelta_to_isoformat


class EvaluationSubset:
    ground_truth: pd.DataFrame
    predictions: pd.DataFrame
    sample_interval: timedelta

    def __init__(
        self,
        ground_truth: pd.DataFrame,
        predictions: pd.DataFrame,
        sample_interval: timedelta,
    ):
        super().__init__()
        if not ground_truth.index.equals(predictions.index):  # type: ignore[reportUnknownMemberType]
            raise ValueError("Ground truth and predictions must have the same index.")

        self.ground_truth = ground_truth
        self.predictions = predictions
        self.sample_interval = sample_interval

        sample_interval_str = timedelta_to_isoformat(self.sample_interval)
        self.ground_truth.attrs["sample_interval"] = sample_interval_str
        self.predictions.attrs["sample_interval"] = sample_interval_str

    @property
    def index(self) -> pd.DatetimeIndex:
        return cast(pd.DatetimeIndex, self.ground_truth.index)

    @classmethod
    def create(
        cls,
        ground_truth: pd.DataFrame,
        predictions: pd.DataFrame,
        sample_interval: timedelta,
        index: pd.DatetimeIndex | None = None,
    ) -> Self:
        combined_index = cast(pd.DatetimeIndex, ground_truth.index.intersection(predictions.index))
        if index is not None:
            combined_index = combined_index.intersection(index)

        return cls(
            ground_truth=ground_truth.loc[combined_index],
            predictions=predictions.loc[combined_index],
            sample_interval=sample_interval,
        )

    def to_parquet(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.ground_truth.to_parquet(path / "ground_truth.parquet")
        self.predictions.to_parquet(path / "predictions.parquet")

    @classmethod
    def from_parquet(cls, path: Path) -> Self:
        ground_truth = pd.read_parquet(path / "ground_truth.parquet")
        predictions = pd.read_parquet(path / "predictions.parquet")
        sample_interval = timedelta_from_isoformat(ground_truth.attrs.get("sample_interval", "PT15M"))
        return cls(
            ground_truth=ground_truth,
            predictions=predictions,
            sample_interval=sample_interval,
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
