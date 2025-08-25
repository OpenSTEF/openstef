# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
from typing import Self

import pandas as pd
from pydantic import TypeAdapter

from openstef_beam.evaluation.models.subset import EvaluationSubset, SubsetMetric
from openstef_beam.evaluation.models.window import Filtering
from openstef_core.base_model import BaseModel
from openstef_core.datasets import TimeSeriesDataset


class EvaluationSubsetReport(BaseModel):
    filtering: Filtering
    subset: EvaluationSubset
    metrics: list[SubsetMetric]

    def to_parquet(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "metrics.json").write_bytes(TypeAdapter(list[SubsetMetric]).dump_json(self.metrics))
        self.subset.to_parquet(path)

    @classmethod
    def read_parquet(cls, path: Path) -> Self:
        metrics = TypeAdapter[list[SubsetMetric]](list[SubsetMetric]).validate_json(
            (path / "metrics.json").read_bytes()
        )
        filtering = TypeAdapter[Filtering](Filtering).validate_python(path.name)
        subset = EvaluationSubset.from_parquet(path)
        return cls(
            filtering=filtering,
            subset=subset,
            metrics=metrics,
        )

    def get_global_metric(self) -> SubsetMetric | None:
        """Returns the SubsetMetric with window='global', or None if not found."""
        return next((m for m in self.metrics if m.window == "global"), None)

    def get_windowed_metrics(self) -> list[SubsetMetric]:
        """Returns all SubsetMetrics with window != 'global'."""
        return [metric for metric in self.metrics if metric.window != "global"]

    def get_measurements(self) -> pd.Series:
        """Extract measurements Series from the report for the given target."""
        return self.subset.ground_truth.data["load"]

    def get_quantile_predictions(self) -> TimeSeriesDataset:
        """Extract forecasted quantiles from the report."""
        return self.subset.predictions


class EvaluationReport(BaseModel):
    subset_reports: list[EvaluationSubsetReport]

    def get_subset(self, filtering: Filtering) -> EvaluationSubsetReport | None:
        for subset_report in self.subset_reports:
            if str(subset_report.filtering) == str(filtering):
                return subset_report
        return None

    def to_parquet(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        for subset_report in self.subset_reports:
            subset_report.to_parquet(path / str(subset_report.filtering))

    @classmethod
    def read_parquet(cls, path: Path) -> Self:
        subset_reports: list[EvaluationSubsetReport] = []
        for subset_path in path.iterdir():
            if subset_path.is_dir():
                subset_report = EvaluationSubsetReport.read_parquet(subset_path)
                subset_reports.append(subset_report)

        return cls(subset_reports=subset_reports)
