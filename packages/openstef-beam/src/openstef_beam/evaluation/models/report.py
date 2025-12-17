# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Evaluation report models for organizing and persisting forecast evaluation results.

Provides structured containers for evaluation results that bundle forecast subsets,
filtering criteria, and computed metrics. Enables saving/loading complete evaluation
reports to/from disk for analysis and comparison across different model versions.
"""

from pathlib import Path
from typing import Self

import pandas as pd
from pydantic import TypeAdapter, field_validator

from openstef_beam.evaluation.models.subset import SubsetMetric
from openstef_beam.evaluation.models.window import Filtering
from openstef_core.base_model import BaseModel
from openstef_core.datasets import ForecastDataset
from openstef_core.datasets.validation import validate_required_columns
from openstef_core.utils import not_none


class EvaluationSubsetReport(BaseModel):
    """Container for evaluation results on a specific data subset.

    Bundles filtering criteria, evaluation subset data, and computed metrics
    for a particular slice of the evaluation dataset. Enables persistence
    and retrieval of evaluation results for analysis.
    """

    filtering: Filtering
    subset: ForecastDataset
    metrics: list[SubsetMetric]

    @field_validator("subset")
    @classmethod
    def _validate_subset_has_target(cls, subset: ForecastDataset) -> ForecastDataset:
        validate_required_columns(subset.data, [subset.target_column])
        return subset

    def to_parquet(self, path: Path):
        """Save the subset report to parquet files in the specified directory.

        Args:
            path: Directory where to save the report data.
        """
        # Sanitize path by replacing colons (invalid on Windows)
        path = Path(str(path).replace(":", "_"))
        path.mkdir(parents=True, exist_ok=True)
        (path / "metrics.json").write_bytes(TypeAdapter(list[SubsetMetric]).dump_json(self.metrics))
        self.subset.to_parquet(path / "subset.parquet")

    @classmethod
    def read_parquet(cls, path: Path) -> Self:
        """Load a subset report from parquet files in the specified directory.

        Args:
            path: Directory containing the saved report data.

        Returns:
            Loaded EvaluationSubsetReport instance.
        """
        metrics = TypeAdapter[list[SubsetMetric]](list[SubsetMetric]).validate_json(
            (path / "metrics.json").read_bytes()
        )
        filtering = TypeAdapter[Filtering](Filtering).validate_python(path.name)
        subset = ForecastDataset.read_parquet(path / "subset.parquet")
        return cls(filtering=filtering, subset=subset, metrics=metrics)

    def get_global_metric(self) -> SubsetMetric | None:
        """Returns the SubsetMetric with window='global', or None if not found."""
        return next((m for m in self.metrics if m.window == "global"), None)

    def get_windowed_metrics(self) -> list[SubsetMetric]:
        """Returns all SubsetMetrics with window != 'global'."""
        return [metric for metric in self.metrics if metric.window != "global"]

    def get_measurements(self) -> pd.Series:
        """Extract measurements Series from the report for the given target.

        Returns:
            Ground truth measurements as a pandas Series.
        """
        return not_none(self.subset.target_series)

    def get_quantile_predictions(self) -> pd.DataFrame:
        """Extract forecasted quantiles from the report.

        Returns:
            Dataset containing forecasted quantile predictions.
        """
        return self.subset.quantiles_data


class EvaluationReport(BaseModel):
    """Complete evaluation report containing results for all data subsets.

    Aggregates evaluation results across different filtering criteria,
    enabling analysis of model performance across various
    conditions (lead times, data availability, etc.).
    """

    subset_reports: list[EvaluationSubsetReport]

    def get_subset(self, filtering: Filtering) -> EvaluationSubsetReport | None:
        """Retrieve the subset report for the specified filtering criteria.

        Args:
            filtering: The filtering criteria to search for.

        Returns:
            The matching subset report, or None if not found.
        """
        for subset_report in self.subset_reports:
            if str(subset_report.filtering) == str(filtering):
                return subset_report
        return None

    def to_parquet(self, path: Path):
        """Save the complete evaluation report to parquet files.

        Args:
            path: Directory where to save all subset reports.
        """
        path.mkdir(parents=True, exist_ok=True)
        for subset_report in self.subset_reports:
            # Sanitize filtering name for Windows compatibility (replace colons)
            filtering_name = str(subset_report.filtering).replace(":", "_")
            subset_report.to_parquet(path / filtering_name)

    @classmethod
    def read_parquet(cls, path: Path) -> Self:
        """Load a complete evaluation report from parquet files.

        Args:
            path: Directory containing all subset report data.

        Returns:
            Loaded EvaluationReport instance with all subset reports.
        """
        subset_reports: list[EvaluationSubsetReport] = []
        for subset_path in path.iterdir():
            if subset_path.is_dir():
                subset_report = EvaluationSubsetReport.read_parquet(subset_path)
                subset_reports.append(subset_report)

        return cls(subset_reports=subset_reports)
