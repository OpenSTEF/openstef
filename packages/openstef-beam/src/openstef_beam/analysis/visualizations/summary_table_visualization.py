# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Summary table visualization module for displaying metrics in tabular format."""

from typing import Any, NamedTuple, override

import pandas as pd

from openstef_beam.analysis.models import AnalysisAggregation, GroupName, RunName, TargetMetadata, VisualizationOutput
from openstef_beam.analysis.plots import (
    SummaryTablePlotter,
)
from openstef_beam.analysis.visualizations.base import ReportTuple, VisualizationProvider
from openstef_beam.evaluation import EvaluationSubsetReport
from openstef_core.types import QuantileOrGlobal


class MetricAggregation(NamedTuple):
    """Container for aggregated metric statistics."""

    mean: float
    min: float
    max: float
    median: float


class SummaryTableVisualization(VisualizationProvider):
    """Creates HTML tables summarizing evaluation metrics.

    Generates sortable HTML tables presenting evaluation metrics organized by quantiles,
    targets, and model runs. Tables automatically aggregate statistics and provide
    formatted overviews for reports and documentation.

    What you'll see:
    - Sortable columns showing metrics with mean, min, max, and median aggregations
    - Color-coded formatting for easy comparison
    - Automatic organization by quantiles (when applicable) and targets
    - Export-ready HTML format

    Aggregation behavior:
    - Single target: Simple metric table with quantile breakdown
    - Multiple targets: Comparative statistics across targets
    - Multiple runs: Model comparison with aggregated performance
    - Target groups: Hierarchical organization by categories

    Example:
        >>> from openstef_beam.analysis import AnalysisConfig
        >>> from openstef_beam.analysis.visualizations import SummaryTableVisualization
        >>> analysis_config = AnalysisConfig(
        ...     visualization_providers=[
        ...         SummaryTableVisualization(name="performance_summary"),
        ...     ]
        ... )
        >>> # Tables will show all available metrics organized by:
        >>> # - Quantile levels (0.1, 0.5, 0.9, global)
        >>> # - Performance metrics (MAE, RMSE, rCRPS, etc.)
        >>> # - Target groupings and model runs
    """

    @property
    @override
    def supported_aggregations(self) -> set[AnalysisAggregation]:
        return {
            AnalysisAggregation.NONE,
            AnalysisAggregation.RUN_AND_NONE,
            AnalysisAggregation.TARGET,
            AnalysisAggregation.GROUP,
            AnalysisAggregation.RUN_AND_GROUP,
            AnalysisAggregation.RUN_AND_TARGET,
        }

    @staticmethod
    def _aggregate_metric_values(values: list[float]) -> MetricAggregation:
        """Compute statistical aggregations for a collection of metric values.

        Args:
            values: List of numeric metric values to aggregate

        Returns:
            MetricAggregation containing mean, min, max, and median statistics
        """
        if not values:
            return MetricAggregation(mean=0.0, min=0.0, max=0.0, median=0.0)

        series = pd.Series(values)
        return MetricAggregation(
            mean=float(series.mean()),
            min=float(series.min()),
            max=float(series.max()),
            median=float(series.median()),
        )

    @staticmethod
    def _format_quantile(quantile: QuantileOrGlobal) -> str:
        """Format quantile values for display by removing trailing zeros.

        Args:
            quantile: Either a float quantile value or "global" string

        Returns:
            Formatted string representation suitable for display
        """
        if isinstance(quantile, float):
            return f"{quantile:f}".rstrip("0").rstrip(".")
        return str(quantile)

    def _extract_metrics_from_report(self, report: EvaluationSubsetReport) -> list[dict[str, Any]]:
        """Extract all metrics from a report into a list of dictionaries.

        Args:
            report: The evaluation subset report to extract metrics from

        Returns:
            List of dictionaries containing metric data, or empty list if no metrics found
        """
        rows: list[dict[str, Any]] = []
        global_metric = report.get_global_metric()

        if global_metric is not None and hasattr(global_metric, "metrics"):
            for quantile, metrics in global_metric.metrics.items():
                formatted_quantile = self._format_quantile(quantile)
                for metric_name, metric_value in metrics.items():
                    rows.append({
                        "Quantile|Global": formatted_quantile,
                        "Metric": metric_name,
                        "Value": metric_value,
                    })
        return rows

    @staticmethod
    def _create_sorted_dataframe(rows: list[dict[str, Any]], columns: list[str], sort_by: list[str]) -> pd.DataFrame:
        """Create and sort a DataFrame from metric rows.

        Args:
            rows: List of dictionaries containing metric data
            columns: Column names for the DataFrame
            sort_by: Column names to sort by

        Returns:
            Sorted pandas DataFrame
        """
        # Create DataFrame with or without columns specified
        dataframe = pd.DataFrame(columns=columns) if not rows else pd.DataFrame(rows)  # type: ignore[arg-type]
        return dataframe.sort_values(by=sort_by) if not dataframe.empty else dataframe

    @override
    def create_by_none(
        self,
        report: EvaluationSubsetReport,
        metadata: TargetMetadata,
    ) -> VisualizationOutput:
        rows = self._extract_metrics_from_report(report)
        dataframe = SummaryTableVisualization._create_sorted_dataframe(
            rows=rows, columns=["Quantile|Global", "Metric", "Value"], sort_by=["Metric", "Quantile|Global"]
        )

        plotter = SummaryTablePlotter(dataframe)
        return VisualizationOutput(name=self.name, html=plotter.plot())

    @override
    def create_by_target(
        self,
        reports: list[ReportTuple],
    ) -> VisualizationOutput:
        rows: list[dict[str, Any]] = []

        for metadata, report in reports:
            metric_rows = self._extract_metrics_from_report(report)
            for row in metric_rows:
                row["Target"] = metadata.name
                rows.append(row)

        dataframe = SummaryTableVisualization._create_sorted_dataframe(
            rows=rows,
            columns=["Target", "Quantile|Global", "Metric", "Value"],
            sort_by=["Metric", "Quantile|Global", "Target"],
        )

        plotter = SummaryTablePlotter(dataframe)
        return VisualizationOutput(name=self.name, html=plotter.plot())

    def create_by_group(self, reports: dict[GroupName, list[ReportTuple]]) -> VisualizationOutput:
        """Create summary table with aggregated metrics by group.

        Args:
            reports: Dictionary mapping group names to lists of (metadata, report) tuples

        Returns:
            VisualizationOutput containing the generated aggregated table
        """
        metric_dict: dict[tuple[str, str, GroupName], list[float]] = {}

        # Collect all metric values per group, quantile/global, and metric
        for group, target_reports in reports.items():
            for _metadata, report in target_reports:
                metric_rows = self._extract_metrics_from_report(report)
                for row in metric_rows:
                    key = (row["Quantile|Global"], row["Metric"], group)
                    metric_dict.setdefault(key, []).append(float(row["Value"]))

        # Aggregate values and build rows
        rows: list[dict[str, Any]] = []
        for (quantile, metric_name, group), values in metric_dict.items():
            agg = self._aggregate_metric_values(values)
            rows.append({
                "Group": group,
                "Quantile|Global": quantile,
                "Metric": metric_name,
                "Mean": agg.mean,
                "Min": agg.min,
                "Max": agg.max,
                "Median": agg.median,
            })

        dataframe = SummaryTableVisualization._create_sorted_dataframe(
            rows=rows,
            columns=["Group", "Quantile|Global", "Metric", "Mean", "Min", "Max", "Median"],
            sort_by=["Metric", "Quantile|Global", "Group"],
        )

        plotter = SummaryTablePlotter(dataframe)
        return VisualizationOutput(name=self.name, html=plotter.plot())

    @override
    def create_by_run_and_none(self, reports: dict[RunName, list[ReportTuple]]) -> VisualizationOutput:
        rows: list[dict[str, Any]] = []

        for run_name, target_reports in reports.items():
            for metadata, report in target_reports:
                metric_rows = self._extract_metrics_from_report(report)
                for row in metric_rows:
                    row["Target"] = metadata.name
                    row["Run"] = run_name
                    rows.append(row)

        dataframe = SummaryTableVisualization._create_sorted_dataframe(
            rows=rows,
            columns=["Target", "Quantile|Global", "Metric", "Run", "Value"],
            sort_by=["Metric", "Quantile|Global", "Target", "Run"],
        )

        plotter = SummaryTablePlotter(dataframe)
        return VisualizationOutput(name=self.name, html=plotter.plot())

    def create_by_run_and_target(
        self,
        reports: dict[RunName, list[ReportTuple]],
    ) -> VisualizationOutput:
        """Create summary table comparing different runs on the same targets.

        Args:
            reports: Dictionary mapping run names to their report lists.

        Returns:
            Visualization output with summary table comparing runs.
        """
        return self.create_by_run_and_none(
            reports=reports,
        )

    def create_by_run_and_group(
        self, reports: dict[tuple[RunName, GroupName], list[ReportTuple]]
    ) -> VisualizationOutput:
        """Create summary table with aggregated metrics by run and group combinations.

        Args:
            reports: Dictionary mapping (run_name, group_name) tuples to lists of (metadata, report) tuples

        Returns:
            VisualizationOutput containing the generated aggregated comparison table
        """
        metric_dict: dict[tuple[str, str, RunName, GroupName], list[float]] = {}

        # Collect all metric values per (run, group), quantile/global, and metric
        for (run_name, group), target_reports in reports.items():
            for _metadata, report in target_reports:
                metric_rows = self._extract_metrics_from_report(report)
                for row in metric_rows:
                    key = (row["Quantile|Global"], row["Metric"], run_name, group)
                    metric_dict.setdefault(key, []).append(float(row["Value"]))

        # Aggregate values and build rows
        rows: list[dict[str, Any]] = []
        for (quantile, metric_name, run_name, group), values in metric_dict.items():
            agg = self._aggregate_metric_values(values)
            rows.append({
                "Group": group,
                "Quantile|Global": quantile,
                "Metric": metric_name,
                "Run": run_name,
                "Mean": agg.mean,
                "Min": agg.min,
                "Max": agg.max,
                "Median": agg.median,
            })

        dataframe = self._create_sorted_dataframe(
            rows=rows,
            columns=["Group", "Quantile|Global", "Metric", "Run", "Mean", "Min", "Max", "Median"],
            sort_by=["Metric", "Quantile|Global", "Group", "Run"],
        )

        plotter = SummaryTablePlotter(dataframe)
        return VisualizationOutput(name=self.name, html=plotter.plot())


__all__ = ["SummaryTableVisualization"]
