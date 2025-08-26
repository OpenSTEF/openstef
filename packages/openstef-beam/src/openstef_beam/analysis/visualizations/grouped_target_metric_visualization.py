# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Grouped target metric visualization provider.

This module provides visualization for comparing performance metrics across
different targets, grouped by various criteria like target groups, model runs,
or combinations thereof.
"""

from openstef_beam.analysis.models import AnalysisAggregation, GroupName, RunName, VisualizationOutput
from openstef_beam.analysis.plots import GroupedTargetMetricPlotter
from openstef_beam.analysis.visualizations.base import ReportTuple, VisualizationProvider
from openstef_beam.evaluation import EvaluationSubsetReport
from openstef_beam.evaluation.models.subset import QuantileOrGlobal
from openstef_core.types import Quantile


class GroupedTargetMetricVisualization(VisualizationProvider):
    """Creates bar charts and box plots for comparing metrics across targets and groups.

    Generates interactive charts comparing performance metrics across different targets,
    model runs, or target groups. Supports deterministic metrics (MAE, RMSE) and
    quantile-based metrics (quantile losses) for comprehensive performance comparisons.

    Key features:
    - Bar charts for individual target comparisons
    - Box plots for grouped target analysis
    - Support for selector-based metrics (e.g., show metric at best-performing quantile)
    - Color-coded grouping for easy identification
    - Interactive tooltips with detailed metric values

    Use cases:
    - Identify which targets are hardest to predict
    - Compare model performance across target categories
    - Analyze performance variations within target groups
    - Evaluate model consistency across different scenarios

    Example:
        Comparing RMAE across targets for different models:

        >>> from openstef_beam.analysis import AnalysisConfig
        >>> from openstef_beam.analysis.visualizations import GroupedTargetMetricVisualization
        >>> from openstef_core.types import Quantile
        >>>
        >>> analysis_config = AnalysisConfig(
        ...     visualization_providers=[
        ...         # Compare median forecast accuracy
        ...         GroupedTargetMetricVisualization(
        ...             name="rmae_comparison",
        ...             metric="rMAE",
        ...             quantile=Quantile(0.5),
        ...         ),
        ...         # Compare overall probabilistic performance
        ...         GroupedTargetMetricVisualization(
        ...             name="rcrps_comparison",
        ...             metric="rCRPS",  # Global metric, no quantile needed
        ...         ),
        ...         # Show metric at best-performing quantile
        ...         GroupedTargetMetricVisualization(
        ...             name="adaptive_accuracy",
        ...             metric="rMAE",
        ...             selector_metric="rCRPS",  # Find quantile with best rCRPS
        ...         ),
        ...     ]
        ... )
    """

    metric: str
    quantile: Quantile | None = None
    selector_metric: str | None = None

    @property
    def supported_aggregations(self) -> set[AnalysisAggregation]:
        """Return the set of aggregation types supported by this provider.

        Returns:
            Set of supported AnalysisAggregation values.
        """
        return {
            AnalysisAggregation.TARGET,
            AnalysisAggregation.GROUP,
            AnalysisAggregation.RUN_AND_NONE,
            AnalysisAggregation.RUN_AND_GROUP,
            AnalysisAggregation.RUN_AND_TARGET,
        }

    def _is_selector_metric(self) -> bool:
        """Check if this is a selector-based metric.

        Returns:
            True if this uses a selector metric, False otherwise.
        """
        return self.selector_metric is not None

    def _get_metric_name(self) -> str:
        """Get the metric name to display.

        Returns:
            The name of the metric to display.
        """
        return self.metric

    def _find_best_quantile_for_selector(
        self,
        report: EvaluationSubsetReport,
    ) -> QuantileOrGlobal | None:
        """Find the quantile with the best (highest) value for the selector metric.

        Returns:
            The quantile with the highest selector metric value, or None if not found.
        """
        if not self._is_selector_metric() or self.selector_metric is None:
            return None

        global_metric = report.get_global_metric()
        if global_metric is None:
            return None

        best_quantile = None
        best_value = -float("inf")

        # Check all quantiles for the selector metric
        for quantile in global_metric.get_quantiles():
            quantile_metrics = global_metric.metrics.get(quantile, {})
            selector_value = quantile_metrics.get(self.selector_metric)
            if selector_value is not None and selector_value > best_value:
                best_value = selector_value
                best_quantile = quantile

        return best_quantile

    def _extract_metric_value(
        self,
        report: EvaluationSubsetReport,
    ) -> float | None:
        """Extract metric value from evaluation report.

        Returns:
            The metric value for the specified quantile/selector, or None if not found.
        """
        global_metric = report.get_global_metric()
        if global_metric is None:
            return None

        if self._is_selector_metric():
            # For selector metrics, find the best quantile for the selector metric
            # then get the display metric value for that quantile
            best_quantile = self._find_best_quantile_for_selector(report)
            if best_quantile is None:
                return None

            quantile_metrics = global_metric.metrics.get(best_quantile, {})
            return quantile_metrics.get(self.metric)
        if self.quantile is not None:
            # Regular quantile-based metric
            return global_metric.metrics.get(self.quantile, {}).get(self.metric)
        # Global metric
        return global_metric.metrics.get("global", {}).get(self.metric)

    def _collect_target_metrics(self, reports: list[ReportTuple]) -> tuple[list[str], list[float]]:
        """Collect target names and their corresponding metric values.

        Returns:
            Tuple of (target_names, metric_values) lists.
        """
        targets: list[str] = []
        metric_values: list[float] = []

        for metadata, report in reports:
            metric_value = self._extract_metric_value(report)
            if metric_value is not None:
                targets.append(metadata.name)
                metric_values.append(metric_value)

        return targets, metric_values

    def _create_plot_title(self, suffix: str) -> str:
        """Create a formatted title for the plot.

        Returns:
            Formatted plot title with metric and suffix information.
        """
        if self._is_selector_metric() and self.selector_metric is not None:
            base_title = f"{self.metric} (best {self.selector_metric} quantile)"
        elif self.quantile is not None:
            base_title = f"{self.metric} (q={self.quantile})"
        else:
            base_title = self.metric
        return f"{base_title} {suffix}"

    @staticmethod
    def _create_unique_target_identifiers(
        targets: list[str],
        group_name: GroupName,
    ) -> list[str]:
        """Create unique target identifiers that include group information.

        Returns:
            List of unique target identifiers prefixed with group name.
        """
        return [f"({group_name}) {target_name}" for target_name in targets]

    def _process_reports_and_add_to_plotter(
        self,
        plotter: GroupedTargetMetricPlotter,
        reports: list[ReportTuple],
        model_name: str,
        group_name: GroupName | None = None,
        target_to_group_map: dict[str, GroupName] | None = None,
    ) -> None:
        """Process reports, collect metrics, and add model to plotter with optional grouping."""
        targets, metric_values = self._collect_target_metrics(reports)

        if not targets or not metric_values:
            return

        # Use unique identifiers if group_name is provided
        if group_name is not None:
            unique_targets = self._create_unique_target_identifiers(targets, group_name)
            if target_to_group_map is not None:
                for unique_target in unique_targets:
                    target_to_group_map[unique_target] = group_name
            targets = unique_targets

        plotter.add_model(
            model_name=model_name,
            targets=targets,
            metric_values=metric_values,
        )

    def create_by_target(self, reports: list[ReportTuple]) -> VisualizationOutput:
        """Create visualization comparing metric values across different targets.

        Args:
            reports: List of (metadata, report) tuples for each target.

        Returns:
            Visualization output showing target comparison.

        Raises:
            ValueError: If no valid metric data is found.
        """
        targets, _metric_values = self._collect_target_metrics(reports)

        if not targets:
            msg = f"No valid metric data found for '{self.metric}'"
            raise ValueError(msg)

        plotter = GroupedTargetMetricPlotter()
        self._process_reports_and_add_to_plotter(
            plotter=plotter,
            reports=reports,
            model_name="Targets",
        )

        title = self._create_plot_title("per Target")
        figure = plotter.plot(title=title, metric_name=self.metric)

        return VisualizationOutput(name=self.name, figure=figure)

    def create_by_run_and_none(
        self,
        reports: dict[RunName, list[ReportTuple]],
    ) -> VisualizationOutput:
        """Create visualization comparing different model runs.

        Args:
            reports: Dictionary mapping run names to their report lists.

        Returns:
            Visualization output showing run comparison.

        Raises:
            ValueError: If no valid metric data is found.
        """
        plotter = GroupedTargetMetricPlotter()

        for run_name, run_reports in reports.items():
            self._process_reports_and_add_to_plotter(
                plotter=plotter,
                reports=run_reports,
                model_name=run_name,
            )

        # Check if any valid data was processed
        has_valid_data = any(bool(self._collect_target_metrics(run_reports)[0]) for run_reports in reports.values())
        if not has_valid_data:
            msg = f"No valid metric data found for '{self.metric}'"
            raise ValueError(msg)

        title = self._create_plot_title("by Run")
        figure = plotter.plot(title=title, metric_name=self.metric)

        return VisualizationOutput(name=self.name, figure=figure)

    def create_by_run_and_target(
        self,
        reports: dict[RunName, list[ReportTuple]],
    ) -> VisualizationOutput:
        """Create visualization comparing different runs on the same targets.

        Args:
            reports: Dictionary mapping run names to their report lists.

        Returns:
            Visualization output showing run comparison on targets.
        """
        return self.create_by_run_and_none(
            reports=reports,
        )

    def create_by_run_and_group(
        self,
        reports: dict[tuple[RunName, GroupName], list[ReportTuple]],
    ) -> VisualizationOutput:
        """Create visualization comparing runs across target groups.

        Args:
            reports: Dictionary mapping (run_name, group_name) to report lists.

        Returns:
            Visualization output showing run and group comparison.

        Raises:
            ValueError: If no valid metric data is found.
        """
        plotter = GroupedTargetMetricPlotter()
        target_to_group_map: dict[str, GroupName] = {}

        for (run_name, group_name), run_group_reports in reports.items():
            self._process_reports_and_add_to_plotter(
                plotter=plotter,
                reports=run_group_reports,
                model_name=run_name,
                group_name=group_name,
                target_to_group_map=target_to_group_map,
            )

        # Check if any valid data was processed
        has_valid_data = any(
            bool(self._collect_target_metrics(run_group_reports)[0]) for run_group_reports in reports.values()
        )
        if not has_valid_data:
            msg = f"No valid metric data found for '{self.metric}'"
            raise ValueError(msg)

        if target_to_group_map:
            plotter.set_target_groups(target_to_group_map)

        title = self._create_plot_title("by Run and Target Group")
        figure = plotter.plot(title=title, metric_name=self.metric)

        return VisualizationOutput(name=self.name, figure=figure)

    def create_by_group(
        self,
        reports: dict[GroupName, list[ReportTuple]],
    ) -> VisualizationOutput:
        """Create visualization comparing different target groups.

        Args:
            reports: Dictionary mapping group names to their report lists.

        Returns:
            Visualization output showing group comparison.

        Raises:
            ValueError: If no valid metric data is found.
        """
        plotter = GroupedTargetMetricPlotter()
        target_to_group_map: dict[str, GroupName] = {}

        for group_name, group_reports in reports.items():
            self._process_reports_and_add_to_plotter(
                plotter=plotter,
                reports=group_reports,
                model_name=group_name,
                group_name=group_name,
                target_to_group_map=target_to_group_map,
            )

        # Check if any valid data was processed
        has_valid_data = any(bool(self._collect_target_metrics(group_reports)[0]) for group_reports in reports.values())
        if not has_valid_data:
            msg = f"No valid metric data found for '{self.metric}'"
            raise ValueError(msg)

        if target_to_group_map:
            plotter.set_target_groups(target_to_group_map)

        title = self._create_plot_title("by Target Group")
        figure = plotter.plot(title=title, metric_name=self.metric)

        return VisualizationOutput(name=self.name, figure=figure)


__all__ = ["GroupedTargetMetricVisualization"]
