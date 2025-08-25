# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import operator
from datetime import datetime
from typing import Literal

from openstef_beam.analysis.models import AnalysisAggregation, RunName, TargetMetadata, VisualizationOutput
from openstef_beam.analysis.plots import (
    WindowedMetricPlotter,
)
from openstef_beam.analysis.visualizations.base import MetricIdentifier, ReportTuple, VisualizationProvider
from openstef_beam.evaluation import EvaluationSubsetReport, Window
from openstef_core.types import Quantile


class WindowedMetricVisualization(VisualizationProvider):
    """Visualization for windowed metrics over time across different aggregation levels."""

    metric: MetricIdentifier
    window: Window

    @property
    def supported_aggregations(self) -> set[AnalysisAggregation]:
        return {
            AnalysisAggregation.NONE,
            AnalysisAggregation.RUN_AND_NONE,
            AnalysisAggregation.TARGET,
        }

    def _get_metric_info(self) -> tuple[str, Quantile | Literal["global"]]:
        """Extract metric name and quantile/global type from the metric config.

        Returns:
            A tuple containing:
            - metric_name: The name of the metric
            - quantile_or_global: Either a Quantile object or the literal "global"
        """
        if isinstance(self.metric, str):
            return self.metric, "global"

        metric_name, quantile = self.metric
        return metric_name, quantile

    def _extract_windowed_metric_values(
        self, report: EvaluationSubsetReport, metric_name: str, quantile_or_global: Quantile | Literal["global"]
    ) -> list[tuple[datetime, float]]:
        """Extract time-value pairs for the specified metric from windowed metrics.

        Args:
            report: The evaluation subset report
            metric_name: Name of the metric to extract
            quantile_or_global: Either a Quantile object or "global"

        Returns:
            List of (timestamp, metric_value) tuples where timestamp is a datetime object
        """
        windowed_metrics = report.get_windowed_metrics()
        if not windowed_metrics:
            return []

        time_value_pairs: list[tuple[datetime, float]] = []

        for window_metrics in windowed_metrics:
            # Only process metrics for the specified window
            if self.window and window_metrics.window == self.window:
                timestamp = window_metrics.timestamp
                metric_value = window_metrics.metrics.get(quantile_or_global, {}).get(metric_name)

                if metric_value is not None:
                    time_value_pairs.append((timestamp, metric_value))

        # Sort by timestamp for proper time series visualization
        time_value_pairs.sort(key=operator.itemgetter(0))
        return time_value_pairs

    def _create_plot_title(
        self, metric_name: str, quantile_or_global: Quantile | Literal["global"], suffix: str
    ) -> str:
        """Create a formatted title for the plot.

        Args:
            metric_name: Name of the metric
            quantile_or_global: Either a Quantile object or "global"
            suffix: Additional suffix for the title

        Returns:
            Formatted plot title
        """
        metric_display = f"{metric_name} (q={quantile_or_global})" if quantile_or_global != "global" else metric_name
        return f"Windowed {metric_display} {self.window} over Time {suffix}"

    def create_by_none(
        self,
        report: EvaluationSubsetReport,
        metadata: TargetMetadata,
    ) -> VisualizationOutput:
        metric_name, quantile_or_global = self._get_metric_info()
        time_value_pairs = self._extract_windowed_metric_values(report, metric_name, quantile_or_global)

        if not time_value_pairs:
            raise ValueError("No windowed metrics found in the report for the specified window and metric.")

        # Unpack the sorted pairs
        timestamps = [pair[0] for pair in time_value_pairs]
        metric_values = [pair[1] for pair in time_value_pairs]

        plotter = WindowedMetricPlotter()
        plotter.add_model(
            model_name=metadata.run_name,
            timestamps=timestamps,
            metric_values=metric_values,
        )

        title = self._create_plot_title(metric_name, quantile_or_global, f"for {metadata.name}")
        figure = plotter.plot(title=title)

        return VisualizationOutput(name=self.name, figure=figure)

    def create_by_run_and_none(self, reports: dict[RunName, list[ReportTuple]]) -> VisualizationOutput:
        metric_name, quantile_or_global = self._get_metric_info()
        plotter = WindowedMetricPlotter()

        # Collect data for each run
        for run_name, report_pairs in reports.items():
            for _metadata, report in report_pairs:
                time_value_pairs = self._extract_windowed_metric_values(report, metric_name, quantile_or_global)

                # Skip if no data points found for this run
                if not time_value_pairs:
                    continue

                # Unpack the sorted pairs
                timestamps = [pair[0] for pair in time_value_pairs]
                metric_values = [pair[1] for pair in time_value_pairs]

                plotter.add_model(
                    model_name=run_name,
                    timestamps=timestamps,
                    metric_values=metric_values,
                )

        title = self._create_plot_title(metric_name, quantile_or_global, "by Run")
        figure = plotter.plot(title=title)

        return VisualizationOutput(name=self.name, figure=figure)

    def create_by_target(
        self,
        reports: list[ReportTuple],
    ) -> VisualizationOutput:
        metric_name, quantile_or_global = self._get_metric_info()
        plotter = WindowedMetricPlotter()

        # Get the run name from the first target metadata for the title
        run_name = reports[0][0].run_name if reports else ""

        # Process each target's report
        for metadata, report in reports:
            time_value_pairs = self._extract_windowed_metric_values(report, metric_name, quantile_or_global)

            # Skip if no data points found for this target
            if not time_value_pairs:
                continue

            # Unpack the sorted pairs
            timestamps = [pair[0] for pair in time_value_pairs]
            metric_values = [pair[1] for pair in time_value_pairs]

            # Add this target to the plotter
            plotter.add_model(
                model_name=metadata.name,  # Use target name as the model name
                timestamps=timestamps,
                metric_values=metric_values,
            )

        title_suffix = "by Target"
        if run_name:
            title_suffix += f" for {run_name}"

        title = self._create_plot_title(metric_name, quantile_or_global, title_suffix)
        figure = plotter.plot(title=title)

        return VisualizationOutput(name=self.name, figure=figure)


__all__ = ["WindowedMetricVisualization"]
