# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Windowed metric visualization provider.

This module provides visualization for windowed metrics over time, showing
how performance metrics evolve across different time windows.
"""

import operator
from collections import defaultdict
from datetime import datetime
from typing import Literal, override

import numpy as np

from openstef_beam.analysis.models import AnalysisAggregation, GroupName, RunName, TargetMetadata, VisualizationOutput
from openstef_beam.analysis.plots import (
    WindowedMetricPlotter,
)
from openstef_beam.analysis.visualizations.base import MetricIdentifier, ReportTuple, VisualizationProvider
from openstef_beam.evaluation import EvaluationSubsetReport, Window
from openstef_core.types import Quantile


class WindowedMetricVisualization(VisualizationProvider):
    """Creates time series plots showing metric evolution across evaluation windows.

    Displays how evaluation metrics change over time by plotting metric values on a
    timeline where each point represents performance over a specific time window.
    The visualization reveals performance trends, seasonal patterns, and helps identify
    periods where model accuracy degrades or improves.

    What you'll see:
    - Time series line plot with metric values on Y-axis and time on X-axis
    - Each point shows metric computed over a sliding evaluation window
    - Multiple lines when comparing across targets or model runs
    - Clear trends showing model performance stability over time

    Useful for identifying:
    - Performance degradation patterns over time
    - Seasonal effects in forecasting accuracy
    - Model stability across different periods
    - Optimal retraining intervals based on performance drops

    Example:
        >>> from openstef_beam.analysis import AnalysisConfig
        >>> from openstef_beam.analysis.visualizations import WindowedMetricVisualization
        >>> from openstef_beam.evaluation import Window
        >>> from datetime import timedelta
        >>>
        >>> analysis_config = AnalysisConfig(
        ...     visualization_providers=[
        ...         WindowedMetricVisualization(
        ...             name="mae_evolution",
        ...             metric="MAE",
        ...             window=Window(lag=timedelta(hours=0), size=timedelta(days=7)),
        ...         ),
        ...     ]
        ... )
    """

    metric: MetricIdentifier
    window: Window

    @property
    @override
    def supported_aggregations(self) -> set[AnalysisAggregation]:
        return {
            AnalysisAggregation.NONE,
            AnalysisAggregation.RUN_AND_NONE,
            AnalysisAggregation.TARGET,
            AnalysisAggregation.RUN_AND_TARGET,
            AnalysisAggregation.GROUP,
            AnalysisAggregation.RUN_AND_GROUP,
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

    def _average_time_series_across_targets(
        self, reports: list[ReportTuple], metric_name: str, quantile_or_global: Quantile | Literal["global"]
    ) -> list[tuple[datetime, float]]:
        """Average windowed metric values across multiple targets at each timestamp.

        Args:
            reports: List of (metadata, report) tuples from different targets
            metric_name: Name of the metric to extract
            quantile_or_global: Either a Quantile object or "global"

        Returns:
            List of (timestamp, averaged_metric_value) tuples
        """
        # Collect all time-value pairs from all targets
        timestamp_values: dict[datetime, list[float]] = defaultdict(list)

        for _metadata, report in reports:
            time_value_pairs = self._extract_windowed_metric_values(report, metric_name, quantile_or_global)

            for timestamp, value in time_value_pairs:
                timestamp_values[timestamp].append(value)

        # Calculate average for each timestamp
        averaged_pairs: list[tuple[datetime, float]] = []
        for timestamp in sorted(timestamp_values.keys()):
            values = timestamp_values[timestamp]
            if values:  # Only include timestamps that have data
                avg_value = float(np.nanmean(values))
                averaged_pairs.append((timestamp, avg_value))

        return averaged_pairs

    @override
    def create_by_none(
        self,
        report: EvaluationSubsetReport,
        metadata: TargetMetadata,
    ) -> VisualizationOutput:
        metric_name, quantile_or_global = self._get_metric_info()
        time_value_pairs = self._extract_windowed_metric_values(report, metric_name, quantile_or_global)

        if not time_value_pairs:
            raise ValueError("No windowed metrics found for the specified window and metric.")

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

    @override
    def create_by_run_and_none(self, reports: dict[RunName, list[ReportTuple]]) -> VisualizationOutput:
        metric_name, quantile_or_global = self._get_metric_info()
        plotter = WindowedMetricPlotter()

        # Collect data for each run
        for run_name, report_pairs in reports.items():
            for _metadata, report in report_pairs:
                time_value_pairs = self._extract_windowed_metric_values(report, metric_name, quantile_or_global)

                # Skip if no data points found for this run
                if not time_value_pairs:
                    raise ValueError("No windowed metrics found for the specified window, metric and run.")

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

    @override
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
                raise ValueError("No windowed metrics found for the specified window, metric and target.")

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

    # averaging over all targets in a single group
    @override
    def create_by_run_and_target(
        self,
        reports: dict[RunName, list[ReportTuple]],
    ) -> VisualizationOutput:
        metric_name, quantile_or_global = self._get_metric_info()
        plotter = WindowedMetricPlotter()

        # Process each run and calculate averaged metrics across its targets
        for run_name, target_reports in reports.items():
            if not target_reports:
                raise ValueError("No windowed metrics found for the specified window, metric and run.")

            # Average windowed metrics across all targets for this run
            averaged_pairs = self._average_time_series_across_targets(
                reports=target_reports,
                metric_name=metric_name,
                quantile_or_global=quantile_or_global,
            )

            # Skip if no averaged data points found for this run
            if not averaged_pairs:
                raise ValueError("No windowed averaged metrics found for the specified window, metric and run.")

            # Unpack the averaged pairs
            timestamps = [pair[0] for pair in averaged_pairs]
            metric_values = [pair[1] for pair in averaged_pairs]

            # Add this run to the plotter with averaged values
            plotter.add_model(
                model_name=run_name,
                timestamps=timestamps,
                metric_values=metric_values,
            )

        title = self._create_plot_title(metric_name, quantile_or_global, "by run (averaged over targets in group)")
        figure = plotter.plot(title=title, metric_name=metric_name)

        return VisualizationOutput(name=self.name, figure=figure)

    # averaging over all targets (also when in different groups)
    @override
    def create_by_run_and_group(
        self,
        reports: dict[tuple[RunName, GroupName], list[ReportTuple]],
    ) -> VisualizationOutput:
        metric_name, quantile_or_global = self._get_metric_info()
        plotter = WindowedMetricPlotter()

        # Collect all targets for each run
        run_to_targets: dict[str, list[ReportTuple]] = {}
        for (run_name, _group_name), target_reports in reports.items():
            run_to_targets.setdefault(run_name, []).extend(target_reports)

        # Average metrics over all targets for each run
        for run_name, all_target_reports in run_to_targets.items():
            if not all_target_reports:
                raise ValueError("No windowed metrics found for the specified window, metric and run.")

            # Average windowed metrics across all targets for this run
            averaged_pairs = self._average_time_series_across_targets(
                reports=all_target_reports,
                metric_name=metric_name,
                quantile_or_global=quantile_or_global,
            )

            if not averaged_pairs:
                raise ValueError("No windowed averaged metrics found for the specified window, metric and run.")

            timestamps = [pair[0] for pair in averaged_pairs]
            metric_values = [pair[1] for pair in averaged_pairs]

            # Add this (run, group) to the plotter with averaged values
            plotter.add_model(
                model_name=run_name,
                timestamps=timestamps,
                metric_values=metric_values,
            )

        title = self._create_plot_title(metric_name, quantile_or_global, "by run (averaged over all targets)")
        figure = plotter.plot(title=title, metric_name=metric_name)

        return VisualizationOutput(name=self.name, figure=figure)

    # averaging over all targets (also when in different groups) for a single run
    @override
    def create_by_group(
        self,
        reports: dict[GroupName, list[ReportTuple]],
    ) -> VisualizationOutput:
        metric_name, quantile_or_global = self._get_metric_info()
        plotter = WindowedMetricPlotter()

        # Collect all targets from all groups
        all_target_reports: list[ReportTuple] = []
        for report_list in reports.values():
            all_target_reports.extend(report_list)

        # Average metrics over all targets
        averaged_pairs = self._average_time_series_across_targets(
            reports=all_target_reports,
            metric_name=metric_name,
            quantile_or_global=quantile_or_global,
        )

        if not averaged_pairs:
            raise ValueError(
                "No windowed averaged metrics found for the specified window, metric and run across all groups."
            )

        timestamps = [pair[0] for pair in averaged_pairs]
        metric_values = [pair[1] for pair in averaged_pairs]

        # Use the run name from the first target if available
        run_name = all_target_reports[0][0].run_name if all_target_reports else ""

        plotter.add_model(
            model_name=run_name,
            timestamps=timestamps,
            metric_values=metric_values,
        )

        title = self._create_plot_title(metric_name, quantile_or_global, "averaged over all targets")
        figure = plotter.plot(title=title, metric_name=metric_name)

        return VisualizationOutput(name=self.name, figure=figure)


__all__ = ["WindowedMetricVisualization"]
