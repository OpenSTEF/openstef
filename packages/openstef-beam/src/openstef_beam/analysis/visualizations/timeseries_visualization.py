# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Time series visualization provider.

This module provides visualization for time series forecast comparisons,
displaying measurements alongside forecast quantiles with capacity limits.
"""

from openstef_beam.analysis.models import AnalysisAggregation, RunName, TargetMetadata, VisualizationOutput
from openstef_beam.analysis.plots import ForecastTimeSeriesPlotter
from openstef_beam.analysis.visualizations.base import ReportTuple, VisualizationProvider
from openstef_beam.evaluation import EvaluationSubsetReport


class TimeSeriesVisualization(VisualizationProvider):
    """Creates interactive time series plots comparing forecasts with actual measurements.

    Displays forecast quantiles as uncertainty bands overlaid with actual measurements
    on a timeline. Shows how well probabilistic forecasts capture reality over time
    and helps identify periods of poor performance or systematic biases.

    What you'll see:
    - Actual measurements as a line plot
    - Forecast quantiles as shaded uncertainty bands (darker = higher confidence)
    - Capacity limits as horizontal reference lines
    - Multiple model runs as different colored bands (when comparing models)

    Useful for:
    - Assessing forecast accuracy across different time periods
    - Identifying when uncertainty bands fail to contain actual values
    - Spotting systematic forecast biases or seasonal patterns
    - Understanding model behavior during extreme events

    Example:
        >>> from openstef_beam.analysis import AnalysisConfig
        >>> from openstef_beam.analysis.visualizations import TimeSeriesVisualization
        >>>
        >>> analysis_config = AnalysisConfig(
        ...     visualization_providers=[
        ...         TimeSeriesVisualization(name="forecast_vs_actual"),
        ...     ]
        ... )
    """

    @property
    def supported_aggregations(self) -> set[AnalysisAggregation]:
        """Return the set of supported aggregation levels for this visualization."""
        return {AnalysisAggregation.NONE, AnalysisAggregation.RUN_AND_NONE}

    @staticmethod
    def _add_capacity_limits(plotter: ForecastTimeSeriesPlotter, limit: float) -> None:
        """Add upper and lower capacity limits to the plot.

        Args:
            plotter: The forecast time series plotter instance
            limit: The capacity limit value (positive)
        """
        plotter.add_limit(value=limit, name="Upper Limit")
        plotter.add_limit(value=-limit, name="Lower Limit")

    @staticmethod
    def _get_first_target_data(reports: dict[RunName, list[ReportTuple]]) -> ReportTuple:
        """Extract metadata and report from the first target in the reports.

        Args:
            reports: Dictionary mapping run names to target report pairs

        Returns:
            Tuple of (metadata, report) from the first available target

        Raises:
            ValueError: If no reports are provided
        """
        if not reports:
            raise ValueError("No reports provided for time series visualization.")

        first_run_reports = next(iter(reports.values()))
        if not first_run_reports:
            raise ValueError("No target reports found in the first run.")

        return first_run_reports[0]

    def create_by_none(
        self,
        report: EvaluationSubsetReport,
        metadata: TargetMetadata,
    ) -> VisualizationOutput:
        """Create time series visualization for a single target from a single run.

        Args:
            report: Evaluation report containing time series data.
            metadata: Target metadata with run and target information.

        Returns:
            Visualization output with time series plot.
        """
        plotter = ForecastTimeSeriesPlotter()

        # Add measurements as the baseline
        plotter.add_measurements(report.subset.ground_truth)

        # Add forecast model with quantile predictions
        plotter.add_model(
            model_name=metadata.run_name,
            quantiles=report.subset.predictions,
        )

        # Add capacity limits for context
        TimeSeriesVisualization._add_capacity_limits(plotter, metadata.limit)

        figure = plotter.plot(title=f"Measurements vs Forecasts for {metadata.name}")
        return VisualizationOutput(name=self.name, figure=figure)

    def create_by_run_and_none(
        self,
        reports: dict[RunName, list[ReportTuple]],
    ) -> VisualizationOutput:
        """Create time series visualization comparing different model runs.

        Args:
            reports: Dictionary mapping run names to their report lists.

        Returns:
            Visualization output with time series comparison.
        """
        plotter = ForecastTimeSeriesPlotter()

        # Get reference data from the first target (all targets expected to be the same)
        first_metadata, first_report = TimeSeriesVisualization._get_first_target_data(reports)

        # Add measurements once (shared across all runs)
        plotter.add_measurements(first_report.subset.ground_truth)

        # Add capacity limits for context
        TimeSeriesVisualization._add_capacity_limits(plotter, first_metadata.limit)

        # Add forecast models for each run
        for run_name, run_reports in reports.items():
            for _metadata, report in run_reports:
                plotter.add_model(
                    model_name=run_name,
                    quantiles=report.subset.predictions,
                )

        figure = plotter.plot(title="Forecast Time Series Comparison")
        return VisualizationOutput(name=self.name, figure=figure)


__all__ = ["TimeSeriesVisualization"]
