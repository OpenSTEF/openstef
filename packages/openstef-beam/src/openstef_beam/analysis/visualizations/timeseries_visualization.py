# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from openstef_beam.analysis.models import AnalysisAggregation, RunName, TargetMetadata, VisualizationOutput
from openstef_beam.analysis.plots import ForecastTimeSeriesPlotter
from openstef_beam.analysis.visualizations.base import ReportTuple, VisualizationProvider
from openstef_beam.evaluation import EvaluationSubsetReport


class TimeSeriesVisualization(VisualizationProvider):
    """Visualization for time series forecast comparisons across different aggregation levels.

    Displays measurements alongside forecast quantiles with capacity limits,
    enabling visual assessment of forecast accuracy and limit violations.
    """

    @property
    def supported_aggregations(self) -> set[AnalysisAggregation]:
        """Return the set of supported aggregation levels for this visualization."""
        return {AnalysisAggregation.NONE, AnalysisAggregation.RUN_AND_NONE}

    def _add_capacity_limits(self, plotter: ForecastTimeSeriesPlotter, limit: float) -> None:
        """Add upper and lower capacity limits to the plot.

        Args:
            plotter: The forecast time series plotter instance
            limit: The capacity limit value (positive)
        """
        plotter.add_limit(value=limit, name="Upper Limit")
        plotter.add_limit(value=-limit, name="Lower Limit")

    def _get_first_target_data(self, reports: dict[RunName, list[ReportTuple]]) -> ReportTuple:
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
        plotter = ForecastTimeSeriesPlotter()

        # Add measurements as the baseline
        plotter.add_measurements(report.subset.ground_truth)

        # Add forecast model with quantile predictions
        plotter.add_model(
            model_name=metadata.run_name,
            quantiles=report.subset.predictions,
        )

        # Add capacity limits for context
        self._add_capacity_limits(plotter, metadata.limit)

        figure = plotter.plot(title=f"Measurements vs Forecasts for {metadata.name}")
        return VisualizationOutput(name=self.name, figure=figure)

    def create_by_run_and_none(
        self,
        reports: dict[RunName, list[ReportTuple]],
    ) -> VisualizationOutput:
        plotter = ForecastTimeSeriesPlotter()

        # Get reference data from the first target (all targets expected to be the same)
        first_metadata, first_report = self._get_first_target_data(reports)

        # Add measurements once (shared across all runs)
        plotter.add_measurements(first_report.subset.ground_truth)

        # Add capacity limits for context
        self._add_capacity_limits(plotter, first_metadata.limit)

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
