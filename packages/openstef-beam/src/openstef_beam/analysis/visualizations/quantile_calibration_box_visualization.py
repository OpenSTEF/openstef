# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Quantile calibration boxplot visualization provider.

This module provides visualization for quantile probability analysis, comparing
calibration error distributions for forecasted quantiles to evaluate
probabilistic forecast calibration.
"""

from typing import NamedTuple, override

from openstef_beam.analysis.models import AnalysisAggregation, GroupName, RunName, VisualizationOutput
from openstef_beam.analysis.plots import QuantileCalibrationBoxPlotter
from openstef_beam.analysis.visualizations.base import ReportTuple, VisualizationProvider
from openstef_beam.evaluation import EvaluationSubsetReport
from openstef_core.types import Quantile


class ProbabilityData(NamedTuple):
    """Container for observed and forecasted probability data."""

    observed_probs: list[Quantile]
    forecasted_probs: list[Quantile]


class QuantileCalibrationBoxVisualization(VisualizationProvider):
    """Creates boxplot visualization for quantile calibration across multiple targets.

    Inherits from QuantileProbabilityVisualization to reuse data extraction and validation logic,
    but overrides the plotting methods to create boxplots.

    Boxplots are particularly useful for:
    - Comparing calibration across multiple targets
    - Showing distribution of calibration errors
    - Identifying outlier targets or systematic biases
    - Evaluating consistency of uncertainty estimates

    Example:
        Basic usage in analysis pipeline:

        >>> from openstef_beam.analysis import AnalysisConfig
        >>> from openstef_beam.analysis.visualizations import QuantileCalibrationBoxVisualization
        >>>
        >>> # Configure quantile calibration boxplot analysis
        >>> analysis_config = AnalysisConfig(
        ...     visualization_providers=[
        ...         QuantileCalibrationBoxVisualization(
        ...             name="quantile_calibration_boxplot",
        ...         ),
        ...     ]
        ... )
        >>>
        >>> # The visualization will generate boxplots showing calibration error
        >>> # distributions across multiple targets for comparative analysis
    """

    @property
    @override
    def supported_aggregations(self) -> set[AnalysisAggregation]:
        """Boxplot visualization requires multiple targets, so NONE aggregation is excluded."""
        return {
            AnalysisAggregation.TARGET,
            AnalysisAggregation.GROUP,
            AnalysisAggregation.RUN_AND_TARGET,
            AnalysisAggregation.RUN_AND_GROUP,
        }

    @staticmethod
    def _extract_probabilities_from_report(report: EvaluationSubsetReport) -> ProbabilityData:
        """Extract observed and forecasted probability metrics from the report's global metrics.

        Args:
            report: The evaluation subset report containing global metrics

        Returns:
            ProbabilityData containing lists of observed and forecasted probabilities

        Raises:
            ValueError: If no global metrics are found or if probability data is inconsistent
        """
        global_metrics = report.get_global_metric()
        if global_metrics is None:
            raise ValueError("No global metrics found in the report.")

        quantile_columns = global_metrics.get_quantiles()
        observed_probs: list[Quantile] = []
        forecasted_probs: list[Quantile] = []

        for quantile in quantile_columns:
            quantile_metrics = global_metrics.metrics.get(quantile)
            if quantile_metrics is not None:
                observed_prob = quantile_metrics.get("observed_probability")
                if observed_prob is not None:
                    observed_probs.append(Quantile(observed_prob))
                    forecasted_probs.append(quantile)

        if len(observed_probs) != len(forecasted_probs):
            raise ValueError("Could not find all the observed probability metrics.")

        return ProbabilityData(observed_probs=observed_probs, forecasted_probs=forecasted_probs)

    @staticmethod
    def _validate_probability_data(prob_data: ProbabilityData) -> None:
        """Validate that probability data is consistent and complete.

        Args:
            prob_data: The probability data to validate

        Raises:
            ValueError: If probability data is invalid or incomplete
        """
        if len(prob_data.observed_probs) != len(prob_data.forecasted_probs):
            raise ValueError("Observed and forecasted probability counts must match.")

        if not prob_data.observed_probs:
            raise ValueError("No probability data found.")

    @staticmethod
    def _create_plot_title(base_title: str, target_name: str = "") -> str:
        """Create a formatted title for the plot.

        Args:
            base_title: The base title describing the aggregation type
            target_name: Optional target name to include in the title

        Returns:
            Formatted plot title
        """
        if target_name and base_title:
            return f"Quantile Calibration {base_title} for {target_name}"
        if target_name:
            return f"Quantile Calibration for {target_name}"
        return f"Quantile Calibration {base_title}"

    # making boxplots over all targets in a single group
    @override
    def create_by_target(
        self,
        reports: list[ReportTuple],
    ) -> VisualizationOutput:
        if not reports:
            raise ValueError("No reports provided for target-based visualization.")

        plotter = QuantileCalibrationBoxPlotter()

        # Get the run name from the first target metadata for the title
        run_name = reports[0][0].run_name

        for _metadata, report in reports:
            prob_data = self._extract_probabilities_from_report(report)
            self._validate_probability_data(prob_data)

            plotter.add_model(
                model_name=run_name,
                observed_probs=prob_data.observed_probs,
                forecasted_probs=prob_data.forecasted_probs,
            )

        title = self._create_plot_title("over all targets in group", run_name)
        figure = plotter.plot(title=title)

        return VisualizationOutput(name=self.name, figure=figure)

    # to plot all targets together (also when in different groups)
    @override
    def create_by_group(
        self,
        reports: dict[GroupName, list[ReportTuple]],
    ) -> VisualizationOutput:
        plotter = QuantileCalibrationBoxPlotter()

        # Use the first run name for the title (if available)
        first_group = next(iter(reports.values()), None)
        run_name = first_group[0][0].run_name if first_group and first_group[0] else ""

        for report_list in reports.values():
            for _metadata, report in report_list:
                prob_data = self._extract_probabilities_from_report(report)
                self._validate_probability_data(prob_data)

                plotter.add_model(
                    model_name=run_name,
                    observed_probs=prob_data.observed_probs,
                    forecasted_probs=prob_data.forecasted_probs,
                )

        title = self._create_plot_title("over all targets", run_name)
        figure = plotter.plot(title=title)

        return VisualizationOutput(name=self.name, figure=figure)

    # making boxplots over all targets in a single group
    @override
    def create_by_run_and_target(self, reports: dict[RunName, list[ReportTuple]]) -> VisualizationOutput:
        plotter = QuantileCalibrationBoxPlotter()

        for run_name, report_list in reports.items():
            for _metadata, report in report_list:
                prob_data = self._extract_probabilities_from_report(report)
                self._validate_probability_data(prob_data)

                plotter.add_model(
                    model_name=run_name,
                    observed_probs=prob_data.observed_probs,
                    forecasted_probs=prob_data.forecasted_probs,
                )

        title = self._create_plot_title("by run and over all targets in group")
        figure = plotter.plot(title=title)

        return VisualizationOutput(name=self.name, figure=figure)

    # to plot all targets together (also when in different groups)
    @override
    def create_by_run_and_group(
        self,
        reports: dict[tuple[RunName, GroupName], list[ReportTuple]],
    ) -> VisualizationOutput:
        plotter = QuantileCalibrationBoxPlotter()

        for (run_name, _group_name), report_list in reports.items():
            for _metadata, report in report_list:
                prob_data = self._extract_probabilities_from_report(report)
                self._validate_probability_data(prob_data)

                plotter.add_model(
                    model_name=run_name,
                    observed_probs=prob_data.observed_probs,
                    forecasted_probs=prob_data.forecasted_probs,
                )

        title = self._create_plot_title("by run and over all targets")
        figure = plotter.plot(title=title)

        return VisualizationOutput(name=self.name, figure=figure)


__all__ = ["QuantileCalibrationBoxVisualization"]
