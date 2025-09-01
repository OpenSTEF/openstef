# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Quantile probability visualization provider.

This module provides visualization for quantile probability analysis, comparing
observed vs forecasted probabilities to evaluate probabilistic forecast calibration.
"""

from typing import NamedTuple, override

from openstef_beam.analysis.models import AnalysisAggregation, RunName, TargetMetadata, VisualizationOutput
from openstef_beam.analysis.plots import (
    QuantileProbabilityPlotter,
)
from openstef_beam.analysis.visualizations.base import ReportTuple, VisualizationProvider
from openstef_beam.evaluation import EvaluationSubsetReport
from openstef_core.types import Quantile


class ProbabilityData(NamedTuple):
    """Container for observed and forecasted probability data."""

    observed_probs: list[Quantile]
    forecasted_probs: list[Quantile]


class QuantileProbabilityVisualization(VisualizationProvider):
    """Creates calibration plots comparing observed vs forecasted probabilities.

    Evaluates calibration quality of probabilistic forecasts by plotting observed
    frequencies against forecasted probabilities for different quantile levels.
    Perfect calibration shows points along the diagonal where observed probability
    equals forecasted probability.

    Identifies forecast issues:
    - Overconfident predictions (points below diagonal)
    - Underconfident predictions (points above diagonal)
    - Systematic biases in uncertainty estimation
    - Overall forecast reliability across quantile ranges

    Supports comparison across different model runs, targets, and aggregation levels
    to evaluate which models provide better calibrated uncertainty estimates.

    Example:
        Basic usage in analysis pipeline:

        >>> from openstef_beam.analysis import AnalysisConfig
        >>> from openstef_beam.analysis.visualizations import QuantileProbabilityVisualization
        >>>
        >>> # Configure probability calibration analysis
        >>> analysis_config = AnalysisConfig(
        ...     visualization_providers=[
        ...         QuantileProbabilityVisualization(
        ...             name="probability_calibration",
        ...         ),
        ...     ]
        ... )
        >>>
        >>> # The visualization will generate calibration plots showing
        >>> # observed vs forecasted probabilities for model evaluation
    """

    @property
    @override
    def supported_aggregations(self) -> set[AnalysisAggregation]:
        return {
            AnalysisAggregation.NONE,
            AnalysisAggregation.RUN_AND_NONE,
            AnalysisAggregation.RUN_AND_TARGET,
            AnalysisAggregation.TARGET,
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
    def _create_plot_title(base_title: str, target_name: str = "") -> str:
        """Create a formatted title for the plot.

        Args:
            base_title: The base title describing the aggregation type
            target_name: Optional target name to include in the title

        Returns:
            Formatted plot title
        """
        if target_name and base_title:
            return f"Quantile Probability {base_title} for {target_name}"
        if target_name:
            return f"Quantile Probability for {target_name}"
        return f"Quantile Probability {base_title}"

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

    @override
    def create_by_none(
        self,
        report: EvaluationSubsetReport,
        metadata: TargetMetadata,
    ) -> VisualizationOutput:
        prob_data = self._extract_probabilities_from_report(report)
        self._validate_probability_data(prob_data)

        plotter = QuantileProbabilityPlotter()
        plotter.add_model(
            model_name=metadata.run_name,
            observed_probs=prob_data.observed_probs,
            forecasted_probs=prob_data.forecasted_probs,
        )

        title = self._create_plot_title("", metadata.name)
        figure = plotter.plot(title=title)

        return VisualizationOutput(name=self.name, figure=figure)

    @override
    def create_by_run_and_none(self, reports: dict[RunName, list[ReportTuple]]) -> VisualizationOutput:
        plotter = QuantileProbabilityPlotter()

        for run_name, report_pairs in reports.items():
            for _metadata, report in report_pairs:
                prob_data = self._extract_probabilities_from_report(report)
                self._validate_probability_data(prob_data)

                plotter.add_model(
                    model_name=run_name,
                    observed_probs=prob_data.observed_probs,
                    forecasted_probs=prob_data.forecasted_probs,
                )

        title = self._create_plot_title("by Run")
        figure = plotter.plot(title=title)

        return VisualizationOutput(name=self.name, figure=figure)

    @override
    def create_by_run_and_target(
        self,
        reports: dict[RunName, list[ReportTuple]],
    ) -> VisualizationOutput:
        return self.create_by_run_and_none(
            reports=reports,
        )

    @override
    def create_by_target(
        self,
        reports: list[ReportTuple],
    ) -> VisualizationOutput:
        if not reports:
            raise ValueError("No reports provided for target-based visualization.")

        plotter = QuantileProbabilityPlotter()

        # Get the run name from the first target metadata for the title
        run_name = reports[0][0].run_name

        for metadata, report in reports:
            prob_data = self._extract_probabilities_from_report(report)
            self._validate_probability_data(prob_data)

            plotter.add_model(
                model_name=metadata.name,
                observed_probs=prob_data.observed_probs,
                forecasted_probs=prob_data.forecasted_probs,
            )

        title = self._create_plot_title("by Target", run_name)
        figure = plotter.plot(title=title)

        return VisualizationOutput(name=self.name, figure=figure)


__all__ = ["QuantileProbabilityVisualization"]
