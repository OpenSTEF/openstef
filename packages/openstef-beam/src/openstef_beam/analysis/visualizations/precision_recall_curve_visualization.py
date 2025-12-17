# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Precision-recall curve visualization provider.

This module provides visualization for precision-recall curves, useful for
evaluating binary classification performance at different threshold levels.
"""

from typing import override

from openstef_beam.analysis.models import AnalysisAggregation, RunName, TargetMetadata, VisualizationOutput
from openstef_beam.analysis.plots import PrecisionRecallCurvePlotter
from openstef_beam.analysis.visualizations.base import ReportTuple, VisualizationProvider
from openstef_beam.evaluation import EvaluationSubsetReport
from openstef_core.types import Quantile


class PrecisionRecallCurveVisualization(VisualizationProvider):
    """Creates precision-recall curves for evaluating binary classification performance.

    Displays the classic precision-recall trade-off as a curve where each point represents
    performance at a different probability threshold. The closer the curve to the top-right
    corner, the better the model performs across all thresholds.

    Two evaluation modes:
    - Standard: Traditional precision/recall based on binary classification accuracy
    - Effective: Specialized for congestion management, evaluating whether forecasts
      provide actionable insights for grid operators (correct direction + sufficient magnitude)

    What you'll see:
    - Curve plotting precision (Y-axis) vs recall (X-axis)
    - Each point represents a different probability threshold
    - Area under curve (AUC-PR) as overall performance metric
    - Multiple curves when comparing models or targets
    - Reference lines showing random classifier performance

    Interpretation guide:
    - High precision: Few false alarms when predicting events
    - High recall: Catches most actual events that occur
    - Effective mode: Focuses on operationally useful predictions for grid management

    Example:
        >>> from openstef_beam.analysis import AnalysisConfig
        >>> from openstef_beam.analysis.visualizations import PrecisionRecallCurveVisualization
        >>>
        >>> analysis_config = AnalysisConfig(
        ...     visualization_providers=[
        ...         PrecisionRecallCurveVisualization(
        ...             name="precision_recall",
        ...             effective_precision_recall=True,  # For congestion management
        ...         ),
        ...     ]
        ... )
    """

    effective_precision_recall: bool = False

    @property
    @override
    def supported_aggregations(self) -> set[AnalysisAggregation]:
        return {
            AnalysisAggregation.NONE,
            AnalysisAggregation.RUN_AND_NONE,
            AnalysisAggregation.TARGET,
        }

    @property
    def _precision_metric_name(self) -> str:
        return "effective_precision" if self.effective_precision_recall else "precision"

    @property
    def _recall_metric_name(self) -> str:
        return "effective_recall" if self.effective_precision_recall else "recall"

    def _extract_precision_recall_values(
        self, report: EvaluationSubsetReport
    ) -> tuple[list[float], list[float], list[Quantile]]:
        """Extract precision and recall values from evaluation report.

        Args:
            report: The evaluation subset report

        Returns:
            A tuple of (precision_values, recall_values, quantiles)

        Raises:
            ValueError: If no global metrics are found
        """
        global_metrics = report.get_global_metric()
        if global_metrics is None:
            raise ValueError("No global metrics found in the report.")

        quantiles = global_metrics.get_quantiles()
        precision_values: list[float] = []
        recall_values: list[float] = []

        for quantile in quantiles:
            quantile_metrics = global_metrics.metrics.get(quantile, {})
            precision = quantile_metrics.get(self._precision_metric_name)
            recall = quantile_metrics.get(self._recall_metric_name)

            if precision is not None and recall is not None:
                precision_values.append(precision)
                recall_values.append(recall)

        return precision_values, recall_values, quantiles

    def _create_plot_title(self, context: str) -> str:
        """Create a formatted title for the plot.

        Args:
            context: The context description (e.g., target name, run name)

        Returns:
            Formatted plot title
        """
        curve_type = "Effective Precision-Recall" if self.effective_precision_recall else "Precision-Recall"
        return f"{curve_type} Curve for {context}"

    @override
    def create_by_none(
        self,
        report: EvaluationSubsetReport,
        metadata: TargetMetadata,
    ) -> VisualizationOutput:
        plotter = PrecisionRecallCurvePlotter()
        precision_values, recall_values, quantiles = self._extract_precision_recall_values(report)

        plotter.add_model(
            model_name=metadata.run_name,
            precision_values=precision_values,
            recall_values=recall_values,
            quantiles=quantiles,
        )

        title = self._create_plot_title(metadata.name)
        figure = plotter.plot(title=title)

        return VisualizationOutput(name=self.name, figure=figure)

    @override
    def create_by_run_and_none(self, reports: dict[RunName, list[ReportTuple]]) -> VisualizationOutput:
        plotter = PrecisionRecallCurvePlotter()

        # Get the first run name for the title (since we're aggregating by run)
        first_run_name = next(iter(reports.keys())) if reports else ""

        for run_name, run_reports in reports.items():
            for _, report in run_reports:
                precision_values, recall_values, quantiles = self._extract_precision_recall_values(report)

                plotter.add_model(
                    model_name=run_name,
                    precision_values=precision_values,
                    recall_values=recall_values,
                    quantiles=quantiles,
                )

        title = self._create_plot_title(f"Run {first_run_name}")
        figure = plotter.plot(title=title)

        return VisualizationOutput(name=self.name, figure=figure)

    @override
    def create_by_target(
        self,
        reports: list[ReportTuple],
    ) -> VisualizationOutput:
        plotter = PrecisionRecallCurvePlotter()

        # Get the run name from the first target metadata for the title
        run_name = reports[0][0].run_name if reports else ""

        for metadata, report in reports:
            precision_values, recall_values, quantiles = self._extract_precision_recall_values(report)

            plotter.add_model(
                model_name=metadata.name,
                precision_values=precision_values,
                recall_values=recall_values,
                quantiles=quantiles,
            )

        title = self._create_plot_title(run_name)
        figure = plotter.plot(title=title)

        return VisualizationOutput(name=self.name, figure=figure)


__all__ = ["PrecisionRecallCurveVisualization"]
