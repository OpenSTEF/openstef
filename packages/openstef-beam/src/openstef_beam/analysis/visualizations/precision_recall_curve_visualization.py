# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from openstef_beam.analysis.models import AnalysisAggregation, RunName, TargetMetadata, VisualizationOutput
from openstef_beam.analysis.plots import PrecisionRecallCurvePlotter
from openstef_beam.analysis.visualizations.base import ReportTuple, VisualizationProvider
from openstef_beam.evaluation import EvaluationSubsetReport
from openstef_core.types import Quantile


class PrecisionRecallCurveVisualization(VisualizationProvider):
    """Visualization for precision-recall curves across different aggregation levels."""

    effective_precision_recall: bool = False

    @property
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
