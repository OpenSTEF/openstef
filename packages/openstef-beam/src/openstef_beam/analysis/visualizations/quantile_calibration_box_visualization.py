# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Quantile calibration boxplot visualization provider.

This module provides visualization for quantile probability analysis, comparing
calibration error distributions for forecasted quantiles to evaluate
probabilistic forecast calibration.
"""

from typing import override

from openstef_beam.analysis.models import AnalysisAggregation, RunName, VisualizationOutput
from openstef_beam.analysis.plots import QuantileCalibrationBoxPlotter
from openstef_beam.analysis.visualizations.base import ReportTuple
from openstef_beam.analysis.visualizations.quantile_probability_visualization import QuantileProbabilityVisualization


class QuantileCalibrationBoxVisualization(QuantileProbabilityVisualization):
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
            AnalysisAggregation.RUN_AND_NONE,
            AnalysisAggregation.RUN_AND_TARGET,
        }

    @override
    def create_by_run_and_none(self, reports: dict[RunName, list[ReportTuple]]) -> VisualizationOutput:
        plotter = QuantileCalibrationBoxPlotter()

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


__all__ = ["QuantileCalibrationBoxVisualization"]
