# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Turns evaluation results into visualizations and reports for decision making.

Numbers and metrics are hard to interpret. This module creates charts, plots, and
summary reports that help you understand model performance and make decisions about
which models to deploy. It automatically generates the visualizations most relevant
for energy forecasting operations.

What you get:
    - Performance charts: See how models compare across different metrics
    - Time series plots: Visualize predictions vs actual consumption over time
    - Error analysis: Understand when and why models make mistakes
    - Comparison reports: Side-by-side model performance analysis
"""

from openstef_beam.analysis import plots, visualizations
from openstef_beam.analysis.analysis_pipeline import AnalysisConfig, AnalysisPipeline
from openstef_beam.analysis.models import AnalysisOutput, AnalysisScope, VisualizationOutput

__all__ = [
    "AnalysisConfig",
    "AnalysisOutput",
    "AnalysisPipeline",
    "AnalysisScope",
    "VisualizationOutput",
    "plots",
    "visualizations",
]
