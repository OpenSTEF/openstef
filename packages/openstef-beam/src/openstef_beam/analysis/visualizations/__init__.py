# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Visualization providers for analysis pipeline.

This package contains specialized visualization implementations that generate
plots and reports from evaluation data at different aggregation levels.
"""

from openstef_beam.analysis.visualizations.base import MetricIdentifier, ReportTuple, VisualizationProvider
from openstef_beam.analysis.visualizations.grouped_target_metric_visualization import GroupedTargetMetricVisualization
from openstef_beam.analysis.visualizations.precision_recall_curve_visualization import PrecisionRecallCurveVisualization
from openstef_beam.analysis.visualizations.quantile_calibration_box_visualization import (
    QuantileCalibrationBoxVisualization,
)
from openstef_beam.analysis.visualizations.quantile_probability_visualization import QuantileProbabilityVisualization
from openstef_beam.analysis.visualizations.summary_table_visualization import SummaryTableVisualization
from openstef_beam.analysis.visualizations.timeseries_visualization import TimeSeriesVisualization
from openstef_beam.analysis.visualizations.windowed_metric_visualization import WindowedMetricVisualization

__all__ = [
    "GroupedTargetMetricVisualization",
    "MetricIdentifier",
    "PrecisionRecallCurveVisualization",
    "QuantileCalibrationBoxVisualization",
    "QuantileProbabilityVisualization",
    "ReportTuple",
    "SummaryTableVisualization",
    "TimeSeriesVisualization",
    "VisualizationProvider",
    "WindowedMetricVisualization",
]
