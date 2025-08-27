# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Organizes forecasting results into structured performance reports.

After running backtests, you get lots of predictions and actual values. This module
helps you make sense of them by calculating metrics across different time periods,
filtering for specific conditions (like weekends or peak hours), and organizing
everything into clear reports.

What it handles:
    - Time windows: Compare performance across days, weeks, seasons
    - Lead times: Evaluate how accuracy changes from 1-hour to 48-hour forecasts
    - Data filtering: Focus on specific conditions (peaks, weekdays, etc.)
    - Metric calculation: Apply the right metrics to the right data subsets
    - Report structure: Organize results for easy analysis and comparison

The evaluation produces raw numerical data (metrics, timestamps, values) that can
be fed into the analysis module for visualization and further interpretation.
"""

from openstef_beam.evaluation import metric_providers
from openstef_beam.evaluation.evaluation_pipeline import EvaluationConfig, EvaluationPipeline
from openstef_beam.evaluation.models import (
    EvaluationReport,
    EvaluationSubset,
    EvaluationSubsetReport,
    Filtering,
    SubsetMetric,
    Window,
)

__all__ = [
    "EvaluationConfig",
    "EvaluationPipeline",
    "EvaluationReport",
    "EvaluationSubset",
    "EvaluationSubsetReport",
    "Filtering",
    "SubsetMetric",
    "Window",
    "metric_providers",
]
