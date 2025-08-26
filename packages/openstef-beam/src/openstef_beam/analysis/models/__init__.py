# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Data models for analysis pipeline components.

This package provides data models and type definitions for the analysis pipeline,
including aggregation types, output structures, and metadata containers.
"""

from openstef_beam.analysis.models.target_metadata import GroupName, RunName, TargetMetadata, TargetName
from openstef_beam.analysis.models.visualization_aggregation import AnalysisAggregation
from openstef_beam.analysis.models.visualization_output import AnalysisOutput, AnalysisScope, VisualizationOutput

__all__ = [
    "AnalysisAggregation",
    "AnalysisOutput",
    "AnalysisScope",
    "GroupName",
    "RunName",
    "TargetMetadata",
    "TargetName",
    "VisualizationOutput",
]
