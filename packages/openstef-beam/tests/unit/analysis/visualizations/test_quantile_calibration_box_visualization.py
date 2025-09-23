# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for QuantileCalibrationBoxVisualization - focuses on boxplot-specific visualization logic only.
Rest of functionality is tested in test_quantile_probability_visualization.py"""

from openstef_beam.analysis.models import AnalysisAggregation
from openstef_beam.analysis.visualizations import QuantileCalibrationBoxVisualization
from openstef_beam.analysis.visualizations.quantile_probability_visualization import QuantileProbabilityVisualization


def test_boxplot_visualization_distinct_from_line_visualization():
    """Test that boxplot visualization has different supported aggregations than parent."""
    # Arrange & Act
    viz = QuantileCalibrationBoxVisualization(name="test_viz")

    # Assert
    # Should be its own class
    assert viz.__class__.__name__ == "QuantileCalibrationBoxVisualization"

    # Should inherit from QuantileProbabilityVisualization but have different behavior
    assert isinstance(viz, QuantileProbabilityVisualization)

    # Should have different supported aggregations than parent (excludes TARGET)
    parent_viz = QuantileProbabilityVisualization(name="parent_test")
    assert viz.supported_aggregations != parent_viz.supported_aggregations
    # Parent should support TARGET, boxplot should not
    assert AnalysisAggregation.TARGET in parent_viz.supported_aggregations
    assert AnalysisAggregation.TARGET not in viz.supported_aggregations
