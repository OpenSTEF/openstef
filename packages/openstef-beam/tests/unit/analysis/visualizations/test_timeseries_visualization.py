# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for TimeSeriesVisualization."""

from unittest.mock import call, patch

import pytest

from openstef_beam.analysis.models.target_metadata import TargetMetadata
from openstef_beam.analysis.plots import ForecastTimeSeriesPlotter
from openstef_beam.analysis.visualizations import TimeSeriesVisualization
from openstef_beam.analysis.visualizations.base import AnalysisAggregation, ReportTuple, RunName
from openstef_beam.evaluation.models.report import EvaluationSubsetReport
from tests.utils.mocks import MockFigure


def test_supported_aggregations_includes_none_and_run():
    """Test that supported aggregations includes none and run aggregation types."""
    # Arrange
    viz = TimeSeriesVisualization(name="test_viz")
    expected = {AnalysisAggregation.NONE, AnalysisAggregation.RUN_AND_NONE}

    # Act
    result = viz.supported_aggregations

    # Assert
    assert result == expected


@pytest.mark.parametrize(
    ("reports", "expected_error"),
    [
        pytest.param({}, "No reports provided for time series visualization.", id="empty_reports"),
        pytest.param({"Run1": []}, "No target reports found in the first run.", id="empty_first_run"),
    ],
)
def test_get_first_target_data_raises_error_for_invalid_input(
    reports: dict[RunName, list[ReportTuple]], expected_error: str
):
    """Test error handling for invalid inputs across methods."""
    # Arrange
    viz = TimeSeriesVisualization(name="test_viz")

    # Act & Assert
    with pytest.raises(ValueError, match=expected_error):
        viz._get_first_target_data(reports)


def test_create_by_run_raises_error_when_no_reports():
    """Test that create_by_run raises error when given empty reports dictionary."""
    # Arrange
    viz = TimeSeriesVisualization(name="test_viz")

    # Act & Assert
    with pytest.raises(ValueError, match="No reports provided for time series visualization"):
        viz.create_by_run_and_none({})


def test_create_by_none_generates_single_target_time_series(
    sample_evaluation_report: EvaluationSubsetReport,
    simple_target_metadata: TargetMetadata,
    mock_plotly_figure: MockFigure,
):
    """Test create_by_none produces complete visualization for single target."""
    # Arrange
    viz = TimeSeriesVisualization(name="test_viz")

    # Act
    with (
        patch.object(ForecastTimeSeriesPlotter, "plot", return_value=mock_plotly_figure) as mock_plot,
        patch.object(ForecastTimeSeriesPlotter, "add_measurements") as mock_add_measurements,
        patch.object(ForecastTimeSeriesPlotter, "add_model") as mock_add_model,
        patch.object(ForecastTimeSeriesPlotter, "add_limit") as mock_add_limit,
    ):
        result = viz.create_by_none(sample_evaluation_report, simple_target_metadata)

    # Assert
    # Verify all expected calls are made with correct parameters
    mock_add_measurements.assert_called_once_with(sample_evaluation_report.subset.ground_truth)
    mock_add_model.assert_called_once_with(model_name="TestRun", quantiles=sample_evaluation_report.subset.predictions)
    mock_add_limit.assert_has_calls([
        call(value=100.0, name="Upper Limit"),
        call(value=-100.0, name="Lower Limit"),
    ])
    mock_plot.assert_called_once_with(title="Measurements vs Forecasts for TestTarget")

    # Verify output structure
    assert result.name == viz.name
    assert result.figure == mock_plotly_figure


def test_create_by_run_generates_multi_run_comparison(
    sample_evaluation_report: EvaluationSubsetReport,
    multiple_target_metadata: list[TargetMetadata],
    mock_plotly_figure: MockFigure,
):
    """Test create_by_run generates multi-run comparison using first target's limits."""
    # Arrange
    viz = TimeSeriesVisualization(name="test_viz")

    # Use targets with different limits to verify first target's limit is used
    reports = {
        "Run1": [(multiple_target_metadata[0], sample_evaluation_report)],  # limit=100.0
        "Run2": [(multiple_target_metadata[2], sample_evaluation_report)],  # limit=150.0
    }

    # Act
    with (
        patch.object(ForecastTimeSeriesPlotter, "plot", return_value=mock_plotly_figure) as mock_plot,
        patch.object(ForecastTimeSeriesPlotter, "add_measurements") as mock_add_measurements,
        patch.object(ForecastTimeSeriesPlotter, "add_model") as mock_add_model,
        patch.object(ForecastTimeSeriesPlotter, "add_limit") as mock_add_limit,
    ):
        result = viz.create_by_run_and_none(reports)

    # Assert
    # Verify measurements are shared and limits use first target's value
    mock_add_measurements.assert_called_once_with(sample_evaluation_report.subset.ground_truth)
    mock_add_limit.assert_has_calls([
        call(value=100.0, name="Upper Limit"),  # First target's limit, not 150.0
        call(value=-100.0, name="Lower Limit"),
    ])

    # Verify each run is added as separate model
    mock_add_model.assert_has_calls([
        call(model_name="Run1", quantiles=sample_evaluation_report.subset.predictions),
        call(model_name="Run2", quantiles=sample_evaluation_report.subset.predictions),
    ])

    mock_plot.assert_called_once_with(title="Forecast Time Series Comparison")
    assert result.name == viz.name
    assert result.figure == mock_plotly_figure
