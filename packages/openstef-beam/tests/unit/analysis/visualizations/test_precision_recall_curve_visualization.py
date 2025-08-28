# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for PrecisionRecallCurveVisualization."""

from unittest.mock import call, patch

import pytest

from openstef_beam.analysis.models.target_metadata import TargetMetadata
from openstef_beam.analysis.plots import PrecisionRecallCurvePlotter
from openstef_beam.analysis.visualizations import PrecisionRecallCurveVisualization
from openstef_beam.analysis.visualizations.base import AnalysisAggregation
from openstef_beam.evaluation.models.report import EvaluationSubsetReport
from tests.utils.mocks import MockFigure


@pytest.mark.parametrize(
    ("effective_mode", "expected_precision", "expected_recall"),
    [
        (False, "precision", "recall"),
        (True, "effective_precision", "effective_recall"),
    ],
)
def test_metric_names_based_on_mode(effective_mode: bool, expected_precision: str, expected_recall: str):
    """Test that metric names are correctly determined based on effective_precision_recall mode."""
    viz = PrecisionRecallCurveVisualization(name="test_viz", effective_precision_recall=effective_mode)
    assert viz._precision_metric_name == expected_precision
    assert viz._recall_metric_name == expected_recall


def test_extract_precision_recall_values_from_global_metrics(sample_evaluation_report: EvaluationSubsetReport):
    """Test extraction of precision-recall values from evaluation report global metrics."""
    viz = PrecisionRecallCurveVisualization(name="test_viz")
    precision_values, recall_values, quantiles = viz._extract_precision_recall_values(sample_evaluation_report)

    # Should extract values for all quantiles that have both precision and recall
    assert len(precision_values) == len(recall_values) == len(quantiles) == 3
    assert precision_values == [0.8, 0.9, 0.7]  # For quantiles 0.1, 0.5, 0.9
    assert recall_values == [0.6, 0.7, 0.8]
    assert quantiles == [0.1, 0.5, 0.9]


def test_extract_precision_recall_values_handles_missing_global_metrics(
    empty_evaluation_report: EvaluationSubsetReport,
):
    """Test that extraction raises appropriate error when no global metrics are available."""
    viz = PrecisionRecallCurveVisualization(name="test_viz")
    with pytest.raises(ValueError, match="No global metrics found in the report"):
        viz._extract_precision_recall_values(empty_evaluation_report)


def test_extract_precision_recall_values_filters_incomplete_pairs(sample_evaluation_report: EvaluationSubsetReport):
    """Test that only quantiles with both precision and recall values are included."""
    # The sample_evaluation_report should have precision and recall for quantiles 0.1, 0.5, 0.9
    # If one quantile was missing precision or recall, it should be filtered out
    viz = PrecisionRecallCurveVisualization(name="test_viz")
    precision_values, recall_values, quantiles = viz._extract_precision_recall_values(sample_evaluation_report)

    # All returned arrays should have same length (complete pairs only)
    assert len(precision_values) == len(recall_values) == len(quantiles)
    # Should contain valid values (not None)
    assert all(p is not None for p in precision_values)
    assert all(r is not None for r in recall_values)


@pytest.mark.parametrize(
    ("effective_mode", "context", "expected_title"),
    [
        (False, "TestTarget", "Precision-Recall Curve for TestTarget"),
        (True, "TestRun", "Effective Precision-Recall Curve for TestRun"),
    ],
)
def test_create_plot_title_formats_correctly(effective_mode: bool, context: str, expected_title: str):
    """Test that plot titles are correctly formatted based on mode and context."""
    viz = PrecisionRecallCurveVisualization(name="test_viz", effective_precision_recall=effective_mode)
    title = viz._create_plot_title(context)
    assert title == expected_title


def test_create_by_none_creates_single_model_visualization(
    sample_evaluation_report: EvaluationSubsetReport,
    simple_target_metadata: TargetMetadata,
    mock_plotly_figure: MockFigure,
):
    """Test create_by_none generates single-model precision-recall curve with correct data."""
    viz = PrecisionRecallCurveVisualization(name="test_viz")

    # Mock the plotter's plot method to return our mock figure
    with (
        patch.object(PrecisionRecallCurvePlotter, "plot", return_value=mock_plotly_figure) as mock_plot,
        patch.object(PrecisionRecallCurvePlotter, "add_model") as mock_add_model,
    ):
        result = viz.create_by_none(sample_evaluation_report, simple_target_metadata)

    # Verify model data is added correctly
    mock_add_model.assert_called_once_with(
        model_name="TestRun", precision_values=[0.8, 0.9, 0.7], recall_values=[0.6, 0.7, 0.8], quantiles=[0.1, 0.5, 0.9]
    )

    # Verify plot generation with correct title
    mock_plot.assert_called_once_with(title="Precision-Recall Curve for TestTarget")

    # Verify output construction
    assert result.name == viz.name
    assert result.figure == mock_plotly_figure


def test_create_by_target_adds_multiple_models(
    multiple_target_metadata: list[TargetMetadata],
    sample_evaluation_report: EvaluationSubsetReport,
    mock_plotly_figure: MockFigure,
):
    """Test create_by_target adds one model per target with correct target names."""
    viz = PrecisionRecallCurveVisualization(name="test_viz")
    reports = [(metadata, sample_evaluation_report) for metadata in multiple_target_metadata]

    # Mock the plotter methods to capture calls and return our mock figure
    with (
        patch.object(PrecisionRecallCurvePlotter, "plot", return_value=mock_plotly_figure) as mock_plot,
        patch.object(PrecisionRecallCurvePlotter, "add_model") as mock_add_model,
    ):
        result = viz.create_by_target(reports)

    # Verify one model is added per target
    assert mock_add_model.call_count == 3

    # Verify each model uses the correct target name
    expected_calls = [
        call(
            model_name="Target1",
            precision_values=[0.8, 0.9, 0.7],
            recall_values=[0.6, 0.7, 0.8],
            quantiles=[0.1, 0.5, 0.9],
        ),
        call(
            model_name="Target2",
            precision_values=[0.8, 0.9, 0.7],
            recall_values=[0.6, 0.7, 0.8],
            quantiles=[0.1, 0.5, 0.9],
        ),
        call(
            model_name="Target3",
            precision_values=[0.8, 0.9, 0.7],
            recall_values=[0.6, 0.7, 0.8],
            quantiles=[0.1, 0.5, 0.9],
        ),
    ]
    mock_add_model.assert_has_calls(expected_calls)

    # Verify plot title uses run name from first target
    mock_plot.assert_called_once_with(title="Precision-Recall Curve for Run1")

    # Verify output
    assert result.name == viz.name
    # Verify output
    assert result.name == viz.name
    assert result.figure == mock_plotly_figure


def test_create_by_run_groups_by_run_name(
    multiple_target_metadata: list[TargetMetadata],
    sample_evaluation_report: EvaluationSubsetReport,
    mock_plotly_figure: MockFigure,
):
    """Test create_by_run groups targets by run name and adds appropriate models."""
    viz = PrecisionRecallCurveVisualization(name="test_viz")
    reports = {
        "Run1": [(multiple_target_metadata[0], sample_evaluation_report)],
        "Run2": [(multiple_target_metadata[2], sample_evaluation_report)],
    }

    # Mock the plotter methods to capture calls and return our mock figure
    with (
        patch.object(PrecisionRecallCurvePlotter, "plot", return_value=mock_plotly_figure) as mock_plot,
        patch.object(PrecisionRecallCurvePlotter, "add_model") as mock_add_model,
    ):
        result = viz.create_by_run_and_none(reports)

    # Verify models are added per run
    assert mock_add_model.call_count == 2

    # Verify each model uses the correct run name
    expected_calls = [
        call(
            model_name="Run1",
            precision_values=[0.8, 0.9, 0.7],
            recall_values=[0.6, 0.7, 0.8],
            quantiles=[0.1, 0.5, 0.9],
        ),
        call(
            model_name="Run2",
            precision_values=[0.8, 0.9, 0.7],
            recall_values=[0.6, 0.7, 0.8],
            quantiles=[0.1, 0.5, 0.9],
        ),
    ]
    mock_add_model.assert_has_calls(expected_calls)

    # Verify plot title references first run
    mock_plot.assert_called_once_with(title="Precision-Recall Curve for Run Run1")

    # Verify output
    assert result.name == viz.name
    assert result.figure == mock_plotly_figure


def test_supported_aggregations_returns_correct_types():
    """Test that supported aggregations includes the expected types for precision-recall curves."""
    viz = PrecisionRecallCurveVisualization(name="test_viz")
    expected = {
        AnalysisAggregation.NONE,
        AnalysisAggregation.RUN_AND_NONE,
        AnalysisAggregation.TARGET,
    }
    assert viz.supported_aggregations == expected
