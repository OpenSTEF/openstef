# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for QuantileProbabilityVisualization."""

from unittest.mock import call, patch

import pytest

from openstef_beam.analysis.models.target_metadata import TargetMetadata
from openstef_beam.analysis.plots import QuantileProbabilityPlotter
from openstef_beam.analysis.visualizations import QuantileProbabilityVisualization
from openstef_beam.analysis.visualizations.base import Quantile
from openstef_beam.analysis.visualizations.quantile_probability_visualization import ProbabilityData
from openstef_beam.evaluation.models.report import EvaluationSubsetReport
from tests.utils.mocks import MockFigure


@pytest.mark.parametrize(
    ("prob_data", "expected_error"),
    [
        pytest.param(
            ProbabilityData(observed_probs=[], forecasted_probs=[]),
            "No probability data found.",
            id="empty_data",
        ),
        pytest.param(
            ProbabilityData(
                observed_probs=[Quantile(0.1), Quantile(0.2)],
                forecasted_probs=[Quantile(0.1)],
            ),
            "Observed and forecasted probability counts must match.",
            id="mismatched_lengths",
        ),
    ],
)
def test_validate_probability_data_raises_error_for_invalid_data(prob_data: ProbabilityData, expected_error: str):
    """Test probability data validation for various invalid inputs."""
    # Arrange
    viz = QuantileProbabilityVisualization(name="test_viz")

    # Act & Assert
    with pytest.raises(ValueError, match=expected_error):
        viz._validate_probability_data(prob_data)


def test_validate_probability_data_passes_for_valid_data():
    """Test that valid probability data passes validation."""
    # Arrange
    viz = QuantileProbabilityVisualization(name="test_viz")
    valid_data = ProbabilityData(
        observed_probs=[Quantile(0.1), Quantile(0.2)],
        forecasted_probs=[Quantile(0.1), Quantile(0.5)],
    )

    # Act & Assert - should not raise
    viz._validate_probability_data(valid_data)


def test_extract_probabilities_from_report_returns_probability_data(
    sample_evaluation_report_with_probabilities: EvaluationSubsetReport,
):
    """Test extraction of probability data from evaluation report."""
    # Arrange
    viz = QuantileProbabilityVisualization(name="test_viz")

    # Act
    prob_data = viz._extract_probabilities_from_report(sample_evaluation_report_with_probabilities)

    # Assert
    # Verify extracted data structure and values
    assert len(prob_data.observed_probs) == 2
    assert len(prob_data.forecasted_probs) == 2
    assert prob_data.observed_probs == [Quantile(0.15), Quantile(0.55)]
    assert prob_data.forecasted_probs == [Quantile(0.1), Quantile(0.5)]


def test_extract_probabilities_raises_error_when_no_global_metrics(empty_evaluation_report: EvaluationSubsetReport):
    """Test error handling when no global metrics are available."""
    # Arrange
    viz = QuantileProbabilityVisualization(name="test_viz")

    # Act & Assert
    with pytest.raises(ValueError, match="No global metrics found in the report"):
        viz._extract_probabilities_from_report(empty_evaluation_report)


def test_create_by_none_generates_single_target_visualization(
    sample_evaluation_report_with_probabilities: EvaluationSubsetReport,
    simple_target_metadata: TargetMetadata,
    mock_plotly_figure: MockFigure,
):
    """Test create_by_none generates visualization for single target."""
    # Arrange
    viz = QuantileProbabilityVisualization(name="test_viz")

    # Act
    with (
        patch.object(QuantileProbabilityPlotter, "plot", return_value=mock_plotly_figure) as mock_plot,
        patch.object(QuantileProbabilityPlotter, "add_model") as mock_add_model,
    ):
        result = viz.create_by_none(sample_evaluation_report_with_probabilities, simple_target_metadata)

    # Assert
    # Verify model and plot calls with correct parameters
    mock_add_model.assert_called_once_with(
        model_name="TestRun",
        observed_probs=[Quantile(0.15), Quantile(0.55)],
        forecasted_probs=[Quantile(0.1), Quantile(0.5)],
    )
    mock_plot.assert_called_once_with(title="Quantile Probability for TestTarget")
    assert result.name == viz.name
    assert result.figure == mock_plotly_figure


def test_create_by_run_generates_multi_run_comparison(
    sample_evaluation_report_with_probabilities: EvaluationSubsetReport,
    multiple_target_metadata: list[TargetMetadata],
    mock_plotly_figure: MockFigure,
):
    """Test create_by_run compares multiple runs using run names as model identifiers."""
    # Arrange
    viz = QuantileProbabilityVisualization(name="test_viz")
    reports = {
        "Run1": [(multiple_target_metadata[0], sample_evaluation_report_with_probabilities)],
        "Run2": [(multiple_target_metadata[2], sample_evaluation_report_with_probabilities)],
    }

    # Act
    with (
        patch.object(QuantileProbabilityPlotter, "plot", return_value=mock_plotly_figure) as mock_plot,
        patch.object(QuantileProbabilityPlotter, "add_model") as mock_add_model,
    ):
        result = viz.create_by_run_and_none(reports)

    # Assert
    # Verify each run is added as a separate model
    mock_add_model.assert_has_calls([
        call(
            model_name="Run1",
            observed_probs=[Quantile(0.15), Quantile(0.55)],
            forecasted_probs=[Quantile(0.1), Quantile(0.5)],
        ),
        call(
            model_name="Run2",
            observed_probs=[Quantile(0.15), Quantile(0.55)],
            forecasted_probs=[Quantile(0.1), Quantile(0.5)],
        ),
    ])
    mock_plot.assert_called_once_with(title="Quantile Probability by Run")
    assert result.name == viz.name
    assert result.figure == mock_plotly_figure


def test_create_by_run_and_target_generates_multi_run_comparison(
    sample_evaluation_report_with_probabilities: EvaluationSubsetReport,
    multiple_target_metadata: list[TargetMetadata],
    mock_plotly_figure: MockFigure,
):
    """Test create_by_run compares multiple runs using run names as model identifiers."""
    # Arrange
    viz = QuantileProbabilityVisualization(name="test_viz")
    reports = {
        "Run1": [(multiple_target_metadata[0], sample_evaluation_report_with_probabilities)],
        "Run2": [(multiple_target_metadata[2], sample_evaluation_report_with_probabilities)],
    }

    # Act
    with (
        patch.object(QuantileProbabilityPlotter, "plot", return_value=mock_plotly_figure) as mock_plot,
        patch.object(QuantileProbabilityPlotter, "add_model") as mock_add_model,
    ):
        result = viz.create_by_run_and_target(reports)

    # Assert
    # Verify each run is added as a separate model
    mock_add_model.assert_has_calls([
        call(
            model_name="Run1",
            observed_probs=[Quantile(0.15), Quantile(0.55)],
            forecasted_probs=[Quantile(0.1), Quantile(0.5)],
        ),
        call(
            model_name="Run2",
            observed_probs=[Quantile(0.15), Quantile(0.55)],
            forecasted_probs=[Quantile(0.1), Quantile(0.5)],
        ),
    ])
    mock_plot.assert_called_once_with(title="Quantile Probability by Run")
    assert result.name == viz.name
    assert result.figure == mock_plotly_figure


def test_create_by_target_generates_multi_target_comparison(
    sample_evaluation_report_with_probabilities: EvaluationSubsetReport,
    multiple_target_metadata: list[TargetMetadata],
    mock_plotly_figure: MockFigure,
):
    """Test create_by_target compares multiple targets using target names as model identifiers."""
    # Arrange
    viz = QuantileProbabilityVisualization(name="test_viz")
    reports = [
        (multiple_target_metadata[0], sample_evaluation_report_with_probabilities),
        (multiple_target_metadata[1], sample_evaluation_report_with_probabilities),
    ]

    # Act
    with (
        patch.object(QuantileProbabilityPlotter, "plot", return_value=mock_plotly_figure) as mock_plot,
        patch.object(QuantileProbabilityPlotter, "add_model") as mock_add_model,
    ):
        result = viz.create_by_target(reports)

    # Assert
    # Verify each target is added as a separate model
    mock_add_model.assert_has_calls([
        call(
            model_name="Target1",
            observed_probs=[Quantile(0.15), Quantile(0.55)],
            forecasted_probs=[Quantile(0.1), Quantile(0.5)],
        ),
        call(
            model_name="Target2",
            observed_probs=[Quantile(0.15), Quantile(0.55)],
            forecasted_probs=[Quantile(0.1), Quantile(0.5)],
        ),
    ])
    mock_plot.assert_called_once_with(title="Quantile Probability by Target for Run1")
    assert result.name == viz.name
    assert result.figure == mock_plotly_figure


def test_create_by_target_raises_error_when_no_reports():
    """Test error handling for empty reports in target aggregation."""
    # Arrange
    viz = QuantileProbabilityVisualization(name="test_viz")

    # Act & Assert
    with pytest.raises(ValueError, match="No reports provided for target-based visualization"):
        viz.create_by_target([])
