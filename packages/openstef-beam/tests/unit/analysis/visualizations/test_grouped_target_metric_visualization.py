# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for GroupedTargetMetricVisualization."""

from unittest.mock import call, patch

import pytest

from openstef_beam.analysis.models.target_metadata import TargetMetadata
from openstef_beam.analysis.plots import GroupedTargetMetricPlotter
from openstef_beam.analysis.visualizations import GroupedTargetMetricVisualization
from openstef_beam.analysis.visualizations.base import AnalysisAggregation, Quantile
from openstef_beam.evaluation.models.report import EvaluationSubsetReport
from tests.utils.mocks import MockFigure


@pytest.mark.parametrize(
    ("metric", "quantile", "selector_metric", "expected_name", "expected_is_selector"),
    [
        ("mae", None, None, "mae", False),
        ("precision", 0.5, None, "precision", False),
        ("effective_precision", None, "effective_F2", "effective_precision", True),
    ],
)
def test_metric_properties(
    metric: str, quantile: Quantile | None, selector_metric: str | None, expected_name: str, expected_is_selector: bool
):
    viz = GroupedTargetMetricVisualization(
        name="test_viz", metric=metric, quantile=quantile, selector_metric=selector_metric
    )
    assert viz._get_metric_name() == expected_name
    assert viz._is_selector_metric() == expected_is_selector


@pytest.mark.parametrize(
    ("metric", "quantile", "selector_metric", "expected_value"),
    [
        ("mae", None, None, 0.5),
        ("precision", 0.5, None, 0.9),
    ],
)
def test_extract_metric_value(
    sample_evaluation_report: EvaluationSubsetReport,
    metric: str,
    quantile: Quantile | None,
    selector_metric: str | None,
    expected_value: float,
):
    viz = GroupedTargetMetricVisualization(
        name="test_viz", metric=metric, quantile=quantile, selector_metric=selector_metric
    )
    value = viz._extract_metric_value(sample_evaluation_report)
    assert value == expected_value


def test_extract_metric_value_missing_returns_none(sample_evaluation_report: EvaluationSubsetReport):
    viz = GroupedTargetMetricVisualization(name="test_viz", metric="nonexistent")
    value = viz._extract_metric_value(sample_evaluation_report)
    assert value is None


def test_is_selector_metric():
    """Test that _is_selector_metric correctly identifies selector metrics."""
    # Regular string metric
    viz1 = GroupedTargetMetricVisualization(name="test_viz", metric="mae")
    assert not viz1._is_selector_metric()

    # Regular quantile metric
    viz2 = GroupedTargetMetricVisualization(name="test_viz", metric="precision", quantile=Quantile(0.5))
    assert not viz2._is_selector_metric()

    # Selector metric
    viz3 = GroupedTargetMetricVisualization(
        name="test_viz", metric="effective_precision", selector_metric="effective_F2"
    )
    assert viz3._is_selector_metric()


def test_extract_selector_metric_value(sample_evaluation_report_with_effective_metrics: EvaluationSubsetReport):
    """Test extraction of selector metric values."""
    viz = GroupedTargetMetricVisualization(
        name="test_viz", metric="effective_precision", selector_metric="effective_F2"
    )

    # Should return the effective_precision value for the quantile with best effective_F2
    value = viz._extract_metric_value(sample_evaluation_report_with_effective_metrics)
    assert value == 0.85  # effective_precision at quantile 0.5 (best effective_F2)


def test_find_best_quantile_for_selector(sample_evaluation_report_with_effective_metrics: EvaluationSubsetReport):
    """Test finding the best quantile for a selector metric."""
    viz = GroupedTargetMetricVisualization(
        name="test_viz", metric="effective_precision", selector_metric="effective_F2"
    )

    best_quantile = viz._find_best_quantile_for_selector(sample_evaluation_report_with_effective_metrics)
    assert best_quantile == 0.5  # Quantile with highest effective_F2 score


def test_extract_selector_metric_value_no_valid_quantile(sample_evaluation_report: EvaluationSubsetReport):
    """Test selector metric extraction when no valid quantile is found."""
    viz = GroupedTargetMetricVisualization(name="test_viz", metric="precision", selector_metric="nonexistent_metric")

    value = viz._extract_metric_value(sample_evaluation_report)
    assert value is None


def test_collect_target_metrics_aggregates_correctly(
    multiple_target_metadata: list[TargetMetadata], sample_evaluation_report: EvaluationSubsetReport
):
    """Test that collect_target_metrics correctly aggregates data from multiple targets."""
    viz = GroupedTargetMetricVisualization(name="test_viz", metric="mae")
    reports = [(metadata, sample_evaluation_report) for metadata in multiple_target_metadata]

    targets, values = viz._collect_target_metrics(reports)

    # Should collect all target names and their corresponding metric values
    assert targets == ["Target1", "Target2", "Target3"]
    assert values == [0.5, 0.5, 0.5]  # All use same report, so same values
    assert len(targets) == len(values) == 3


@pytest.mark.parametrize(
    ("metric", "quantile", "selector_metric", "expected_title"),
    [
        ("mae", None, None, "mae per Target"),
        ("precision", 0.5, None, "precision (q=0.5) per Target"),
        ("effective_precision", None, "effective_F2", "effective_precision (best effective_F2 quantile) per Target"),
    ],
)
def test_create_plot_title(metric: str, quantile: Quantile | None, selector_metric: str | None, expected_title: str):
    viz = GroupedTargetMetricVisualization(
        name="test_viz", metric=metric, quantile=quantile, selector_metric=selector_metric
    )
    title = viz._create_plot_title("per Target")
    assert title == expected_title


def test_create_by_target(
    multiple_target_metadata: list[TargetMetadata],
    sample_evaluation_report: EvaluationSubsetReport,
    mock_plotly_figure: MockFigure,
):
    """Test create_by_target properly calls plotter with correct data and creates visualization output."""
    viz = GroupedTargetMetricVisualization(name="test_viz", metric="mae")
    reports = [(metadata, sample_evaluation_report) for metadata in multiple_target_metadata]

    # Mock the plotter's plot method to return our mock figure
    with (
        patch.object(GroupedTargetMetricPlotter, "plot", return_value=mock_plotly_figure) as mock_plot,
        patch.object(GroupedTargetMetricPlotter, "add_model") as mock_add_model,
    ):
        result = viz.create_by_target(reports)

    # Verify add_model is called with correct aggregated data
    mock_add_model.assert_called_once_with(
        model_name="Targets", targets=["Target1", "Target2", "Target3"], metric_values=[0.5, 0.5, 0.5]
    )

    # Verify plot is called with correct title and metric name
    mock_plot.assert_called_once_with(title="mae per Target", metric_name="mae")

    # Verify return value is properly constructed
    assert result.name == viz.name
    assert result.figure == mock_plotly_figure


def test_create_by_target_no_valid_data_raises_error(
    empty_evaluation_report: EvaluationSubsetReport, simple_target_metadata: TargetMetadata
):
    """Test that create_by_target raises ValueError when no valid metric data is found."""
    viz = GroupedTargetMetricVisualization(name="test_viz", metric="nonexistent")
    reports = [(simple_target_metadata, empty_evaluation_report)]

    with pytest.raises(ValueError, match="No valid metric data found for 'nonexistent'"):
        viz.create_by_target(reports)


def test_create_by_run_and_group_ensures_unique_target_names(
    multiple_target_metadata: list[TargetMetadata],
    sample_evaluation_report: EvaluationSubsetReport,
    mock_plotly_figure: MockFigure,
):
    """Test that create_by_run_and_group makes target names unique when they have same names in different groups."""
    viz = GroupedTargetMetricVisualization(name="test_viz", metric="mae")
    reports = {
        ("Run1", "GroupA"): [(multiple_target_metadata[0], sample_evaluation_report)],
        ("Run2", "GroupB"): [(multiple_target_metadata[2], sample_evaluation_report)],
    }

    # Mock the plotter methods to capture calls and return our mock figure
    with (
        patch.object(GroupedTargetMetricPlotter, "plot", return_value=mock_plotly_figure) as mock_plot,
        patch.object(GroupedTargetMetricPlotter, "add_model") as mock_add_model,
        patch.object(GroupedTargetMetricPlotter, "set_target_groups") as mock_set_target_groups,
    ):
        result = viz.create_by_run_and_group(reports)

    # Verify add_model calls with unique target names
    expected_calls = [
        call(model_name="Run1", targets=["(GroupA) Target1"], metric_values=[0.5]),
        call(model_name="Run2", targets=["(GroupB) Target3"], metric_values=[0.5]),
    ]
    mock_add_model.assert_has_calls(expected_calls)
    assert mock_add_model.call_count == 2

    # Verify set_target_groups is called with unique target mapping
    mock_set_target_groups.assert_called_once()
    target_groups_arg = mock_set_target_groups.call_args[0][0]

    # Verify the target groups mapping is correct
    assert "(GroupA) Target1" in target_groups_arg
    assert "(GroupB) Target3" in target_groups_arg
    assert target_groups_arg["(GroupA) Target1"] == "GroupA"
    assert target_groups_arg["(GroupB) Target3"] == "GroupB"

    # Verify plot call
    mock_plot.assert_called_once_with(title="mae by Run and Target Group", metric_name="mae")

    # Verify return value
    assert result.name == viz.name
    assert result.figure == mock_plotly_figure


def test_create_by_group_with_unique_target_names(
    multiple_target_metadata: list[TargetMetadata],
    sample_evaluation_report: EvaluationSubsetReport,
    mock_plotly_figure: MockFigure,
):
    """Test create_by_group creates unique target names when targets have same names in different groups."""
    viz = GroupedTargetMetricVisualization(name="test_viz", metric="mae")

    # Create reports grouped by group name
    reports = {
        "GroupA": [
            (multiple_target_metadata[0], sample_evaluation_report),  # Target1
            (multiple_target_metadata[1], sample_evaluation_report),  # Target2
        ],
        "GroupB": [
            (multiple_target_metadata[2], sample_evaluation_report),  # Target3
        ],
    }

    # Mock the plotter methods to capture calls and return our mock figure
    with (
        patch.object(GroupedTargetMetricPlotter, "plot", return_value=mock_plotly_figure) as mock_plot,
        patch.object(GroupedTargetMetricPlotter, "add_model") as mock_add_model,
        patch.object(GroupedTargetMetricPlotter, "set_target_groups") as mock_set_target_groups,
    ):
        result = viz.create_by_group(reports)

    # Verify add_model is called for each group with unique target names
    expected_calls = [
        call(model_name="GroupA", targets=["(GroupA) Target1", "(GroupA) Target2"], metric_values=[0.5, 0.5]),
        call(model_name="GroupB", targets=["(GroupB) Target3"], metric_values=[0.5]),
    ]
    mock_add_model.assert_has_calls(expected_calls, any_order=True)

    # Verify set_target_groups is called with the correct mapping using unique names
    expected_target_groups = {
        "(GroupA) Target1": "GroupA",
        "(GroupA) Target2": "GroupA",
        "(GroupB) Target3": "GroupB",
    }
    mock_set_target_groups.assert_called_once_with(expected_target_groups)

    # Verify plot is called with correct title and metric name
    mock_plot.assert_called_once_with(title="mae by Target Group", metric_name="mae")

    # Verify return value is properly constructed
    assert result.name == viz.name
    assert result.figure == mock_plotly_figure


def test_create_by_group_empty_groups_raises_error(
    multiple_target_metadata: list[TargetMetadata],
    empty_evaluation_report: EvaluationSubsetReport,
    mock_plotly_figure: MockFigure,
):
    """Test that create_by_group raises error when all groups have no valid metric data."""
    viz = GroupedTargetMetricVisualization(name="test_viz", metric="nonexistent")

    # Create reports with empty evaluation reports
    reports = {
        "GroupA": [(multiple_target_metadata[0], empty_evaluation_report)],
        "GroupB": [(multiple_target_metadata[2], empty_evaluation_report)],
    }

    # Should raise ValueError when no valid data is found
    with pytest.raises(ValueError, match="No valid metric data found for 'nonexistent'"):
        viz.create_by_group(reports)


def test_create_by_group_handles_duplicate_target_names_across_groups(
    sample_evaluation_report: EvaluationSubsetReport, mock_plotly_figure: MockFigure
):
    """Test create_by_group handles duplicate target names across different groups correctly."""
    viz = GroupedTargetMetricVisualization(name="test_viz", metric="mae")

    # Create target metadata with same target names in different groups
    target_metadata_group_a = TargetMetadata(
        name="SameTarget", run_name="Run1", group_name="GroupA", filtering=None, limit=100.0
    )
    target_metadata_group_b = TargetMetadata(
        name="SameTarget", run_name="Run1", group_name="GroupB", filtering=None, limit=200.0
    )

    # Create reports with duplicate target names in different groups
    reports = {
        "GroupA": [(target_metadata_group_a, sample_evaluation_report)],
        "GroupB": [(target_metadata_group_b, sample_evaluation_report)],
    }

    # Mock the plotter methods to capture calls and return our mock figure
    with (
        patch.object(GroupedTargetMetricPlotter, "plot", return_value=mock_plotly_figure) as mock_plot,
        patch.object(GroupedTargetMetricPlotter, "add_model") as mock_add_model,
        patch.object(GroupedTargetMetricPlotter, "set_target_groups") as mock_set_target_groups,
    ):
        result = viz.create_by_group(reports)

    # Verify add_model is called with unique target identifiers
    expected_calls = [
        call(model_name="GroupA", targets=["(GroupA) SameTarget"], metric_values=[0.5]),
        call(model_name="GroupB", targets=["(GroupB) SameTarget"], metric_values=[0.5]),
    ]
    mock_add_model.assert_has_calls(expected_calls, any_order=True)

    # Verify set_target_groups is called with unique target identifiers
    expected_target_groups = {
        "(GroupA) SameTarget": "GroupA",
        "(GroupB) SameTarget": "GroupB",
    }
    mock_set_target_groups.assert_called_once_with(expected_target_groups)

    # Verify plot is called with correct title and metric name
    mock_plot.assert_called_once_with(title="mae by Target Group", metric_name="mae")

    # Verify return value is properly constructed
    assert result.name == viz.name
    assert result.figure == mock_plotly_figure


def test_supported_aggregations_returns_correct_set():
    """Test that supported_aggregations returns the expected aggregation types."""
    viz = GroupedTargetMetricVisualization(name="test_viz", metric="mae")
    expected = {
        AnalysisAggregation.TARGET,
        AnalysisAggregation.GROUP,
        AnalysisAggregation.RUN_AND_NONE,
        AnalysisAggregation.RUN_AND_GROUP,
        AnalysisAggregation.RUN_AND_TARGET,
    }
    assert viz.supported_aggregations == expected


def test_create_by_target_with_selector_metric(
    multiple_target_metadata: list[TargetMetadata],
    sample_evaluation_report_with_effective_metrics: EvaluationSubsetReport,
    mock_plotly_figure: MockFigure,
):
    """Test create_by_target with selector metric properly uses best quantile for each target."""
    viz = GroupedTargetMetricVisualization(
        name="test_viz", metric="effective_precision", selector_metric="effective_F2"
    )
    reports = [(metadata, sample_evaluation_report_with_effective_metrics) for metadata in multiple_target_metadata]

    # Mock the plotter's plot method to return our mock figure
    with (
        patch.object(GroupedTargetMetricPlotter, "plot", return_value=mock_plotly_figure) as mock_plot,
        patch.object(GroupedTargetMetricPlotter, "add_model") as mock_add_model,
    ):
        result = viz.create_by_target(reports)

    # Verify add_model is called with correct aggregated data
    # All reports use the same data, so all should get effective_precision from quantile 0.5 (best effective_F2)
    mock_add_model.assert_called_once_with(
        model_name="Targets", targets=["Target1", "Target2", "Target3"], metric_values=[0.85, 0.85, 0.85]
    )

    # Verify plot is called with correct title and metric name
    mock_plot.assert_called_once_with(
        title="effective_precision (best effective_F2 quantile) per Target", metric_name="effective_precision"
    )

    # Verify return value is properly constructed
    assert result.name == viz.name
    assert result.figure == mock_plotly_figure


def test_create_by_target_with_selector_metric_no_data_raises_error(
    multiple_target_metadata: list[TargetMetadata], sample_evaluation_report: EvaluationSubsetReport
):
    """Test that create_by_target with selector metric raises error when no valid selector data exists."""
    # Use a selector metric that doesn't exist in the sample data
    viz = GroupedTargetMetricVisualization(
        name="test_viz", metric="effective_precision", selector_metric="nonexistent_selector"
    )
    reports = [(metadata, sample_evaluation_report) for metadata in multiple_target_metadata]

    with pytest.raises(ValueError, match="No valid metric data found for 'effective_precision'"):
        viz.create_by_target(reports)


def test_create_by_run_with_selector_metric_no_data_raises_error(
    multiple_target_metadata: list[TargetMetadata], sample_evaluation_report: EvaluationSubsetReport
):
    """Test that create_by_run with selector metric raises error when no valid selector data exists."""
    viz = GroupedTargetMetricVisualization(
        name="test_viz", metric="effective_precision", selector_metric="nonexistent_selector"
    )
    reports = {
        "Run1": [(multiple_target_metadata[0], sample_evaluation_report)],
        "Run2": [(multiple_target_metadata[1], sample_evaluation_report)],
    }

    with pytest.raises(ValueError, match="No valid metric data found for 'effective_precision'"):
        viz.create_by_run_and_none(reports)


def test_create_by_run_and_target_with_selector_metric_no_data_raises_error(
    multiple_target_metadata: list[TargetMetadata], sample_evaluation_report: EvaluationSubsetReport
):
    """Test that create_by_run_and_target with selector metric raises error when no valid selector data exists."""
    viz = GroupedTargetMetricVisualization(
        name="test_viz", metric="effective_precision", selector_metric="nonexistent_selector"
    )
    reports = {
        "Run1": [(multiple_target_metadata[0], sample_evaluation_report)],
        "Run2": [(multiple_target_metadata[1], sample_evaluation_report)],
    }

    with pytest.raises(ValueError, match="No valid metric data found for 'effective_precision'"):
        viz.create_by_run_and_target(reports)


def test_create_by_group_with_selector_metric_no_data_raises_error(
    multiple_target_metadata: list[TargetMetadata], sample_evaluation_report: EvaluationSubsetReport
):
    """Test that create_by_group with selector metric raises error when no valid selector data exists."""
    viz = GroupedTargetMetricVisualization(
        name="test_viz", metric="effective_precision", selector_metric="nonexistent_selector"
    )
    reports = {
        "GroupA": [(multiple_target_metadata[0], sample_evaluation_report)],
        "GroupB": [(multiple_target_metadata[1], sample_evaluation_report)],
    }

    with pytest.raises(ValueError, match="No valid metric data found for 'effective_precision'"):
        viz.create_by_group(reports)


def test_create_by_run_and_group_with_selector_metric_no_data_raises_error(
    multiple_target_metadata: list[TargetMetadata], sample_evaluation_report: EvaluationSubsetReport
):
    """Test that create_by_run_and_group with selector metric raises error when no valid selector data exists."""
    viz = GroupedTargetMetricVisualization(
        name="test_viz", metric="effective_precision", selector_metric="nonexistent_selector"
    )
    reports = {
        ("Run1", "GroupA"): [(multiple_target_metadata[0], sample_evaluation_report)],
        ("Run2", "GroupB"): [(multiple_target_metadata[1], sample_evaluation_report)],
    }

    with pytest.raises(ValueError, match="No valid metric data found for 'effective_precision'"):
        viz.create_by_run_and_group(reports)
