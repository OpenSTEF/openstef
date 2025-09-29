# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for WindowedMetricVisualization."""

import copy
from datetime import datetime
from unittest.mock import call, patch

import numpy as np
import pytest

from openstef_beam.analysis.models.target_metadata import TargetMetadata
from openstef_beam.analysis.plots import WindowedMetricPlotter
from openstef_beam.analysis.visualizations import MetricIdentifier, WindowedMetricVisualization
from openstef_beam.analysis.visualizations.base import AnalysisAggregation
from openstef_beam.evaluation.models.report import EvaluationSubsetReport
from openstef_beam.evaluation.models.window import Window
from tests.utils.mocks import MockFigure


@pytest.mark.parametrize(
    ("metric", "expected_name", "expected_quantile"),
    [
        ("mae", "mae", "global"),
        (("precision", 0.5), "precision", 0.5),
    ],
)
def test_get_metric_info_parses_metric_configuration(
    metric: MetricIdentifier, expected_name: str, expected_quantile: float | str, sample_window: Window
):
    """Test that metric configuration is correctly parsed into name and quantile/global type."""
    viz = WindowedMetricVisualization(name="test_viz", metric=metric, window=sample_window)
    name, quantile_or_global = viz._get_metric_info()
    assert name == expected_name
    assert quantile_or_global == expected_quantile


@pytest.mark.parametrize(
    ("metric", "expected_pairs"),
    [
        (
            "mae",
            [
                (datetime.fromisoformat("2023-01-01T01:00:00"), 0.4),
                (datetime.fromisoformat("2023-01-01T02:00:00"), 0.6),
            ],
        ),
        (
            ("precision", 0.5),
            [
                (datetime.fromisoformat("2023-01-01T01:00:00"), 0.85),
                (datetime.fromisoformat("2023-01-01T02:00:00"), 0.75),
            ],
        ),
    ],
)
def test_extract_windowed_metric_values_filters_by_window_and_metric(
    sample_evaluation_report: EvaluationSubsetReport,
    sample_window: Window,
    metric: MetricIdentifier,
    expected_pairs: list[tuple[datetime, float]],
):
    """Test extraction of time-value pairs for specific window and metric configuration."""
    viz = WindowedMetricVisualization(name="test_viz", metric=metric, window=sample_window)

    if isinstance(metric, str):
        name, quantile_or_global = metric, "global"
    else:
        name, quantile_or_global = metric

    time_value_pairs = viz._extract_windowed_metric_values(sample_evaluation_report, name, quantile_or_global)

    # Should return sorted pairs for the specified window and metric
    assert time_value_pairs == expected_pairs
    # Verify chronological order
    timestamps = [pair[0] for pair in time_value_pairs]
    assert timestamps == sorted(timestamps)


def test_extract_windowed_metric_values_returns_empty_for_no_data(
    empty_evaluation_report: EvaluationSubsetReport, sample_window: Window
):
    """Test that extraction returns empty list when no windowed metrics are available."""
    viz = WindowedMetricVisualization(name="test_viz", metric="mae", window=sample_window)
    time_value_pairs = viz._extract_windowed_metric_values(empty_evaluation_report, "mae", "global")
    assert time_value_pairs == []


def test_extract_windowed_metric_values_handles_missing_metric_values(
    sample_evaluation_report: EvaluationSubsetReport, sample_window: Window
):
    """Test that extraction handles cases where some time windows don't have the requested metric."""
    viz = WindowedMetricVisualization(name="test_viz", metric="nonexistent", window=sample_window)
    time_value_pairs = viz._extract_windowed_metric_values(sample_evaluation_report, "nonexistent", "global")
    # Should return empty list when metric doesn't exist
    assert time_value_pairs == []


@pytest.mark.parametrize(
    ("metric", "expected_title"),
    [
        ("mae", "Windowed mae (lag=PT1H,size=PT2H,stride=PT1H) over Time for TestTarget"),
        (("precision", 0.5), "Windowed precision (q=0.5) (lag=PT1H,size=PT2H,stride=PT1H) over Time for TestTarget"),
    ],
)
def test_create_plot_title_formats_window_and_metric_info(
    metric: MetricIdentifier, expected_title: str, sample_window: Window
):
    """Test that plot titles correctly include window specification and metric details."""
    viz = WindowedMetricVisualization(name="test_viz", metric=metric, window=sample_window)
    metric_name, quantile_or_global = viz._get_metric_info()
    title = viz._create_plot_title(metric_name, quantile_or_global, "for TestTarget")
    assert title == expected_title


def test_create_by_none_creates_time_series_visualization(
    sample_evaluation_report: EvaluationSubsetReport,
    simple_target_metadata: TargetMetadata,
    sample_window: Window,
    mock_plotly_figure: MockFigure,
):
    """Test create_by_none generates time series visualization with correct timestamps and values."""
    viz = WindowedMetricVisualization(name="test_viz", metric="mae", window=sample_window)

    # Mock the plotter's plot method to return our mock figure
    with (
        patch.object(WindowedMetricPlotter, "plot", return_value=mock_plotly_figure) as mock_plot,
        patch.object(WindowedMetricPlotter, "add_model") as mock_add_model,
    ):
        result = viz.create_by_none(sample_evaluation_report, simple_target_metadata)

    # Verify time series data is added correctly
    mock_add_model.assert_called_once_with(
        model_name="TestRun",
        timestamps=[datetime.fromisoformat("2023-01-01T01:00:00"), datetime.fromisoformat("2023-01-01T02:00:00")],
        metric_values=[0.4, 0.6],
    )

    # Verify plot generation with descriptive title
    expected_title = "Windowed mae (lag=PT1H,size=PT2H,stride=PT1H) over Time for TestTarget"
    mock_plot.assert_called_once_with(title=expected_title)

    # Verify output construction
    assert result.name == viz.name
    assert result.figure == mock_plotly_figure


def test_create_by_none_raises_error_when_no_windowed_data(
    empty_evaluation_report: EvaluationSubsetReport, simple_target_metadata: TargetMetadata, sample_window: Window
):
    """Test that create_by_none raises appropriate error when no windowed metrics are found."""
    viz = WindowedMetricVisualization(name="test_viz", metric="mae", window=sample_window)

    with pytest.raises(ValueError, match="No windowed metrics found for the specified window and metric"):
        viz.create_by_none(empty_evaluation_report, simple_target_metadata)


def test_create_by_target_adds_time_series_per_target(
    multiple_target_metadata: list[TargetMetadata],
    sample_evaluation_report: EvaluationSubsetReport,
    sample_window: Window,
    mock_plotly_figure: MockFigure,
):
    """Test create_by_target adds separate time series for each target."""
    viz = WindowedMetricVisualization(name="test_viz", metric="mae", window=sample_window)
    reports = [(metadata, sample_evaluation_report) for metadata in multiple_target_metadata]

    # Mock the plotter methods to capture calls and return our mock figure
    with (
        patch.object(WindowedMetricPlotter, "plot", return_value=mock_plotly_figure) as mock_plot,
        patch.object(WindowedMetricPlotter, "add_model") as mock_add_model,
    ):
        result = viz.create_by_target(reports)

    # Verify one time series per target
    assert mock_add_model.call_count == 3

    # Verify each model uses correct target name and same time series data
    expected_calls = [
        call(
            model_name="Target1",
            timestamps=[datetime.fromisoformat("2023-01-01T01:00:00"), datetime.fromisoformat("2023-01-01T02:00:00")],
            metric_values=[0.4, 0.6],
        ),
        call(
            model_name="Target2",
            timestamps=[datetime.fromisoformat("2023-01-01T01:00:00"), datetime.fromisoformat("2023-01-01T02:00:00")],
            metric_values=[0.4, 0.6],
        ),
        call(
            model_name="Target3",
            timestamps=[datetime.fromisoformat("2023-01-01T01:00:00"), datetime.fromisoformat("2023-01-01T02:00:00")],
            metric_values=[0.4, 0.6],
        ),
    ]
    mock_add_model.assert_has_calls(expected_calls)

    # Verify plot title includes run context
    expected_title = "Windowed mae (lag=PT1H,size=PT2H,stride=PT1H) over Time by Target for Run1"
    mock_plot.assert_called_once_with(title=expected_title)

    # Verify output
    assert result.name == viz.name
    assert result.figure == mock_plotly_figure


def test_create_by_run_groups_time_series_by_run(
    multiple_target_metadata: list[TargetMetadata],
    sample_evaluation_report: EvaluationSubsetReport,
    sample_window: Window,
    mock_plotly_figure: MockFigure,
):
    """Test create_by_run groups targets by run and creates appropriate time series."""
    viz = WindowedMetricVisualization(name="test_viz", metric="mae", window=sample_window)
    reports = {
        "Run1": [(multiple_target_metadata[0], sample_evaluation_report)],
        "Run2": [(multiple_target_metadata[2], sample_evaluation_report)],
    }

    # Mock the plotter methods to capture calls and return our mock figure
    with (
        patch.object(WindowedMetricPlotter, "plot", return_value=mock_plotly_figure) as mock_plot,
        patch.object(WindowedMetricPlotter, "add_model") as mock_add_model,
    ):
        result = viz.create_by_run_and_none(reports)

    # Verify one time series per run
    assert mock_add_model.call_count == 2

    # Verify each model uses run name and correct time series
    expected_calls = [
        call(
            model_name="Run1",
            timestamps=[datetime.fromisoformat("2023-01-01T01:00:00"), datetime.fromisoformat("2023-01-01T02:00:00")],
            metric_values=[0.4, 0.6],
        ),
        call(
            model_name="Run2",
            timestamps=[datetime.fromisoformat("2023-01-01T01:00:00"), datetime.fromisoformat("2023-01-01T02:00:00")],
            metric_values=[0.4, 0.6],
        ),
    ]
    mock_add_model.assert_has_calls(expected_calls)

    # Verify plot title for run comparison
    expected_title = "Windowed mae (lag=PT1H,size=PT2H,stride=PT1H) over Time by Run"
    mock_plot.assert_called_once_with(title=expected_title)

    # Verify output
    assert result.name == viz.name
    assert result.figure == mock_plotly_figure


def test_supported_aggregations_includes_time_series_types(sample_window: Window):
    """Test that supported aggregations include appropriate types for time series visualizations."""
    viz = WindowedMetricVisualization(name="test_viz", metric="mae", window=sample_window)
    expected = {
        AnalysisAggregation.NONE,
        AnalysisAggregation.RUN_AND_NONE,
        AnalysisAggregation.TARGET,
        AnalysisAggregation.RUN_AND_TARGET,
        AnalysisAggregation.GROUP,
        AnalysisAggregation.RUN_AND_GROUP,
    }
    assert viz.supported_aggregations == expected


def test_average_time_series_across_targets_averages_correctly(
    multiple_target_metadata: list[TargetMetadata],
    sample_evaluation_report: EvaluationSubsetReport,
    sample_window: Window,
):
    # Arrange
    # Change the windowed metric values in sample_evaluation_report_2 to simulate different targets
    sample_evaluation_report_2 = copy.deepcopy(sample_evaluation_report)
    for metric in sample_evaluation_report_2.metrics:
        if metric.window == sample_window:
            metric.metrics["global"]["mae"] += 1.0  # add 1.0 to each value

    reports = [
        (multiple_target_metadata[0], sample_evaluation_report),
        (multiple_target_metadata[1], sample_evaluation_report_2),
    ]

    # Act
    viz = WindowedMetricVisualization(name="test_viz", metric="mae", window=sample_window)

    averaged = viz._average_time_series_across_targets(reports, metric_name="mae", quantile_or_global="global")

    # Assert
    # The expected average is mean of [0.4, 1.4] = 0.9 and [0.6, 1.6] = 1.1
    assert averaged == [
        (datetime.fromisoformat("2023-01-01T01:00:00"), pytest.approx(0.9)),
        (datetime.fromisoformat("2023-01-01T02:00:00"), pytest.approx(1.1)),
    ]


def test_average_time_series_across_targets_averages_correctly_with_nan(
    multiple_target_metadata: list[TargetMetadata],
    sample_evaluation_report: EvaluationSubsetReport,
    sample_window: Window,
):
    # Arrange
    # Change the windowed metric values in sample_evaluation_report_2 to simulate different targets
    sample_evaluation_report_2 = copy.deepcopy(sample_evaluation_report)
    for metric in sample_evaluation_report_2.metrics:
        if metric.window == sample_window:
            metric.metrics["global"]["mae"] = np.nan  # set to NaN

    reports = [
        (multiple_target_metadata[0], sample_evaluation_report),
        (multiple_target_metadata[1], sample_evaluation_report_2),
    ]

    # Act
    viz = WindowedMetricVisualization(name="test_viz", metric="mae", window=sample_window)

    averaged = viz._average_time_series_across_targets(reports, metric_name="mae", quantile_or_global="global")

    # Assert
    # The expected average is mean of [0.4, np.nan] = 0.4 and [0.6, np.nan] = 0.6
    assert averaged == [
        (datetime.fromisoformat("2023-01-01T01:00:00"), 0.4),
        (datetime.fromisoformat("2023-01-01T02:00:00"), 0.6),
    ]


def test_create_by_run_and_target_handles_multiple_runs_and_targets(
    sample_window: Window,
    multiple_target_metadata: list[TargetMetadata],
    sample_evaluation_report: EvaluationSubsetReport,
    mock_plotly_figure: MockFigure,
):
    # Arrange
    """Test create_by_run_and_target handles multiple runs with multiple targets each."""
    viz = WindowedMetricVisualization(name="test_viz", metric="mae", window=sample_window)

    # Create a second report with modified metrics for the second target in Run1
    sample_evaluation_report_2 = copy.deepcopy(sample_evaluation_report)
    for metric in sample_evaluation_report_2.metrics:
        if metric.window == sample_window:
            metric.metrics["global"]["mae"] += 1.0  # add 1.0 to each value

    # Run1: two targets, Run2: one target
    reports = {
        "Run1": [
            (multiple_target_metadata[0], sample_evaluation_report),
            (multiple_target_metadata[1], sample_evaluation_report_2),
        ],
        "Run2": [
            (multiple_target_metadata[2], sample_evaluation_report),
        ],
    }

    # Act
    with (
        patch.object(WindowedMetricPlotter, "plot", return_value=mock_plotly_figure) as mock_plot,
        patch.object(WindowedMetricPlotter, "add_model") as mock_add_model,
    ):
        result = viz.create_by_run_and_target(reports)

    # Assert
    expected_calls = [
        call(
            model_name="Run1",
            timestamps=[datetime.fromisoformat("2023-01-01T01:00:00"), datetime.fromisoformat("2023-01-01T02:00:00")],
            metric_values=[pytest.approx(0.9), pytest.approx(1.1)],
        ),
        call(
            model_name="Run2",
            timestamps=[datetime.fromisoformat("2023-01-01T01:00:00"), datetime.fromisoformat("2023-01-01T02:00:00")],
            metric_values=[pytest.approx(0.4), pytest.approx(0.6)],
        ),
    ]
    mock_add_model.assert_has_calls(expected_calls)
    mock_plot.assert_called_once_with(
        title="Windowed mae (lag=PT1H,size=PT2H,stride=PT1H) over Time by run (averaged over targets in group)",
        metric_name="mae",
    )
    assert result.name == viz.name
    assert result.figure == mock_plotly_figure


def test_create_by_run_and_group_handles_multiple_runs_and_groups(
    sample_window: Window,
    multiple_target_metadata: list[TargetMetadata],
    sample_evaluation_report: EvaluationSubsetReport,
    mock_plotly_figure: MockFigure,
):
    """Test create_by_run_and_group aggregates by both run and group, averaging over all targets for each run."""
    viz = WindowedMetricVisualization(name="test_viz", metric="mae", window=sample_window)

    # Create a second report with modified metrics for the second target in Run1
    sample_evaluation_report_2 = copy.deepcopy(sample_evaluation_report)
    for metric in sample_evaluation_report_2.metrics:
        if metric.window == sample_window:
            metric.metrics["global"]["mae"] += 1.0

    # Run1 appears in two groups, Run2 in one group
    # This tests that aggregation is done over all targets for each run, not per group
    reports = {
        ("Run1", "GroupA"): [
            (multiple_target_metadata[0], sample_evaluation_report),
        ],
        ("Run1", "GroupB"): [
            (multiple_target_metadata[1], sample_evaluation_report_2),
        ],
        ("Run2", "GroupB"): [
            (multiple_target_metadata[2], sample_evaluation_report),
        ],
    }

    # Act
    with (
        patch.object(WindowedMetricPlotter, "plot", return_value=mock_plotly_figure) as mock_plot,
        patch.object(WindowedMetricPlotter, "add_model") as mock_add_model,
    ):
        result = viz.create_by_run_and_group(reports)

    # Assert: only one call for Run1, averaging over both groups
    expected_calls = [
        call(
            model_name="Run1",
            timestamps=[datetime.fromisoformat("2023-01-01T01:00:00"), datetime.fromisoformat("2023-01-01T02:00:00")],
            metric_values=[pytest.approx(0.9), pytest.approx(1.1)],
        ),
        call(
            model_name="Run2",
            timestamps=[datetime.fromisoformat("2023-01-01T01:00:00"), datetime.fromisoformat("2023-01-01T02:00:00")],
            metric_values=[pytest.approx(0.4), pytest.approx(0.6)],
        ),
    ]
    mock_add_model.assert_has_calls(expected_calls)
    # Assert only one call per run, not per run-group
    assert mock_add_model.call_count == 2
    mock_plot.assert_called_once_with(
        title="Windowed mae (lag=PT1H,size=PT2H,stride=PT1H) over Time by run (averaged over all targets)",
        metric_name="mae",
    )
    assert result.name == viz.name
    assert result.figure == mock_plotly_figure


def test_create_by_group_handles_multiple_groups(
    sample_window: Window,
    multiple_target_metadata: list[TargetMetadata],
    sample_evaluation_report: EvaluationSubsetReport,
    mock_plotly_figure: MockFigure,
):
    """Test create_by_run_and_group aggregates by both run and group, averaging over all targets for each run."""
    viz = WindowedMetricVisualization(name="test_viz", metric="mae", window=sample_window)

    # Create a second report with modified metrics for the second target in Run1
    sample_evaluation_report_2 = copy.deepcopy(sample_evaluation_report)
    for metric in sample_evaluation_report_2.metrics:
        if metric.window == sample_window:
            metric.metrics["global"]["mae"] += 1.0

    # Run1 appears in two groups
    reports = {
        "GroupA": [
            (multiple_target_metadata[0], sample_evaluation_report),
        ],
        "GroupB": [
            (multiple_target_metadata[1], sample_evaluation_report_2),
        ],
    }

    # Act
    with (
        patch.object(WindowedMetricPlotter, "plot", return_value=mock_plotly_figure) as mock_plot,
        patch.object(WindowedMetricPlotter, "add_model") as mock_add_model,
    ):
        result = viz.create_by_group(reports)

    # Assert: only one call for Run1, averaging over both groups
    expected_calls = [
        call(
            model_name="Run1",
            timestamps=[datetime.fromisoformat("2023-01-01T01:00:00"), datetime.fromisoformat("2023-01-01T02:00:00")],
            metric_values=[pytest.approx(0.9), pytest.approx(1.1)],
        ),
    ]
    mock_add_model.assert_has_calls(expected_calls)
    # Assert only one call per run, not per run-group
    assert mock_add_model.call_count == 1
    mock_plot.assert_called_once_with(
        title="Windowed mae (lag=PT1H,size=PT2H,stride=PT1H) over Time averaged over all targets", metric_name="mae"
    )
    assert result.name == viz.name
    assert result.figure == mock_plotly_figure
