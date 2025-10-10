# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for SummaryTableVisualization."""

from unittest.mock import patch

import pytest

from openstef_beam.analysis.models.target_metadata import TargetMetadata
from openstef_beam.analysis.plots import SummaryTablePlotter
from openstef_beam.analysis.visualizations import SummaryTableVisualization
from openstef_beam.analysis.visualizations.summary_table_visualization import MetricAggregation
from openstef_beam.evaluation.models.report import EvaluationSubsetReport
from openstef_core.types import QuantileOrGlobal


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        pytest.param(
            [1.0, 2.0, 3.0],
            MetricAggregation(mean=2.0, min=1.0, max=3.0, median=2.0),
            id="normal_values",
        ),
        pytest.param(
            [5.0],
            MetricAggregation(mean=5.0, min=5.0, max=5.0, median=5.0),
            id="single_value",
        ),
        pytest.param(
            [],
            MetricAggregation(mean=0.0, min=0.0, max=0.0, median=0.0),
            id="empty_values",
        ),
    ],
)
def test_aggregate_metric_values_computes_statistics(values: list[float], expected: MetricAggregation):
    """Test metric aggregation computes correct statistical measures."""
    # Arrange
    viz = SummaryTableVisualization(name="test_viz")

    # Act
    result = viz._aggregate_metric_values(values)

    # Assert
    assert result == expected


@pytest.mark.parametrize(
    ("quantile", "expected"),
    [
        pytest.param(0.1, "0.1", id="quantile_value"),
        pytest.param(0.50, "0.5", id="quantile_trailing_zero"),
        pytest.param("global", "global", id="global_string"),
    ],
)
def test_format_quantile_handles_different_types(quantile: QuantileOrGlobal, expected: str):
    """Test quantile formatting handles different input types correctly."""
    # Arrange
    viz = SummaryTableVisualization(name="test_viz")

    # Act
    result = viz._format_quantile(quantile)

    # Assert
    assert result == expected


def test_extract_metrics_from_report_returns_structured_data(sample_evaluation_report: EvaluationSubsetReport):
    """Test extraction and formatting of metrics from evaluation report."""
    # Arrange
    viz = SummaryTableVisualization(name="test_viz")

    # Act
    rows = viz._extract_metrics_from_report(sample_evaluation_report)

    # Assert
    # Verify structure and content
    assert len(rows) > 0
    expected_structure = {"Quantile|Global", "Metric", "Value"}
    for row in rows:
        assert set(row.keys()) == expected_structure

    # Test quantile formatting within extracted data
    quantile_values = {row["Quantile|Global"] for row in rows}
    assert "global" in quantile_values  # Global should be preserved as string
    # Float quantiles should be formatted without trailing zeros
    assert any(qv in quantile_values for qv in ["0.1", "0.5", "0.9"])
    # Check specific values from conftest
    for row in rows:
        if row["Quantile|Global"] == "global" and row["Metric"] == "mae":
            assert row["Value"] == 0.5


def test_extract_metrics_from_report_returns_empty_for_no_global_metrics(
    empty_evaluation_report: EvaluationSubsetReport,
):
    """Test handling of reports with no global metrics."""
    # Arrange
    viz = SummaryTableVisualization(name="test_viz")

    # Act
    rows = viz._extract_metrics_from_report(empty_evaluation_report)

    # Assert
    assert rows == []


def test_create_sorted_dataframe_handles_empty_data_and_sorting():
    """Test DataFrame creation with empty data and sorting functionality."""
    # Arrange
    viz = SummaryTableVisualization(name="test_viz")

    # Act & Assert - Test empty data handling
    empty_df = viz._create_sorted_dataframe([], ["Col1", "Col2"], ["Col1"])
    assert empty_df.empty
    assert list(empty_df.columns) == ["Col1", "Col2"]

    # Test sorting functionality
    rows = [
        {"Metric": "rmse", "Value": 0.7},
        {"Metric": "mae", "Value": 0.5},
    ]
    sorted_df = viz._create_sorted_dataframe(rows, ["Metric", "Value"], ["Metric"])
    assert list(sorted_df["Metric"]) == ["mae", "rmse"]
    assert list(sorted_df["Value"]) == [0.5, 0.7]


@pytest.mark.parametrize(
    ("method_name", "reports_structure"),
    [
        pytest.param("create_by_none", "single_report_with_metadata", id="none_aggregation"),
        pytest.param("create_by_target", "list_of_report_tuples", id="target_aggregation"),
        pytest.param("create_by_run_and_none", "dict_of_run_reports", id="run_aggregation"),
        pytest.param("create_by_group", "dict_of_group_reports", id="group_aggregation"),
        pytest.param("create_by_run_and_group", "dict_of_run_group_reports", id="run_group_aggregation"),
        pytest.param("create_by_run_and_target", "dict_of_run_reports", id="run_target_aggregation"),
    ],
)
def test_visualization_creation_methods_produce_html_output(
    method_name: str,
    reports_structure: str,
    sample_evaluation_report: EvaluationSubsetReport,
    multiple_target_metadata: list[TargetMetadata],
):
    """Test all visualization creation methods produce expected HTML output."""
    # Arrange
    viz = SummaryTableVisualization(name="test_viz")

    # Prepare different report structures based on method requirements
    if reports_structure == "single_report_with_metadata":
        args = [sample_evaluation_report, multiple_target_metadata[0]]
    elif reports_structure == "list_of_report_tuples":
        args = [[(metadata, sample_evaluation_report) for metadata in multiple_target_metadata[:2]]]
    elif reports_structure == "dict_of_run_reports":
        args = [
            {
                "Run1": [(multiple_target_metadata[0], sample_evaluation_report)],
                "Run2": [(multiple_target_metadata[2], sample_evaluation_report)],
            }
        ]
    elif reports_structure == "dict_of_group_reports":
        args = [
            {
                "GroupA": [(multiple_target_metadata[0], sample_evaluation_report)],
                "GroupB": [(multiple_target_metadata[2], sample_evaluation_report)],
            }
        ]
    elif reports_structure == "dict_of_run_group_reports":
        args = [
            {
                ("Run1", "GroupA"): [(multiple_target_metadata[0], sample_evaluation_report)],
                ("Run2", "GroupB"): [(multiple_target_metadata[2], sample_evaluation_report)],
            }
        ]

    # Act
    with patch.object(SummaryTablePlotter, "plot", return_value="<table>Mock HTML</table>"):
        method = getattr(viz, method_name)
        result = method(*args)  # pyright: ignore[reportPossiblyUnboundVariable]

    # Assert - Verify consistent output structure
    assert result.name == viz.name
    assert result.html == "<table>Mock HTML</table>"
