# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for Analysis pipeline module."""

from datetime import timedelta
from typing import Any

import pandas as pd
import pytest
from pydantic import Field

from openstef_beam.analysis.analysis_pipeline import AnalysisConfig, AnalysisPipeline
from openstef_beam.analysis.models import AnalysisAggregation, AnalysisScope, TargetMetadata, VisualizationOutput
from openstef_beam.analysis.visualizations import VisualizationProvider
from openstef_beam.evaluation import EvaluationReport, EvaluationSubsetReport
from openstef_core.datasets import ForecastDataset
from openstef_core.types import LeadTime


class MockVisualizationProvider(VisualizationProvider):
    """Mock implementation of VisualizationProvider for testing."""

    create_calls: list[Any] = Field(default_factory=list)

    def __init__(self, name: str, supported_aggs: set[AnalysisAggregation]):
        super().__init__(name=name)
        self._supported_aggregations = supported_aggs
        # Store create calls without using Pydantic field assignment
        object.__setattr__(self, "create_calls", [])

    @property
    def supported_aggregations(self) -> set[AnalysisAggregation]:
        return self._supported_aggregations

    def create(
        self, reports: list[tuple[TargetMetadata, EvaluationSubsetReport]], aggregation: AnalysisAggregation
    ) -> VisualizationOutput:
        self.create_calls.append((reports, aggregation))
        return VisualizationOutput(
            name=f"{self.name}_{aggregation.value}.html",
            html=f"<html>Mock visualization for {aggregation.value}</html>",
        )


def create_empty_evaluation_subset() -> ForecastDataset:
    """Create an empty EvaluationSubset for testing."""
    return ForecastDataset(
        data=pd.DataFrame(
            {"quantile_P50": [0.0], "load": [0.0]}, index=pd.date_range("2020-01-01", periods=1, freq="h")
        ),
        sample_interval=timedelta(hours=1),
    )


@pytest.fixture
def target_metadata() -> TargetMetadata:
    """Create basic target metadata for testing."""
    return TargetMetadata(
        name="test_target",
        group_name="test_group",
        filtering=LeadTime.from_string("PT1H"),
        limit=100.0,
        run_name="test_run",
    )


@pytest.fixture
def evaluation_subset_report() -> EvaluationSubsetReport:
    """Create minimal evaluation subset report for testing."""
    return EvaluationSubsetReport(
        filtering=LeadTime.from_string("PT1H"), subset=create_empty_evaluation_subset(), metrics=[]
    )


@pytest.fixture
def evaluation_report(evaluation_subset_report: EvaluationSubsetReport) -> EvaluationReport:
    """Create evaluation report with multiple subsets."""
    subset2 = EvaluationSubsetReport(
        filtering=LeadTime.from_string("PT2H"), subset=create_empty_evaluation_subset(), metrics=[]
    )
    return EvaluationReport(subset_reports=[evaluation_subset_report, subset2])


def test_run_filters_unsupported_aggregations(
    target_metadata: TargetMetadata, evaluation_subset_report: EvaluationSubsetReport
):
    # Arrange
    provider1 = MockVisualizationProvider("provider1", {AnalysisAggregation.NONE})
    provider2 = MockVisualizationProvider("provider2", {AnalysisAggregation.TARGET})
    config = AnalysisConfig(visualization_providers=[provider1, provider2])
    pipeline = AnalysisPipeline(config=config)
    reports = [(target_metadata, evaluation_subset_report)]

    # Act
    result = pipeline.run_for_subsets(reports, AnalysisAggregation.NONE)

    # Assert
    assert len(result) == 1
    assert len(provider1.create_calls) == 1
    assert len(provider2.create_calls) == 0


@pytest.mark.parametrize(
    ("aggregation", "expected_calls"),
    [
        pytest.param(AnalysisAggregation.NONE, 1, id="none_aggregation"),
        pytest.param(AnalysisAggregation.TARGET, 1, id="target_aggregation"),
        pytest.param(AnalysisAggregation.GROUP, 1, id="group_aggregation"),
    ],
)
def test_run_calls_supported_providers(
    target_metadata: TargetMetadata,
    evaluation_subset_report: EvaluationSubsetReport,
    aggregation: AnalysisAggregation,
    expected_calls: int,
):
    # Arrange
    provider = MockVisualizationProvider("test_provider", {aggregation})
    config = AnalysisConfig(visualization_providers=[provider])
    pipeline = AnalysisPipeline(config=config)
    reports = [(target_metadata, evaluation_subset_report)]

    # Act
    result = pipeline.run_for_subsets(reports, aggregation)

    # Assert
    assert len(result) == 1
    assert len(provider.create_calls) == expected_calls
    assert provider.create_calls[0][1] == aggregation


def test_run_for_target_creates_visualizations_per_subset(
    target_metadata: TargetMetadata, evaluation_report: EvaluationReport
):
    # Arrange
    provider = MockVisualizationProvider("test_provider", {AnalysisAggregation.NONE})
    config = AnalysisConfig(visualization_providers=[provider])
    pipeline = AnalysisPipeline(config=config)
    scope = AnalysisScope(
        aggregation=AnalysisAggregation.NONE,
        group_name=target_metadata.group_name,
        target_name=target_metadata.name,
        run_name=None,
    )

    # Act
    result = pipeline.run_for_reports(
        reports=[(target_metadata, evaluation_report)],
        scope=scope,
    )

    # Assert
    assert len(result.visualizations) == 2  # Two subsets in evaluation_report
    assert LeadTime.from_string("PT1H") in result.visualizations
    assert LeadTime.from_string("PT2H") in result.visualizations

    # Each subset should have one visualization
    assert len(result.visualizations[LeadTime.from_string("PT1H")]) == 1
    assert len(result.visualizations[LeadTime.from_string("PT2H")]) == 1

    # Provider should be called twice (once per subset)
    assert len(provider.create_calls) == 2


def test_run_for_target_updates_metadata_filtering(
    target_metadata: TargetMetadata, evaluation_report: EvaluationReport
):
    # Arrange
    provider = MockVisualizationProvider("test_provider", {AnalysisAggregation.NONE})
    config = AnalysisConfig(visualization_providers=[provider])
    pipeline = AnalysisPipeline(config=config)
    scope = AnalysisScope(
        aggregation=AnalysisAggregation.NONE,
        group_name=target_metadata.group_name,
        target_name=target_metadata.name,
        run_name=None,
    )

    # Act
    pipeline.run_for_reports(
        reports=[(target_metadata, evaluation_report)],
        scope=scope,
    )

    # Assert
    # Check that metadata filtering was updated for each call
    call1_metadata, _ = provider.create_calls[0][0][0]
    call2_metadata, _ = provider.create_calls[1][0][0]

    # Filtering should match the subset's filtering
    assert call1_metadata.filtering in {LeadTime.from_string("PT1H"), LeadTime.from_string("PT2H")}
    assert call2_metadata.filtering in {LeadTime.from_string("PT1H"), LeadTime.from_string("PT2H")}
    assert call1_metadata.filtering != call2_metadata.filtering


def test_run_for_targets_groups_by_filtering():
    # Arrange
    provider1 = MockVisualizationProvider("provider1", {AnalysisAggregation.TARGET})
    provider2 = MockVisualizationProvider("provider2", {AnalysisAggregation.TARGET})
    config = AnalysisConfig(visualization_providers=[provider1, provider2])
    pipeline = AnalysisPipeline(config=config)
    scope = AnalysisScope(
        aggregation=AnalysisAggregation.TARGET,
        group_name="group1",
        target_name="target1",
        run_name="run1",
    )

    # Create reports with different targets but same filtering
    metadata1 = TargetMetadata(
        name="target1", group_name="group1", filtering=LeadTime.from_string("PT1H"), limit=100.0, run_name="run1"
    )
    metadata2 = TargetMetadata(
        name="target2", group_name="group1", filtering=LeadTime.from_string("PT1H"), limit=200.0, run_name="run1"
    )

    subset_report = EvaluationSubsetReport(
        filtering=LeadTime.from_string("PT1H"), subset=create_empty_evaluation_subset(), metrics=[]
    )

    report1 = EvaluationReport(subset_reports=[subset_report])
    report2 = EvaluationReport(subset_reports=[subset_report])

    reports = [(metadata1, report1), (metadata2, report2)]

    # Act
    result = pipeline.run_for_reports(
        reports=reports,
        scope=scope,
    )

    # Assert
    assert len(result.visualizations) == 1  # One filtering group
    assert LeadTime.from_string("PT1H") in result.visualizations

    # Should have visualizations from both providers
    filtering_result = result.visualizations[LeadTime.from_string("PT1H")]
    assert len(filtering_result) == 2  # One from each provider

    # Both providers should be called once
    assert len(provider1.create_calls) == 1
    assert len(provider2.create_calls) == 1


def test_run_for_targets_handles_multiple_filtering_groups():
    # Arrange
    provider = MockVisualizationProvider("provider", {AnalysisAggregation.TARGET, AnalysisAggregation.GROUP})
    config = AnalysisConfig(visualization_providers=[provider])
    pipeline = AnalysisPipeline(config=config)

    # Create metadata with different filterings
    metadata1 = TargetMetadata(
        name="target1", group_name="group1", filtering=LeadTime.from_string("PT1H"), limit=100.0, run_name="run1"
    )
    metadata2 = TargetMetadata(
        name="target2", group_name="group1", filtering=LeadTime.from_string("PT2H"), limit=200.0, run_name="run1"
    )

    # Create reports with different subset filterings
    subset1 = EvaluationSubsetReport(
        filtering=LeadTime.from_string("PT1H"), subset=create_empty_evaluation_subset(), metrics=[]
    )
    subset2 = EvaluationSubsetReport(
        filtering=LeadTime.from_string("PT2H"), subset=create_empty_evaluation_subset(), metrics=[]
    )

    report1 = EvaluationReport(subset_reports=[subset1])
    report2 = EvaluationReport(subset_reports=[subset2])

    reports = [(metadata1, report1), (metadata2, report2)]
    scope = AnalysisScope(
        aggregation=AnalysisAggregation.TARGET,
        group_name="group1",
        target_name="target1",
        run_name="run1",
    )

    # Act
    result = pipeline.run_for_reports(
        reports=reports,
        scope=scope,
    )

    # Assert
    assert len(result.visualizations) == 2  # Two filtering groups
    assert LeadTime.from_string("PT1H") in result.visualizations
    assert LeadTime.from_string("PT2H") in result.visualizations

    # Each filtering group should have 1 visualizations (TARGET)
    assert len(result.visualizations[LeadTime.from_string("PT1H")]) == 1
    assert len(result.visualizations[LeadTime.from_string("PT2H")]) == 1

    # Provider should be called 4 times (2 filterings * 1 aggregations)
    assert len(provider.create_calls) == 2
