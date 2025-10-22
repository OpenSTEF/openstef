# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Shared fixtures for visualization tests."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from openstef_beam.analysis.models import TargetMetadata
from openstef_beam.evaluation import EvaluationSubsetReport, SubsetMetric, Window
from openstef_core.datasets import ForecastDataset
from openstef_core.types import LeadTime, Quantile
from tests.utils.mocks import MockFigure


@pytest.fixture
def simple_target_metadata() -> TargetMetadata:
    """Create simple target metadata for testing."""
    return TargetMetadata(
        name="TestTarget",
        run_name="TestRun",
        group_name="TestGroup",
        filtering=None,
        limit=100.0,
    )


@pytest.fixture
def multiple_target_metadata() -> list[TargetMetadata]:
    """Create multiple target metadata for testing aggregations."""
    return [
        TargetMetadata(name="Target1", run_name="Run1", group_name="GroupA", filtering=None, limit=100.0),
        TargetMetadata(name="Target2", run_name="Run1", group_name="GroupA", filtering=None, limit=200.0),
        TargetMetadata(name="Target3", run_name="Run2", group_name="GroupB", filtering=None, limit=150.0),
    ]


@pytest.fixture
def sample_evaluation_subset() -> ForecastDataset:
    return ForecastDataset(
        data=pd.DataFrame(
            {"quantile_P10": [1.0, 2.0], "quantile_P50": [2.0, 3.0], "quantile_P90": [3.0, 4.0], "load": [1.5, 2.5]},
            index=pd.to_datetime([
                datetime.fromisoformat("2023-01-01T00:00:00"),
                datetime.fromisoformat("2023-01-01T01:00:00"),
            ]),
        ),
        sample_interval=timedelta(hours=1),
    )


@pytest.fixture
def sample_subset_metrics() -> list[SubsetMetric]:
    """Create sample subset metrics for testing."""
    return [
        # Global metrics
        SubsetMetric(
            window="global",
            timestamp=datetime.fromisoformat("2023-01-01T00:00:00"),
            metrics={
                "global": {"mae": 0.5, "rmse": 0.7},
                Quantile(0.1): {"precision": 0.8, "recall": 0.6},
                Quantile(0.5): {"precision": 0.9, "recall": 0.7},
                Quantile(0.9): {"precision": 0.7, "recall": 0.8},
            },
        ),
        # Windowed metrics
        SubsetMetric(
            window=Window(lag=timedelta(hours=1), size=timedelta(hours=2), stride=timedelta(hours=1)),
            timestamp=datetime.fromisoformat("2023-01-01T01:00:00"),
            metrics={
                "global": {"mae": 0.4},
                Quantile(0.5): {"precision": 0.85},
            },
        ),
        SubsetMetric(
            window=Window(lag=timedelta(hours=1), size=timedelta(hours=2), stride=timedelta(hours=1)),
            timestamp=datetime.fromisoformat("2023-01-01T02:00:00"),
            metrics={
                "global": {"mae": 0.6},
                Quantile(0.5): {"precision": 0.75},
            },
        ),
    ]


@pytest.fixture
def sample_subset_metrics_with_effective() -> list[SubsetMetric]:
    """Create sample subset metrics with effective metrics for testing selector functionality."""
    return [
        # Global metrics with effective metrics
        SubsetMetric(
            window="global",
            timestamp=datetime.fromisoformat("2023-01-01T00:00:00"),
            metrics={
                "global": {"mae": 0.5, "rmse": 0.7},
                Quantile(0.1): {
                    "precision": 0.8,
                    "recall": 0.6,
                    "effective_precision": 0.75,
                    "effective_recall": 0.55,
                    "F2": 0.65,
                    "effective_F2": 0.61,
                },
                Quantile(0.5): {
                    "precision": 0.9,
                    "recall": 0.7,
                    "effective_precision": 0.85,
                    "effective_recall": 0.65,
                    "F2": 0.75,
                    "effective_F2": 0.71,
                },
                Quantile(0.9): {
                    "precision": 0.7,
                    "recall": 0.8,
                    "effective_precision": 0.65,
                    "effective_recall": 0.75,
                    "F2": 0.77,
                    "effective_F2": 0.71,
                },
            },
        ),
    ]


@pytest.fixture
def sample_evaluation_report(
    sample_evaluation_subset: ForecastDataset, sample_subset_metrics: list[SubsetMetric]
) -> EvaluationSubsetReport:
    """Create a complete evaluation subset report for testing."""
    return EvaluationSubsetReport(
        filtering=LeadTime.from_string("PT1H"),
        subset=sample_evaluation_subset,
        metrics=sample_subset_metrics,
    )


@pytest.fixture
def sample_evaluation_report_with_effective_metrics(
    sample_evaluation_subset: ForecastDataset, sample_subset_metrics_with_effective: list[SubsetMetric]
) -> EvaluationSubsetReport:
    """Create a complete evaluation subset report with effective metrics for testing selector functionality."""
    return EvaluationSubsetReport(
        filtering=LeadTime.from_string("PT1H"),
        subset=sample_evaluation_subset,
        metrics=sample_subset_metrics_with_effective,
    )


@pytest.fixture
def empty_evaluation_report() -> EvaluationSubsetReport:
    """Create an empty evaluation report for testing error cases."""
    empty_subset = ForecastDataset(
        data=pd.DataFrame(index=pd.DatetimeIndex([]), columns=["load"]),
        sample_interval=timedelta(hours=1),
    )
    return EvaluationSubsetReport(filtering=LeadTime.from_string("PT1H"), subset=empty_subset, metrics=[])


@pytest.fixture
def sample_window() -> Window:
    """Create a sample window for windowed metric tests."""
    return Window(lag=timedelta(hours=1), size=timedelta(hours=2), stride=timedelta(hours=1))


@pytest.fixture
def mock_plotly_figure() -> MockFigure:
    """Create a mock plotly figure for testing plot outputs."""
    return MockFigure()


@pytest.fixture
def sample_evaluation_report_with_probabilities(sample_evaluation_subset: ForecastDataset) -> EvaluationSubsetReport:
    """Create evaluation report containing probability metrics for quantile visualization testing."""
    subset_metrics_with_probs = [
        SubsetMetric(
            window="global",
            timestamp=datetime.fromisoformat("2023-01-01T00:00:00"),
            metrics={
                "global": {"mae": 0.5, "rmse": 0.7},
                Quantile(0.1): {"observed_probability": 0.15},
                Quantile(0.5): {"observed_probability": 0.55},
            },
        ),
    ]

    return EvaluationSubsetReport(
        filtering=LeadTime.from_string("PT1H"),
        subset=sample_evaluation_subset,
        metrics=subset_metrics_with_probs,
    )
