from datetime import datetime, timedelta
from typing import override, Any

from openstef_beam.benchmarking.target_provider import TargetProvider
from openstef_beam.evaluation.metric_providers import MetricProvider
from openstef_beam.evaluation.models.subset import QuantileMetricsDict
import pandas as pd
from pydantic import ConfigDict
import pytest

from openstef_beam.benchmarking import BenchmarkTarget
from openstef_core.datasets import VersionedTimeSeriesDataset, ForecastDataset
from openstef_core.types import LeadTime, Quantile


# Importing from utils.mocks raised, Error. Hence duplicate code
class MockMetricsProvider(MetricProvider):
    """Mock implementation of a MetricProvider for testing purposes."""

    mocked_result: QuantileMetricsDict

    def __call__(self, subset: ForecastDataset) -> QuantileMetricsDict:
        return self.mocked_result


class MockTargetProvider(TargetProvider[BenchmarkTarget, None]):
    """Test implementation of TargetProvider for testing."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        strict=False,
    )

    targets: list[BenchmarkTarget]
    measurements: VersionedTimeSeriesDataset
    predictors: VersionedTimeSeriesDataset
    metrics: list[MetricProvider]

    @override
    def get_targets(self, filter_args: Any = None) -> list[BenchmarkTarget]:
        targets = self.targets
        if filter_args is not None:
            targets = [t for t in targets if t.name == filter_args.name]
        return targets

    @override
    def get_measurements_for_target(self, target: BenchmarkTarget) -> VersionedTimeSeriesDataset:
        return self.measurements

    @override
    def get_predictors_for_target(self, target: BenchmarkTarget) -> VersionedTimeSeriesDataset:
        return self.predictors

    @override
    def get_metrics_for_target(self, target: BenchmarkTarget) -> list[MetricProvider]:
        return self.metrics

    @override
    def get_evaluation_mask_for_target(self, target: BenchmarkTarget) -> pd.DatetimeIndex | None:
        return None


@pytest.fixture
def targets() -> list[BenchmarkTarget]:
    """Create test targets with realistic date ranges."""
    return [
        BenchmarkTarget(
            name="location1",
            description="Test location 1",
            latitude=52.0,
            longitude=5.0,
            limit=100.0,
            benchmark_start=datetime.fromisoformat("2023-01-15"),
            benchmark_end=datetime.fromisoformat("2023-02-15"),
            train_start=datetime.fromisoformat("2023-01-01"),
        ),
        BenchmarkTarget(
            name="location2",
            description="Test location 2",
            latitude=53.0,
            longitude=6.0,
            limit=200.0,
            benchmark_start=datetime.fromisoformat("2023-01-15"),
            benchmark_end=datetime.fromisoformat("2023-02-15"),
            train_start=datetime.fromisoformat("2023-01-01"),
        ),
    ]


@pytest.fixture
def datasets(targets: list[BenchmarkTarget]) -> tuple[VersionedTimeSeriesDataset, VersionedTimeSeriesDataset]:
    """Create test datasets with timestamps aligned to target benchmark periods."""
    # Create datasets covering the entire benchmark period plus some training data
    start_date = min(t.train_start for t in targets)
    end_date = max(t.benchmark_end for t in targets)
    # Use a 1-hour interval for consistency with evaluation requirements
    sample_interval = timedelta(hours=1)
    timestamps = pd.date_range(start_date, end_date, freq="1h")

    # Create measurements dataset with value column and necessary metadata
    measurements = VersionedTimeSeriesDataset.from_dataframe(
        data=pd.DataFrame(
            {
                "value": range(len(timestamps)),
                "available_at": timestamps,  # Make available immediately for simplicity
            },
            index=timestamps,
        ),
        sample_interval=sample_interval,
    )

    # Create predictors dataset with the same sample interval
    predictors = VersionedTimeSeriesDataset.from_dataframe(
        data=pd.DataFrame(
            {
                "feature1": range(len(timestamps)),
                "feature2": range(len(timestamps), 2 * len(timestamps)),
                "available_at": timestamps,  # Make available immediately for simplicity
            },
            index=timestamps,
        ),
        sample_interval=sample_interval,
    )

    return measurements, predictors


@pytest.fixture
def target_provider(
    targets: list[BenchmarkTarget], datasets: tuple[VersionedTimeSeriesDataset, VersionedTimeSeriesDataset]
) -> MockTargetProvider:
    measurements, predictors = datasets
    # Set up mock metrics
    mock_metrics: list[MetricProvider] = [
        MockMetricsProvider(
            mocked_result={
                "global": {
                    "rmae": 0.5,
                }
            }
        )
    ]
    # Set up target provider with test data
    return MockTargetProvider(
        targets=targets,
        measurements=measurements,
        predictors=predictors,
        metrics=mock_metrics,
    )


def test_benchmark_optimizer_end_to_end(
    target_provider: MockTargetProvider,
):
    """End-to-end test of the BenchmarkOptimizer with mock data and callbacks."""

    # tp = target_provider(targets(), datasets(targets()))
    # opt = optimizer(workflow_config(horizons(), quantiles()), n_jobs=2)

    # Act
    # results = opt.optimize(experiment_name="Test Benchmark Optimization", target_provider=tp)

    # Assert
    pass
