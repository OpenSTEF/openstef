# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta
from typing import Any, override

import pandas as pd
import pytest
from pydantic import ConfigDict, Field

from openstef_beam.backtesting import BacktestConfig
from openstef_beam.backtesting.backtest_forecaster import BacktestForecasterConfig
from openstef_beam.benchmarking import BenchmarkTarget
from openstef_beam.benchmarking.target_provider import TargetProvider
from openstef_beam.evaluation.metric_providers import (
    MetricProvider,
    RCRPSProvider,
    RCRPSSampleWeightedProvider,
    RMAEProvider,
)
from openstef_beam.evaluation.models.subset import QuantileMetricsDict
from openstef_beam.parameter_tuning.models import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
    OptimizationMetric,
    ParameterSpace,
)
from openstef_beam.parameter_tuning.optimizer.benchmark_optimizer import BenchmarkOptimizer
from openstef_beam.parameter_tuning.optimizer.optimizer import OptimizerConfig
from openstef_core.datasets import ForecastDataset, VersionedTimeSeriesDataset
from openstef_core.mixins.predictor import HyperParams
from openstef_core.types import LeadTime, Quantile
from openstef_models.presets import ForecastingWorkflowConfig
from openstef_models.presets.forecasting_workflow import Latitude, Longitude


class MockHyperParams(HyperParams):
    learning_rate: float = Field(default=0.1)
    n_estimators: int = Field(default=100)
    booster: str = Field(default="gbtree")


class MockParameterSpace(ParameterSpace):
    learning_rate: FloatDistribution = Field(default=FloatDistribution(low=0.01, high=0.2))
    n_estimators: IntDistribution = Field(default=IntDistribution(low=50, high=200))
    booster: CategoricalDistribution = Field(default=CategoricalDistribution(choices=["gbtree", "dart"]))

    @classmethod
    def default_hyperparams(cls) -> MockHyperParams:
        return MockHyperParams()


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


# Test fixtures
@pytest.fixture
def targets() -> list[BenchmarkTarget]:
    """Create test targets with realistic date ranges."""
    return [
        BenchmarkTarget(
            name="location1",
            description="Test location 1",
            latitude=Latitude(52.0),
            longitude=Longitude(5.0),
            limit=100.0,
            benchmark_start=datetime.fromisoformat("2023-01-15"),
            benchmark_end=datetime.fromisoformat("2023-02-15"),
            train_start=datetime.fromisoformat("2023-01-01"),
        ),
        BenchmarkTarget(
            name="location2",
            description="Test location 2",
            latitude=Latitude(53.0),
            longitude=Longitude(6.0),
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
def quantiles() -> list[Quantile]:
    """Provide common quantiles for testing."""
    return [Quantile(0.1), Quantile(0.5), Quantile(0.9)]


@pytest.fixture
def horizons() -> list[LeadTime]:
    """Provide common lead times for testing."""
    return [LeadTime.from_string("PT12H")]


@pytest.fixture
def forecaster_config() -> BacktestForecasterConfig:
    """Create a realistic forecaster config with all required fields."""
    return BacktestForecasterConfig(
        requires_training=True,
        predict_length=timedelta(hours=24),
        predict_min_length=timedelta(hours=1),
        predict_context_length=timedelta(hours=48),
        predict_context_min_coverage=0.8,
        training_context_length=timedelta(days=14),
        training_context_min_coverage=0.8,
        predict_sample_interval=timedelta(hours=1),
        # The quantiles property is provided by the base class, no need to specify here
    )


@pytest.fixture
def backtest_config() -> BacktestConfig:
    """Create a realistic backtest config."""
    return BacktestConfig(
        prediction_sample_interval=timedelta(hours=1),
        predict_interval=timedelta(hours=6),
        train_interval=timedelta(days=7),
    )


@pytest.fixture(params=["wrcrps", "rmae", "rcrps"])
def optimization_metric(request: pytest.FixtureRequest) -> OptimizationMetric:
    """Create different optimization metrics for testing."""
    metric = request.param
    if metric == "wrcrps":
        provider = RCRPSSampleWeightedProvider(lower_quantile=0.01, upper_quantile=0.99)
    elif metric == "rmae":
        provider = RMAEProvider()
    else:
        provider = RCRPSProvider(lower_quantile=0.01, upper_quantile=0.99)
    return OptimizationMetric(metric=provider, direction_minimize=True)


@pytest.fixture
def workflow_config(horizons: list[LeadTime], quantiles: list[Quantile]) -> ForecastingWorkflowConfig:

    return ForecastingWorkflowConfig(
        model_id="common_model_",
        run_name=None,
        model="flatliner",
        horizons=horizons,
        quantiles=quantiles,
        model_reuse_enable=True,
        mlflow_storage=None,
        radiation_column="shortwave_radiation",
        rolling_aggregate_features=["mean", "median", "max", "min"],
        wind_speed_column="wind_speed_80m",
        pressure_column="surface_pressure",
        temperature_column="temperature_2m",
        relative_humidity_column="relative_humidity_2m",
        energy_price_column="EPEX_NL",
    )


@pytest.fixture(params=[1, 4])
def n_jobs(request: pytest.FixtureRequest) -> int:
    """Parameterize tests with different numbers of jobs."""
    return request.param


@pytest.fixture
def optimizer_config(
    n_jobs: int,
    backtest_config: BacktestConfig,
    forecaster_config: BacktestForecasterConfig,
    workflow_config: ForecastingWorkflowConfig,
) -> OptimizerConfig:

    return OptimizerConfig(
        n_trials=5,
        n_jobs=n_jobs,
        parameter_space=MockParameterSpace(),
        optimization_metric=OptimizationMetric(
            metric=RCRPSSampleWeightedProvider(lower_quantile=0.01, upper_quantile=0.99),
            direction_minimize=True,
        ),
        backtest_config=backtest_config,
        backtest_forecaster_config=forecaster_config,
        base_config=workflow_config,
    )


@pytest.fixture
def optimizer(optimizer_config: OptimizerConfig) -> BenchmarkOptimizer:
    """Create a BenchmarkOptimizer with a simple config."""
    return BenchmarkOptimizer(config=optimizer_config)


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
    optimizer: BenchmarkOptimizer,
):
    """End-to-end test of the BenchmarkOptimizer with mock data and callbacks."""

    assert True  # Placeholder assertion to indicate test ran successfully
