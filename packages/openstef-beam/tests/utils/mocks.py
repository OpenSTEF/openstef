# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from typing import Any, cast, override

import pandas as pd
from pydantic import ConfigDict

from openstef_beam.backtesting.backtest_forecaster import BacktestForecasterConfig, BacktestForecasterMixin
from openstef_beam.benchmarking import BenchmarkTarget, TargetProvider
from openstef_beam.evaluation.metric_providers import MetricProvider, QuantileMetricsDict
from openstef_beam.evaluation.models import EvaluationSubset
from openstef_core.datasets import TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.datasets.versioned_timeseries.accessors import RestrictedHorizonVersionedTimeSeries
from openstef_core.types import Q, Quantile


class DummyMetricProvider(MetricProvider):
    """Returns a constant metric for every call."""

    value: float = 1.0

    def __call__(self, subset: EvaluationSubset) -> QuantileMetricsDict:
        # Return metrics for global since test is not using actual quantile data
        return cast(QuantileMetricsDict, {"global": {"dummy_metric": self.value}})


class MockMetricsProvider(MetricProvider):
    """Mock implementation of a MetricProvider for testing purposes."""

    mocked_result: QuantileMetricsDict

    def __call__(self, subset: EvaluationSubset) -> QuantileMetricsDict:
        return self.mocked_result


class MockFigure:
    def __init__(self):
        self.data = []
        self.layout = type("Layout", (), {"title": type("Title", (), {"text": "Test Plot"})})()


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


class MockForecaster(BacktestForecasterMixin):
    """Simplified mock implementation of ModelInterface."""

    def __init__(
        self,
        config: BacktestForecasterConfig,
        quantiles: list[Quantile] | None = None,
    ):
        self.config = config
        self._quantiles = quantiles or [Q(0.5)]

    @property
    def quantiles(self) -> list[Quantile]:
        return self._quantiles

    @override
    def fit(self, data: RestrictedHorizonVersionedTimeSeries) -> None:
        """Mock implementation to satisfy the abstract interface."""

    @override
    def predict(self, data: RestrictedHorizonVersionedTimeSeries) -> TimeSeriesDataset:
        """Mock implementation that returns a simple dataframe with the required quantile columns."""
        # Create a simple prediction dataframe with required quantile columns
        return TimeSeriesDataset(
            data=pd.DataFrame(
                {f"quantile_P{int(q * 100)}": [50.0] for q in self.quantiles},
                index=pd.DatetimeIndex([data.horizon]),
            ),
            sample_interval=data.sample_interval,
        )
