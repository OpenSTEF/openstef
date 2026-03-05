# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openstef_beam.backtesting.restricted_horizon_timeseries import RestrictedHorizonVersionedTimeSeries
from openstef_beam.benchmarking.baselines.openstef4 import create_openstef4_preset_backtest_forecaster
from openstef_beam.benchmarking.benchmark_pipeline import BenchmarkContext, BenchmarkTarget
from openstef_core.datasets import TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.types import LeadTime, Q
from openstef_models.presets import ForecastingWorkflowConfig


@pytest.fixture
def xgboost_config() -> ForecastingWorkflowConfig:
    return ForecastingWorkflowConfig(
        model_id="test_xgb",
        model="xgboost",
        horizons=[LeadTime.from_string("PT24H")],
        quantiles=[Q(0.5)],
    )


@pytest.fixture
def benchmark_target() -> BenchmarkTarget:
    now = datetime(2024, 6, 1, tzinfo=UTC)
    return BenchmarkTarget(
        name="target_1",
        description="test",
        latitude=52.0,
        longitude=5.0,
        limit=100.0,
        benchmark_start=now - timedelta(days=365),
        benchmark_end=now,
        train_start=now - timedelta(days=730),
    )


@pytest.fixture
def training_data() -> VersionedTimeSeriesDataset:
    """Synthetic 15-min data covering 120 days."""
    t0 = datetime(2024, 2, 1, tzinfo=UTC)
    n = 96 * 120
    rng = np.random.default_rng(42)
    index = pd.date_range(t0, periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "load": np.sin(np.arange(n) * 2 * np.pi / 96) * 50 + 100 + rng.normal(0, 5, n),
            "radiation": np.maximum(0, np.sin(np.arange(n) * 2 * np.pi / 96 - 1)) * 500,
            "temperature": 15 + 5 * np.sin(np.arange(n) * 2 * np.pi / 96),
            "windspeed": np.abs(rng.normal(5, 2, n)),
            "pressure": 1013 + rng.normal(0, 5, n),
            "relative_humidity": 70 + rng.normal(0, 10, n),
            "day_ahead_electricity_price": 50 + rng.normal(0, 10, n),
            "available_at": [t0 + timedelta(minutes=15) * (i + 1) for i in range(n)],
        },
        index=index,
    )
    return VersionedTimeSeriesDataset([TimeSeriesDataset(df, timedelta(minutes=15))])


def test_fit_does_not_mutate_template(
    xgboost_config: ForecastingWorkflowConfig,
    benchmark_target: BenchmarkTarget,
    training_data: VersionedTimeSeriesDataset,
    tmp_path: Path,
):
    """fit() should train a deep copy; the template must remain untouched."""
    # Arrange
    factory = create_openstef4_preset_backtest_forecaster(
        workflow_config=xgboost_config,
        cache_dir=tmp_path / "test_no_mutate",
    )
    forecaster = factory(BenchmarkContext(run_name="run"), benchmark_target)
    template_model_id_before = id(forecaster.workflow_template.model)

    horizon = datetime(2024, 5, 25, tzinfo=UTC)
    rhvts = RestrictedHorizonVersionedTimeSeries(dataset=training_data, horizon=horizon)

    # Act
    forecaster.fit(rhvts)

    # Assert — template model object identity unchanged
    assert id(forecaster.workflow_template.model) == template_model_id_before
    assert forecaster.workflow_template.run_name is None

    # The fitted workflow should be different from the template
    assert forecaster._workflow is not forecaster.workflow_template
    assert forecaster._workflow is not None
    assert forecaster._workflow.run_name == horizon.isoformat()


def test_fit_then_predict_returns_forecast(
    xgboost_config: ForecastingWorkflowConfig,
    benchmark_target: BenchmarkTarget,
    training_data: VersionedTimeSeriesDataset,
    tmp_path: Path,
):
    """End-to-end: fit then predict should return a ForecastDataset."""
    # Arrange
    factory = create_openstef4_preset_backtest_forecaster(
        workflow_config=xgboost_config,
        cache_dir=tmp_path / "test_e2e",
    )
    forecaster = factory(BenchmarkContext(run_name="e2e"), benchmark_target)
    horizon = datetime(2024, 5, 25, tzinfo=UTC)
    rhvts = RestrictedHorizonVersionedTimeSeries(dataset=training_data, horizon=horizon)

    # Act
    forecaster.fit(rhvts)
    result = forecaster.predict(rhvts)

    # Assert
    assert result is not None
    assert len(result.data) > 0
