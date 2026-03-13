# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from openstef_beam.backtesting.restricted_horizon_timeseries import RestrictedHorizonVersionedTimeSeries
from openstef_beam.benchmarking.baselines.openstef4 import create_openstef4_preset_backtest_forecaster
from openstef_beam.benchmarking.benchmark_pipeline import BenchmarkContext, BenchmarkTarget
from openstef_core.datasets import VersionedTimeSeriesDataset
from openstef_core.testing import create_synthetic_forecasting_dataset
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
    ts = create_synthetic_forecasting_dataset(
        start=datetime(2024, 2, 1, tzinfo=UTC),
        length=timedelta(days=120),
        sample_interval=timedelta(minutes=15),
        include_atmosphere=True,
        include_price=True,
        include_available_at=True,
    )
    return VersionedTimeSeriesDataset([ts])


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


def test_fit_retains_previous_model_on_insufficient_data(
    benchmark_target: BenchmarkTarget,
    training_data: VersionedTimeSeriesDataset,
    tmp_path: Path,
):
    """fit() should skip training and retain the previous model when data has all-NaN targets."""
    # Arrange — disable model reuse to avoid caching side-effects
    config = ForecastingWorkflowConfig(
        model_id="test_insufficient",
        model="xgboost",
        horizons=[LeadTime.from_string("PT24H")],
        quantiles=[Q(0.5)],
        model_reuse_enable=False,
    )
    factory = create_openstef4_preset_backtest_forecaster(
        workflow_config=config,
        cache_dir=tmp_path / "test_insufficient",
    )
    forecaster = factory(BenchmarkContext(run_name="insufficient"), benchmark_target)

    # First fit succeeds — establishes a baseline model
    horizon_good = datetime(2024, 5, 25, tzinfo=UTC)
    rhvts_good = RestrictedHorizonVersionedTimeSeries(dataset=training_data, horizon=horizon_good)
    forecaster.fit(rhvts_good)
    assert forecaster._workflow is not None
    previous_workflow = forecaster._workflow

    # Build a dataset with all-NaN load to trigger InsufficientlyCompleteError naturally
    nan_ts = create_synthetic_forecasting_dataset(
        start=datetime(2024, 2, 1, tzinfo=UTC),
        length=timedelta(days=120),
        sample_interval=timedelta(minutes=15),
        include_atmosphere=True,
        include_price=True,
        include_available_at=True,
    )
    nan_ts.data["load"] = np.nan
    nan_dataset = VersionedTimeSeriesDataset([nan_ts])
    rhvts_nan = RestrictedHorizonVersionedTimeSeries(dataset=nan_dataset, horizon=datetime(2024, 5, 26, tzinfo=UTC))

    # Act — fit with data that has no valid targets
    forecaster.fit(rhvts_nan)

    # Assert — previous model is retained (fit returned early without updating _workflow)
    assert forecaster._workflow is previous_workflow


def test_predict_returns_none_when_never_fitted(
    xgboost_config: ForecastingWorkflowConfig,
    benchmark_target: BenchmarkTarget,
    training_data: VersionedTimeSeriesDataset,
    tmp_path: Path,
):
    """predict() should return None when no model has been fitted yet."""
    # Arrange
    factory = create_openstef4_preset_backtest_forecaster(
        workflow_config=xgboost_config,
        cache_dir=tmp_path / "test_no_fit",
    )
    forecaster = factory(BenchmarkContext(run_name="no_fit"), benchmark_target)
    horizon = datetime(2024, 5, 25, tzinfo=UTC)
    rhvts = RestrictedHorizonVersionedTimeSeries(dataset=training_data, horizon=horizon)

    # Act
    result = forecaster.predict(rhvts)

    # Assert
    assert result is None
