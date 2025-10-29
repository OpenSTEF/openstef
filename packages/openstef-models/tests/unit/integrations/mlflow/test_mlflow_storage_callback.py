# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import time
from datetime import timedelta
from typing import TYPE_CHECKING, cast, override

import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import ModelNotFoundError, SkipFitting
from openstef_core.types import LeadTime, Q
from openstef_models.integrations.mlflow import MLFlowStorage, MLFlowStorageCallback
from openstef_models.mixins.callbacks import WorkflowContext
from openstef_models.models.forecasting import Forecaster, ForecasterConfig
from openstef_models.models.forecasting_model import ForecastingModel, ModelFitResult
from openstef_models.workflows.custom_forecasting_workflow import CustomForecastingWorkflow

if TYPE_CHECKING:
    from pathlib import Path


class SimpleTestForecaster(Forecaster):
    """Simple forecaster for testing that stores and restores median value."""

    def __init__(self, config: ForecasterConfig):
        self._config = config
        self._median_value: float = 0.0
        self._is_fitted = False

    @property
    def config(self) -> ForecasterConfig:
        return self._config

    @property
    @override
    def is_fitted(self) -> bool:
        return self._is_fitted

    @override
    def fit(self, data: ForecastInputDataset, data_val: ForecastInputDataset | None = None) -> None:
        if self._is_fitted:
            return

        self._median_value = float(data.target_series.median())
        self._is_fitted = True

    @override
    def predict(self, data: ForecastInputDataset) -> ForecastDataset:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        forecast_data = pd.DataFrame(
            {quantile.format(): [self._median_value] * len(data.index) for quantile in self.config.quantiles},
            index=data.index,
        )
        return ForecastDataset(forecast_data, data.sample_interval, data.forecast_start)


@pytest.fixture
def storage(tmp_path: Path) -> MLFlowStorage:
    """Create MLflow storage with temporary paths."""
    return MLFlowStorage(
        tracking_uri=str(tmp_path / "mlflow"),
        local_artifacts_path=tmp_path / "artifacts",
    )


@pytest.fixture
def callback(storage: MLFlowStorage) -> MLFlowStorageCallback:
    """Create callback with test storage."""
    return MLFlowStorageCallback(storage=storage)


@pytest.fixture
def sample_dataset() -> TimeSeriesDataset:
    return TimeSeriesDataset(
        data=pd.DataFrame(
            {"load": [100.0, 110.0, 120.0, 105.0, 95.0, 115.0, 125.0, 130.0], "value": 100.0},
            index=pd.date_range("2025-01-01", periods=8, freq="h"),
        ),
        sample_interval=timedelta(hours=1),
    )


@pytest.fixture
def workflow(sample_dataset: TimeSeriesDataset) -> CustomForecastingWorkflow:
    """Create a forecasting workflow for testing."""
    horizons = [LeadTime(timedelta(hours=1))]
    quantiles = [Q(0.5)]

    # Disable train/val/test splitting to avoid R^2 warnings with small dataset
    # (R^2 is undefined with less than 2 samples, and splitting 8 samples creates tiny sets)
    model = ForecastingModel(
        forecaster=SimpleTestForecaster(config=ForecasterConfig(horizons=horizons, quantiles=quantiles)),
        test_fraction=0.0,
        val_fraction=0.0,
    )

    return CustomForecastingWorkflow(model_id="test_model", model=model)


@pytest.fixture
def fit_result(sample_dataset: TimeSeriesDataset, workflow: CustomForecastingWorkflow) -> ModelFitResult:
    """Create a fit result with metrics for testing."""
    # Fit the model and get the result which includes metrics
    return workflow.model.fit(sample_dataset)


def test_mlflow_storage_callback__on_fit_end__stores_model_and_metrics(
    callback: MLFlowStorageCallback, workflow: CustomForecastingWorkflow, fit_result: ModelFitResult
):
    """Test that on_fit_end stores model, data, and metrics to MLflow."""
    # Arrange
    context = WorkflowContext(workflow=workflow)

    # Act
    callback.on_fit_end(context=context, result=fit_result)

    # Assert - Run was created
    runs = callback.storage.search_latest_runs(model_id=workflow.model_id, limit=1)
    assert len(runs) == 1

    # Assert - Model can be loaded from the run
    run_id = cast(str, runs[0].info.run_id)
    loaded_model = callback.storage.load_run_model(run_id=run_id)
    assert isinstance(loaded_model, ForecastingModel)
    assert loaded_model.is_fitted


def test_mlflow_storage_callback__on_predict_start__loads_model_when_not_fitted(
    callback: MLFlowStorageCallback,
    workflow: CustomForecastingWorkflow,
    fit_result: ModelFitResult,
    sample_dataset: TimeSeriesDataset,
):
    """Test that on_predict_start loads model from MLflow when not fitted."""
    # Arrange - Store a fitted model first
    context = WorkflowContext(workflow=workflow)
    callback.on_fit_end(context=context, result=fit_result)

    # Create a new unfitted workflow
    horizons = [LeadTime(timedelta(hours=1))]
    unfitted_workflow = CustomForecastingWorkflow(
        model_id="test_model",
        model=ForecastingModel(
            forecaster=SimpleTestForecaster(config=ForecasterConfig(horizons=horizons, quantiles=[Q(0.5)])),
        ),
    )
    unfitted_context = WorkflowContext(workflow=unfitted_workflow)

    # Act
    callback.on_predict_start(context=unfitted_context, data=sample_dataset)

    # Assert - Model was loaded and is now fitted
    assert unfitted_context.workflow.model.is_fitted


def test_mlflow_storage_callback__on_predict_start__raises_error_when_no_model_exists(
    callback: MLFlowStorageCallback, sample_dataset: TimeSeriesDataset
):
    """Test that on_predict_start raises ModelNotFoundError when no stored model exists."""
    # Arrange - Create unfitted workflow without any stored runs
    horizons = [LeadTime(timedelta(hours=1))]
    unfitted_workflow = CustomForecastingWorkflow(
        model_id="nonexistent_model",
        model=ForecastingModel(
            forecaster=SimpleTestForecaster(config=ForecasterConfig(horizons=horizons, quantiles=[Q(0.5)])),
        ),
    )
    context = WorkflowContext(workflow=unfitted_workflow)

    # Act & Assert
    with pytest.raises(ModelNotFoundError, match="nonexistent_model"):
        callback.on_predict_start(context=context, data=sample_dataset)


@pytest.mark.parametrize(
    ("max_age", "wait_seconds", "should_skip"),
    [
        pytest.param(timedelta(days=7), 0, True, id="recent_model_skip"),
        pytest.param(timedelta(seconds=1), 2, False, id="old_model_refit"),
    ],
)
def test_mlflow_storage_callback__on_fit_start__model_reuse_logic(
    storage: MLFlowStorage,
    workflow: CustomForecastingWorkflow,
    fit_result: ModelFitResult,
    max_age: timedelta,
    wait_seconds: int,
    should_skip: bool,
):
    """Test that on_fit_start skips fitting when model is recent enough."""
    # Arrange - Create callback with specific max age
    callback = MLFlowStorageCallback(storage=storage, model_reuse_max_age=max_age)

    # Store an initial model
    context = WorkflowContext(workflow=workflow)
    callback.on_fit_end(context=context, result=fit_result)

    # Wait if needed for the old model test case
    if wait_seconds > 0:
        time.sleep(wait_seconds)

    # Act & Assert
    if should_skip:
        with pytest.raises(SkipFitting, match="Model is recent enough"):
            callback.on_fit_start(context=context, data=fit_result.input_dataset)
    else:
        # Should not raise - model is too old
        callback.on_fit_start(context=context, data=fit_result.input_dataset)


def test_mlflow_storage_callback__model_selection__keeps_better_model(
    storage: MLFlowStorage,
    workflow: CustomForecastingWorkflow,
    fit_result: ModelFitResult,
    sample_dataset: TimeSeriesDataset,
):
    """Test that model selection keeps the better performing model."""
    # Arrange - Create callback with R2 metric (capital letters)
    callback = MLFlowStorageCallback(
        storage=storage,
        model_selection_metric=(Q(0.5), "R2", "higher_is_better"),
    )

    # Store an initial model with good performance
    context = WorkflowContext(workflow=workflow)
    callback.on_fit_end(context=context, result=fit_result)

    # Create a new "worse" model by manually setting a bad median value
    worse_horizons = [LeadTime(timedelta(hours=1))]
    worse_forecaster = SimpleTestForecaster(config=ForecasterConfig(horizons=worse_horizons, quantiles=[Q(0.5)]))
    worse_forecaster._median_value = 50.0  # Much lower than actual values (~110)
    worse_forecaster._is_fitted = True

    # Disable splits for worse model too to avoid R^2 warnings
    worse_model = ForecastingModel(
        forecaster=worse_forecaster,
        test_fraction=0.0,
        val_fraction=0.0,
    )

    # Create a worse result by fitting with the worse model
    worse_result = worse_model.fit(sample_dataset)

    worse_workflow = CustomForecastingWorkflow(model_id="test_model", model=worse_model)
    worse_context = WorkflowContext(workflow=worse_workflow)

    # Act & Assert - Should raise SkipFitting because new model is worse
    with pytest.raises(SkipFitting, match="New model did not improve"):
        callback.on_fit_end(context=worse_context, result=worse_result)
