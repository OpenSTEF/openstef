# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for EnsembleMLFlowStorageCallback."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, cast, override

import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.validated_datasets import EnsembleForecastDataset, ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import SkipFitting
from openstef_core.mixins.predictor import HyperParams
from openstef_core.types import LeadTime, Q
from openstef_meta.integrations.mlflow import EnsembleMLFlowStorageCallback
from openstef_meta.models.ensemble_forecasting_model import EnsembleForecastingModel, EnsembleModelFitResult
from openstef_meta.models.forecast_combiners.forecast_combiner import ForecastCombiner, ForecastCombinerConfig
from openstef_meta.workflows import CustomEnsembleForecastingWorkflow
from openstef_models.integrations.mlflow import MLFlowStorage
from openstef_models.mixins.callbacks import WorkflowContext
from openstef_models.models.forecasting.forecaster import Forecaster, ForecasterConfig

if TYPE_CHECKING:
    from pathlib import Path


class SimpleForecasterHyperParams(HyperParams):
    """Test hyperparameters for the simple forecaster."""

    alpha: float = 0.5
    n_rounds: int = 100


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

    @property
    @override
    def hyperparams(self) -> SimpleForecasterHyperParams:
        return SimpleForecasterHyperParams()

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


class SimpleCombinerHyperParams(HyperParams):
    """Test hyperparameters for the simple combiner."""

    learning_rate: float = 0.01


class SimpleTestCombiner(ForecastCombiner):
    """Simple combiner for testing that averages base forecaster predictions."""

    def __init__(self, config: ForecastCombinerConfig):
        self.config = config
        self._is_fitted = False
        self.quantiles = config.quantiles

    @override
    def fit(
        self,
        data: EnsembleForecastDataset,
        data_val: EnsembleForecastDataset | None = None,
        additional_features: ForecastInputDataset | None = None,
    ) -> None:
        self._is_fitted = True

    @override
    def predict(
        self,
        data: EnsembleForecastDataset,
        additional_features: ForecastInputDataset | None = None,
    ) -> ForecastDataset:
        combined_data = pd.DataFrame(index=data.data.index)
        for quantile in self.quantiles:
            quantile_cols = [col for col in data.data.columns if col.endswith(quantile.format())]
            combined_data[quantile.format()] = data.data[quantile_cols].mean(axis=1)
        return ForecastDataset(
            data=combined_data, sample_interval=data.sample_interval, forecast_start=data.forecast_start
        )

    @property
    @override
    def is_fitted(self) -> bool:
        return self._is_fitted

    @override
    def predict_contributions(
        self,
        data: EnsembleForecastDataset,
        additional_features: ForecastInputDataset | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()


# --- Fixtures ---


@pytest.fixture
def storage(tmp_path: Path) -> MLFlowStorage:
    """Create MLflow storage with temporary paths."""
    return MLFlowStorage(
        tracking_uri=str(tmp_path / "mlflow"),
        local_artifacts_path=tmp_path / "artifacts",
    )


@pytest.fixture
def callback(storage: MLFlowStorage) -> EnsembleMLFlowStorageCallback:
    """Create ensemble callback with test storage."""
    return EnsembleMLFlowStorageCallback(storage=storage)


@pytest.fixture
def sample_dataset() -> TimeSeriesDataset:
    return TimeSeriesDataset(
        data=pd.DataFrame(
            {"load": [100.0, 110.0, 120.0, 105.0, 95.0, 115.0, 125.0, 130.0], "value": 100.0},
            index=pd.date_range("2025-01-01", periods=8, freq="h"),
        ),
        sample_interval=timedelta(hours=1),
    )


def _create_ensemble_workflow() -> CustomEnsembleForecastingWorkflow:
    """Create an ensemble forecasting workflow for testing."""
    horizons = [LeadTime(timedelta(hours=1))]
    quantiles = [Q(0.5)]
    config = ForecasterConfig(horizons=horizons, quantiles=quantiles)

    forecasters: dict[str, Forecaster] = {
        "model_a": SimpleTestForecaster(config=config),
        "model_b": SimpleTestForecaster(config=config),
    }
    combiner_config = ForecastCombinerConfig(
        quantiles=quantiles,
        horizons=horizons,
        hyperparams=SimpleCombinerHyperParams(),
    )
    combiner = SimpleTestCombiner(config=combiner_config)

    ensemble_model = EnsembleForecastingModel(
        forecasters=forecasters,
        combiner=combiner,
    )

    return CustomEnsembleForecastingWorkflow(model_id="test_ensemble", model=ensemble_model)


@pytest.fixture
def ensemble_workflow() -> CustomEnsembleForecastingWorkflow:
    return _create_ensemble_workflow()


@pytest.fixture
def ensemble_fit_result(
    sample_dataset: TimeSeriesDataset, ensemble_workflow: CustomEnsembleForecastingWorkflow
) -> EnsembleModelFitResult:
    """Create a fit result from the ensemble model."""
    return ensemble_workflow.model.fit(sample_dataset)


# --- Tests ---


def test_on_fit_end__stores_ensemble_model(
    callback: EnsembleMLFlowStorageCallback,
    ensemble_workflow: CustomEnsembleForecastingWorkflow,
    ensemble_fit_result: EnsembleModelFitResult,
):
    """Test that on_fit_end stores an EnsembleForecastingModel to MLflow."""
    context = WorkflowContext(workflow=ensemble_workflow)

    callback.on_fit_end(context=context, result=ensemble_fit_result)

    runs = callback.storage.search_latest_runs(model_id=ensemble_workflow.model_id, limit=1)
    assert len(runs) == 1

    run_id = cast(str, runs[0].info.run_id)
    loaded_model = callback.storage.load_run_model(model_id=ensemble_workflow.model_id, run_id=run_id)
    assert isinstance(loaded_model, EnsembleForecastingModel)
    assert loaded_model.is_fitted
    assert set(loaded_model.forecasters.keys()) == {"model_a", "model_b"}


def test_on_fit_end__logs_combiner_hyperparams_as_primary(
    callback: EnsembleMLFlowStorageCallback,
    ensemble_workflow: CustomEnsembleForecastingWorkflow,
    ensemble_fit_result: EnsembleModelFitResult,
):
    """Test that combiner hyperparams are logged as the run's primary params."""
    context = WorkflowContext(workflow=ensemble_workflow)

    callback.on_fit_end(context=context, result=ensemble_fit_result)

    runs = callback.storage.search_latest_runs(model_id=ensemble_workflow.model_id, limit=1)
    run = runs[0]
    params = run.data.params  # pyright: ignore[reportUnknownMemberType]

    # Combiner hyperparams should be logged as primary params
    assert "learning_rate" in params
    assert params["learning_rate"] == "0.01"


def test_on_fit_end__logs_per_forecaster_hyperparams(
    callback: EnsembleMLFlowStorageCallback,
    ensemble_workflow: CustomEnsembleForecastingWorkflow,
    ensemble_fit_result: EnsembleModelFitResult,
):
    """Test that per-forecaster hyperparams are logged with name prefixes."""
    context = WorkflowContext(workflow=ensemble_workflow)

    callback.on_fit_end(context=context, result=ensemble_fit_result)

    runs = callback.storage.search_latest_runs(model_id=ensemble_workflow.model_id, limit=1)
    run = runs[0]
    params = run.data.params  # pyright: ignore[reportUnknownMemberType]

    # Per-forecaster hyperparams should be prefixed
    assert "model_a.alpha" in params
    assert params["model_a.alpha"] == "0.5"
    assert "model_a.n_rounds" in params
    assert params["model_a.n_rounds"] == "100"
    assert "model_b.alpha" in params
    assert "model_b.n_rounds" in params


def test_on_predict_start__loads_ensemble_model(
    callback: EnsembleMLFlowStorageCallback,
    ensemble_workflow: CustomEnsembleForecastingWorkflow,
    ensemble_fit_result: EnsembleModelFitResult,
    sample_dataset: TimeSeriesDataset,
):
    """Test that on_predict_start loads an ensemble model from MLflow."""
    # Store a fitted model first
    context = WorkflowContext(workflow=ensemble_workflow)
    callback.on_fit_end(context=context, result=ensemble_fit_result)

    # Create a new unfitted ensemble workflow
    unfitted_workflow = _create_ensemble_workflow()
    unfitted_context = WorkflowContext(workflow=unfitted_workflow)

    callback.on_predict_start(context=unfitted_context, data=sample_dataset)

    assert unfitted_context.workflow.model.is_fitted


def test_model_selection__keeps_better_ensemble_model(
    storage: MLFlowStorage,
    ensemble_workflow: CustomEnsembleForecastingWorkflow,
    ensemble_fit_result: EnsembleModelFitResult,
    sample_dataset: TimeSeriesDataset,
):
    """Test that model selection keeps the better performing ensemble model."""
    callback = EnsembleMLFlowStorageCallback(
        storage=storage,
        model_selection_metric=(Q(0.5), "R2", "higher_is_better"),
    )

    # Store an initial ensemble model
    context = WorkflowContext(workflow=ensemble_workflow)
    callback.on_fit_end(context=context, result=ensemble_fit_result)

    # Create a worse ensemble
    horizons = [LeadTime(timedelta(hours=1))]
    quantiles = [Q(0.5)]
    config = ForecasterConfig(horizons=horizons, quantiles=quantiles)

    worse_a = SimpleTestForecaster(config=config)
    worse_a._median_value = 50.0
    worse_a._is_fitted = True
    worse_b = SimpleTestForecaster(config=config)
    worse_b._median_value = 50.0
    worse_b._is_fitted = True

    worse_ensemble = EnsembleForecastingModel(
        forecasters={"model_a": worse_a, "model_b": worse_b},
        combiner=SimpleTestCombiner(
            config=ForecastCombinerConfig(
                quantiles=quantiles,
                horizons=horizons,
                hyperparams=SimpleCombinerHyperParams(),
            )
        ),
    )
    worse_result = worse_ensemble.fit(sample_dataset)
    worse_workflow = CustomEnsembleForecastingWorkflow(model_id="test_ensemble", model=worse_ensemble)
    worse_context = WorkflowContext(workflow=worse_workflow)

    with pytest.raises(SkipFitting, match="New model did not improve"):
        callback.on_fit_end(context=worse_context, result=worse_result)
