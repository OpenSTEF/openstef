# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the Optuna hyperparameter tuning integration."""

from typing import Any
from unittest.mock import MagicMock

import optuna
import pytest

from openstef_core.base_model import BaseConfig
from openstef_core.mixins.param_ranges import CategoricalRange, FloatRange, IntRange, ModelTuningInfo
from openstef_models.integrations.optuna import (
    HyperparameterTuner,
    TuningResult,
)
from openstef_models.models.forecasting.xgboost_forecaster import XGBoostHyperParams
from openstef_models.presets.forecasting_workflow import ForecastingWorkflowConfig

# Suppress Optuna progress output during tests
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _config(**kwargs: Any) -> ForecastingWorkflowConfig:
    """Minimal ForecastingWorkflowConfig for tuning tests."""
    return ForecastingWorkflowConfig(model_id="test", model="xgboost", **kwargs)


def _make_mock_workflow(score: float = 0.8) -> MagicMock:
    """Return a mock workflow whose fit() reports a fixed metric score."""
    mock_metrics = MagicMock()
    mock_metrics.get_metric.return_value = score
    mock_fit_result = MagicMock()
    mock_fit_result.metrics_val = mock_metrics
    mock_workflow = MagicMock()
    mock_workflow.fit.return_value = mock_fit_result
    return mock_workflow


def _make_tuner(**overrides: Any) -> HyperparameterTuner[ForecastingWorkflowConfig]:
    """Create a HyperparameterTuner with sensible defaults for testing."""
    defaults: dict[str, Any] = {
        "config": _config(),
        "train_dataset": MagicMock(),
        "create_workflow": MagicMock(return_value=_make_mock_workflow()),
        "target_quantile": "global",
        "metric_name": "R2",
        "n_trials": 2,
        "seed": 0,
    }
    defaults.update(overrides)
    return HyperparameterTuner(**defaults)


# _suggest_value


@pytest.mark.parametrize(
    ("tuning_range", "suggest_method", "call_args", "call_kwargs"),
    [
        pytest.param(FloatRange(0.1, 0.9), "suggest_float", ("param", 0.1, 0.9), {"log": False}, id="float"),
        pytest.param(
            FloatRange(0.01, 0.5, log=True), "suggest_float", ("param", 0.01, 0.5), {"log": True}, id="float-log"
        ),
        pytest.param(IntRange(10, 100), "suggest_int", ("param", 10, 100), {"log": False}, id="int"),
        pytest.param(CategoricalRange(("a", "b")), "suggest_categorical", ("param", ["a", "b"]), {}, id="categorical"),
    ],
)
def test_suggest_value__calls_correct_optuna_api(
    tuning_range: FloatRange | IntRange | CategoricalRange,
    suggest_method: str,
    call_args: tuple[object, ...],
    call_kwargs: dict[str, object],
) -> None:
    """_suggest_value calls the correct optuna.Trial method for each TuningRange type."""
    # Arrange
    trial = MagicMock(spec=optuna.Trial)
    getattr(trial, suggest_method).return_value = 0.5

    # Act
    HyperparameterTuner.suggest_value(trial, "param", tuning_range)

    # Assert
    getattr(trial, suggest_method).assert_called_once_with(*call_args, **call_kwargs)


@pytest.mark.parametrize(
    "incomplete_range",
    [
        pytest.param(FloatRange(None, 0.9), id="float-missing-low"),
        pytest.param(FloatRange(0.1, None), id="float-missing-high"),
        pytest.param(IntRange(None, 100), id="int-missing-low"),
        pytest.param(CategoricalRange(None), id="categorical-missing-choices"),
    ],
)
def test_suggest_value__returns_none_for_incomplete_range(
    incomplete_range: FloatRange | IntRange | CategoricalRange,
) -> None:
    """_suggest_value returns None when bounds or choices are missing."""
    # Arrange
    trial = MagicMock(spec=optuna.Trial)

    # Act / Assert
    assert HyperparameterTuner.suggest_value(trial, "param", incomplete_range) is None


# TuningResult


@pytest.mark.parametrize(
    ("best_params", "expected_repr"),
    [
        pytest.param({"alpha": 0.5, "n_iter": 42}, "TuningResult(2 params tuned)", id="with-params"),
        pytest.param({}, "TuningResult(no tuning)", id="no-params"),
    ],
)
def test_tuning_result__repr_reflects_param_count(best_params: dict[str, Any], expected_repr: str) -> None:
    """TuningResult.__repr__ summarises the count of tuned parameters."""
    # Arrange
    study = MagicMock(spec=optuna.Study)
    study.best_params = best_params

    result = TuningResult(study=study, best_config=MagicMock(), workflow=MagicMock())

    # Act / Assert
    assert repr(result) == expected_repr


# HyperparameterTuner._reconstruct_best_config


def test_reconstruct_best_config__single_model_applies_best_params() -> None:
    """_reconstruct_best_config updates the single hyperparams field with best_params values."""
    # Arrange
    config = _config()
    tuner = _make_tuner(config=config)
    info = ModelTuningInfo(
        field_name="xgboost_hyperparams",
        hyperparams=config.xgboost_hyperparams,
        search_space={"n_estimators": IntRange(50, 500, tune=True)},
    )
    study = MagicMock(spec=optuna.Study)
    study.best_params = {"n_estimators": 300}

    # Act
    best_config = tuner._reconstruct_best_config(config, [info], study)

    # Assert
    assert best_config.xgboost_hyperparams.n_estimators == 300
    assert best_config.xgboost_hyperparams.max_depth == config.xgboost_hyperparams.max_depth


def test_reconstruct_best_config__multi_model_parses_dotted_trial_keys() -> None:
    """_reconstruct_best_config routes 'field.param' keys to the correct HP group."""
    # Arrange
    config = _config()
    tuner = _make_tuner(config=config)
    info_xgb = ModelTuningInfo(
        field_name="xgboost_hyperparams",
        hyperparams=config.xgboost_hyperparams,
        search_space={"n_estimators": IntRange(50, 500, tune=True)},
    )
    info_gblinear = ModelTuningInfo(
        field_name="gblinear_hyperparams",
        hyperparams=config.gblinear_hyperparams,
        search_space={"n_estimators": IntRange(50, 500, tune=True)},
    )
    study = MagicMock(spec=optuna.Study)
    study.best_params = {"xgboost_hyperparams.n_estimators": 300, "gblinear_hyperparams.n_estimators": 150}

    # Act
    best_config = tuner._reconstruct_best_config(config, [info_xgb, info_gblinear], study)

    # Assert
    assert best_config.xgboost_hyperparams.n_estimators == 300
    assert best_config.gblinear_hyperparams.n_estimators == 150


# HyperparameterTuner.tune / fit_with_tuning


def test_hyperparameter_tuner__tune_raises_when_no_tunable_hyperparams() -> None:
    """HyperparameterTuner.tune() raises ValueError when no tune=True ranges exist."""
    # Arrange
    tuner = _make_tuner()

    # Act / Assert
    with pytest.raises(ValueError, match="No tunable hyperparameters"):
        tuner.tune()


def test_hyperparameter_tuner__tune_returns_best_config_and_study() -> None:
    """HyperparameterTuner.tune() returns a best_config and completed study."""
    # Arrange
    config = _config(xgboost_hyperparams=XGBoostHyperParams(n_estimators=IntRange(50, 500, tune=True)))
    tuner = _make_tuner(config=config)

    # Act
    best_config, study = tuner.tune()

    # Assert
    assert isinstance(best_config, ForecastingWorkflowConfig)
    assert isinstance(study, optuna.Study)
    assert len(study.trials) == 2
    assert "n_estimators" in study.best_params
    assert 50 <= study.best_params["n_estimators"] <= 500


def test_hyperparameter_tuner__fit_with_tuning_returns_tuning_result() -> None:
    """HyperparameterTuner.fit_with_tuning() returns a TuningResult with best config and study."""
    # Arrange
    config = _config(xgboost_hyperparams=XGBoostHyperParams(n_estimators=IntRange(50, 500, tune=True)))
    create_workflow = MagicMock(return_value=_make_mock_workflow())
    tuner = _make_tuner(config=config, create_workflow=create_workflow)

    # Act
    result = tuner.fit_with_tuning()

    # Assert
    assert isinstance(result, TuningResult)
    assert isinstance(result.study, optuna.Study)
    assert isinstance(result.workflow, MagicMock)
    # 2 trial fits during tuning + 1 final fit with the best config
    assert create_workflow.call_count == 3


def test_hyperparameter_tuner__create_study_returns_configured_study() -> None:
    """_create_study creates a study with the configured direction and study name."""
    # Arrange
    tuner = _make_tuner(direction="minimize", study_name="custom_study")

    # Act
    study = tuner._create_study()

    # Assert
    assert study.direction.name == "MINIMIZE"
    assert study.study_name == "custom_study"


# Metric name validation


def test_hyperparameter_tuner__tune_raises_for_invalid_metric_name_with_providers() -> None:
    """tune() raises ValueError when metric_name doesn't match any provider metric_names."""
    # Arrange — config with evaluation_metrics whose providers declare metric_names
    mock_provider = MagicMock()
    mock_provider.metric_names = frozenset({"R2", "MAE"})
    config = _config()
    config = config.model_copy(update={"evaluation_metrics": [mock_provider]})
    tuner = _make_tuner(
        config=config,
        metric_name="nonexistent_metric",
        # Need at least one tunable field so tune() doesn't exit early
    )

    # Act / Assert
    with pytest.raises(ValueError, match=r"nonexistent_metric.*not provided"):
        tuner.tune()


def test_hyperparameter_tuner__tune_skips_validation_when_no_providers() -> None:
    """tune() skips metric_name validation when config lacks evaluation_metrics."""
    # Arrange — use a bare BaseConfig without evaluation_metrics field
    bare_config = BaseConfig()
    tuner = HyperparameterTuner(
        config=bare_config,
        train_dataset=MagicMock(),
        create_workflow=MagicMock(),
        target_quantile="global",
        metric_name="anything",
        n_trials=2,
        seed=0,
    )

    # Act / Assert — raises for missing tunable params, NOT for invalid metric
    with pytest.raises(ValueError, match="No tunable hyperparameters"):
        tuner.tune()


# n_jobs parallelism


def test_hyperparameter_tuner__n_jobs_defaults_to_one() -> None:
    """HyperparameterTuner.n_jobs defaults to 1."""
    # Arrange / Act
    tuner = _make_tuner()

    # Assert
    assert tuner.n_jobs == 1
