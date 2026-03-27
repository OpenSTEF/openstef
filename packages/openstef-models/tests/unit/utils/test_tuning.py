# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the hyperparameter tuning utilities."""

from typing import Any
from unittest.mock import MagicMock

import optuna
import pytest

from openstef_core.mixins.param_ranges import CategoricalRange, FloatRange, IntRange, ModelTuningInfo
from openstef_models.models.forecasting.xgboost_forecaster import XGBoostHyperParams
from openstef_models.presets.forecasting_workflow import ForecastingWorkflowConfig
from openstef_models.utils.tuning import (
    HyperparameterTuner,
    TuningResult,
    _reconstruct_best_config,  # noqa: PLC2701
    _suggest_hyperparam_value,  # noqa: PLC2701
    apply_trial_suggestions,
    run_optuna_study,
)

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


# _suggest_hyperparam_value


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
def test_suggest_hyperparam_value__calls_correct_optuna_api(
    tuning_range: FloatRange | IntRange | CategoricalRange,
    suggest_method: str,
    call_args: tuple[object, ...],
    call_kwargs: dict[str, object],
) -> None:
    """_suggest_hyperparam_value calls the correct optuna.Trial method for each TuningRange type."""
    # Arrange
    trial = MagicMock(spec=optuna.Trial)
    getattr(trial, suggest_method).return_value = 0.5

    # Act
    _suggest_hyperparam_value(trial, "param", tuning_range)

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
def test_suggest_hyperparam_value__returns_none_for_incomplete_range(
    incomplete_range: FloatRange | IntRange | CategoricalRange,
) -> None:
    """_suggest_hyperparam_value returns None when bounds or choices are missing."""
    # Arrange
    trial = MagicMock(spec=optuna.Trial)

    # Act / Assert
    assert _suggest_hyperparam_value(trial, "param", incomplete_range) is None


# apply_trial_suggestions


def test_apply_trial_suggestions__updates_all_fields_from_trial() -> None:
    """apply_trial_suggestions applies all trial suggestions and returns an updated HyperParams."""
    # Arrange
    hp = XGBoostHyperParams()
    trial = MagicMock(spec=optuna.Trial)
    trial.suggest_float.return_value = 0.1
    trial.suggest_int.return_value = 200
    trial.suggest_categorical.return_value = "lossguide"
    space: dict[str, FloatRange | IntRange | CategoricalRange] = {
        "learning_rate": FloatRange(0.01, 0.5, log=True),
        "n_estimators": IntRange(50, 500),
        "grow_policy": CategoricalRange(("depthwise", "lossguide")),
    }

    # Act
    result = apply_trial_suggestions(trial, space, hp)

    # Assert
    assert result.learning_rate == pytest.approx(0.1)
    assert result.n_estimators == 200
    assert result.grow_policy == "lossguide"


def test_apply_trial_suggestions__skips_fields_with_none_bounds() -> None:
    """apply_trial_suggestions leaves fields unchanged when their range has None bounds."""
    # Arrange
    hp = XGBoostHyperParams()
    trial = MagicMock(spec=optuna.Trial)

    # Act
    result = apply_trial_suggestions(trial, {"n_estimators": IntRange(None, None)}, hp)

    # Assert
    assert result.n_estimators == hp.n_estimators


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

    result = TuningResult(workflow=MagicMock(), fit_result=None, study=study, best_config=MagicMock())

    # Act / Assert
    assert repr(result) == expected_repr


# run_optuna_study


@pytest.mark.parametrize("n_trials", [1, 3])
def test_run_optuna_study__runs_exactly_n_trials(n_trials: int) -> None:
    """run_optuna_study produces a study containing exactly n_trials completed trials."""
    study = run_optuna_study(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=n_trials, seed=0)

    assert len(study.trials) == n_trials


@pytest.mark.parametrize(
    ("direction", "expected_direction_name"),
    [
        pytest.param("maximize", "MAXIMIZE", id="maximize"),
        pytest.param("minimize", "MINIMIZE", id="minimize"),
    ],
)
def test_run_optuna_study__respects_direction(direction: str, expected_direction_name: str) -> None:
    """run_optuna_study creates a study with the specified optimisation direction."""
    study = run_optuna_study(
        lambda trial: trial.suggest_float("x", 0.0, 1.0),
        n_trials=1,
        seed=0,
        direction=direction,  # type: ignore[arg-type]
    )

    assert study.direction.name == expected_direction_name


# _reconstruct_best_config


def test_reconstruct_best_config__single_model_applies_best_params() -> None:
    """_reconstruct_best_config updates the single hyperparams field with best_params values."""
    # Arrange
    config = _config()
    info = ModelTuningInfo(
        field_name="xgboost_hyperparams",
        hyperparams=config.xgboost_hyperparams,
        search_space={"n_estimators": IntRange(50, 500, tune=True)},
    )
    study = MagicMock(spec=optuna.Study)
    study.best_params = {"n_estimators": 300}

    # Act
    best_config = _reconstruct_best_config(config, [info], study)

    # Assert
    assert best_config.xgboost_hyperparams.n_estimators == 300
    assert best_config.xgboost_hyperparams.max_depth == config.xgboost_hyperparams.max_depth


def test_reconstruct_best_config__multi_model_parses_dotted_trial_keys() -> None:
    """_reconstruct_best_config routes 'field.param' keys to the correct HP group."""
    # Arrange
    config = _config()
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
    best_config = _reconstruct_best_config(config, [info_xgb, info_gblinear], study)

    # Assert
    assert best_config.xgboost_hyperparams.n_estimators == 300
    assert best_config.gblinear_hyperparams.n_estimators == 150


# HyperparameterTuner


def test_hyperparameter_tuner__tune_raises_when_no_tunable_hyperparams() -> None:
    """HyperparameterTuner.tune() raises ValueError when no tune=True ranges exist."""
    # Arrange
    tuner = HyperparameterTuner(
        config=_config(),
        train_dataset=MagicMock(),
        create_workflow=MagicMock(),
        target_quantile="global",
        metric_name="R2",
    )

    # Act / Assert
    with pytest.raises(ValueError, match="No tunable hyperparameters"):
        tuner.tune()


def test_hyperparameter_tuner__tune_returns_best_config_study_and_params() -> None:
    """HyperparameterTuner.tune() returns a best_config, completed study, and best_params dict."""
    # Arrange
    config = _config(xgboost_hyperparams=XGBoostHyperParams(n_estimators=IntRange(50, 500, tune=True)))
    create_workflow = MagicMock(return_value=_make_mock_workflow())
    tuner = HyperparameterTuner(
        config=config,
        train_dataset=MagicMock(),
        create_workflow=create_workflow,
        target_quantile="global",
        metric_name="R2",
        n_trials=2,
        seed=0,
    )

    # Act
    best_config, study, best_params = tuner.tune()

    # Assert
    assert isinstance(best_config, ForecastingWorkflowConfig)
    assert isinstance(study, optuna.Study)
    assert len(study.trials) == 2
    assert "n_estimators" in best_params
    assert 50 <= best_params["n_estimators"] <= 500


def test_hyperparameter_tuner__fit_with_tuning_returns_tuning_result() -> None:
    """HyperparameterTuner.fit_with_tuning() returns a TuningResult with best config and fitted workflow."""
    # Arrange
    config = _config(xgboost_hyperparams=XGBoostHyperParams(n_estimators=IntRange(50, 500, tune=True)))
    create_workflow = MagicMock(return_value=_make_mock_workflow())
    tuner = HyperparameterTuner(
        config=config,
        train_dataset=MagicMock(),
        create_workflow=create_workflow,
        target_quantile="global",
        metric_name="R2",
        n_trials=2,
        seed=0,
    )

    # Act
    result = tuner.fit_with_tuning()

    # Assert
    assert isinstance(result, TuningResult)
    assert isinstance(result.study, optuna.Study)
    # 2 trial fits during tuning + 1 final fit with the best config
    assert create_workflow.call_count == 2 + 1
