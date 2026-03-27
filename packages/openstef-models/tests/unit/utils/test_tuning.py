# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the hyperparameter tuning utilities."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import optuna
import pytest

from openstef_models.models.forecasting.xgboost_forecaster import XGBoostHyperParams
from openstef_models.presets.forecasting_workflow import ForecastingWorkflowConfig
from openstef_models.utils.tuning import (
    CategoricalRange,
    FloatRange,
    IntRange,
    ModelTuningInfo,
    TuningResult,
    _merge_range,  # noqa: PLC2701
    _reconstruct_best_config,  # noqa: PLC2701
    _suggest_hyperparam_value,  # noqa: PLC2701
    apply_trial_suggestions,
    fit_with_tuning,
    get_search_space,
    run_optuna_study,
    tune,
)

# Suppress Optuna progress output during tests
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _config(**kwargs: Any) -> ForecastingWorkflowConfig:
    """Minimal ForecastingWorkflowConfig for tuning tests."""
    return ForecastingWorkflowConfig(
        model_id="test",
        model="xgboost",
        optuna_n_trials=2,
        optuna_seed=0,
        **kwargs,
    )


# Helper


def _make_mock_workflow(score: float = 0.8) -> MagicMock:
    """Return a mock workflow whose fit() reports a fixed metric score."""
    mock_metrics = MagicMock()
    mock_metrics.get_metric.return_value = score
    mock_fit_result = MagicMock()
    mock_fit_result.metrics_val = mock_metrics
    mock_workflow = MagicMock()
    mock_workflow.fit.return_value = mock_fit_result
    return mock_workflow


# TunableHyperParams


def test_tunable_hyperparams__range_extracted_and_field_keeps_default() -> None:
    """A TuningRange passed at construction is stored in instance_ranges; the field keeps its default."""
    hp = XGBoostHyperParams(n_estimators=IntRange(100, 800, tune=True))

    assert hp.instance_ranges["n_estimators"] == IntRange(100, 800, tune=True)
    assert hp.n_estimators == 100  # default preserved


def test_tunable_hyperparams__multiple_ranges_all_extracted() -> None:
    """All TuningRange values passed at construction appear in instance_ranges."""
    hp = XGBoostHyperParams(
        n_estimators=IntRange(100, 800, tune=True),
        learning_rate=FloatRange(0.01, 0.5, log=True, tune=True),
        grow_policy=CategoricalRange(("depthwise",), tune=True),
    )

    assert set(hp.instance_ranges.keys()) == {"n_estimators", "learning_rate", "grow_policy"}


def test_tunable_hyperparams__scalar_values_not_extracted() -> None:
    """Scalar field values are stored normally and never appear in instance_ranges."""
    hp = XGBoostHyperParams(n_estimators=200)

    assert hp.n_estimators == 200
    assert hp.instance_ranges == {}


# ModelTuningInfo


def test_model_tuning_info__raises_on_empty_search_space() -> None:
    """ModelTuningInfo raises ValueError when search_space is empty."""
    with pytest.raises(ValueError, match=r"search_space.*must not be empty"):
        ModelTuningInfo(
            model_hyperparams_field_name="xgboost_hyperparams",
            tunable_hyperparams=XGBoostHyperParams(),
            search_space={},
        )


# _merge_range


@pytest.mark.parametrize(
    ("override", "class_range", "expected"),
    [
        pytest.param(
            FloatRange(None, None, tune=True),
            FloatRange(0.01, 0.5, log=True),
            FloatRange(0.01, 0.5, log=False, tune=True),
            id="float-none-bounds-fallback",
        ),
        pytest.param(
            FloatRange(0.2, 0.8, tune=True),
            FloatRange(0.01, 0.5),
            FloatRange(0.2, 0.8, log=False, tune=True),
            id="float-override-bounds-preserved",
        ),
        pytest.param(
            FloatRange(None, 0.8, tune=True),
            None,
            FloatRange(None, 0.8, log=False, tune=True),
            id="float-none-stays-without-class-range",
        ),
        pytest.param(
            IntRange(None, None, tune=True),
            IntRange(10, 100),
            IntRange(10, 100, log=False, tune=True),
            id="int-none-bounds-fallback",
        ),
        pytest.param(
            IntRange(20, 80, tune=True),
            IntRange(10, 100),
            IntRange(20, 80, log=False, tune=True),
            id="int-override-bounds-preserved",
        ),
        pytest.param(
            CategoricalRange(None, tune=True),
            CategoricalRange(("x", "y")),
            CategoricalRange(("x", "y"), tune=True),
            id="categorical-none-choices-fallback",
        ),
        pytest.param(
            CategoricalRange(("a",), tune=True),
            CategoricalRange(("x", "y")),
            CategoricalRange(("a",), tune=True),
            id="categorical-override-choices-preserved",
        ),
    ],
)
def test_merge_range__fills_none_bounds_from_class_range(
    override: FloatRange | IntRange | CategoricalRange,
    class_range: FloatRange | IntRange | CategoricalRange | None,
    expected: FloatRange | IntRange | CategoricalRange,
) -> None:
    """_merge_range fills None bounds/choices from class_range; tune always comes from override."""
    assert _merge_range(override, class_range) == expected


# ===========================================================================
# get_search_space
# ===========================================================================


def test_get_search_space__returns_instance_ranges_with_tune_true() -> None:
    """get_search_space includes instance ranges where tune=True."""
    space = get_search_space(XGBoostHyperParams(n_estimators=IntRange(100, 800, tune=True)))

    assert space["n_estimators"] == IntRange(100, 800, tune=True)


def test_get_search_space__excludes_instance_ranges_with_tune_false() -> None:
    """get_search_space excludes instance ranges where tune=False."""
    assert "n_estimators" not in get_search_space(XGBoostHyperParams(n_estimators=IntRange(100, 800, tune=False)))


def test_get_search_space__returns_empty_when_no_tune_true_annotations() -> None:
    """get_search_space returns empty dict when no field has tune=True (XGBoostHyperParams class defaults)."""
    assert get_search_space(XGBoostHyperParams()) == {}


def test_get_search_space__merges_none_bounds_from_class_annotation() -> None:
    """Instance ranges with None bounds are filled from the class-level Annotated defaults."""
    # XGBoostHyperParams.learning_rate has FloatRange(0.01, 0.5, log=True) as its class annotation
    space = get_search_space(XGBoostHyperParams(learning_rate=FloatRange(None, None, log=True, tune=True)))

    result = space["learning_rate"]
    assert isinstance(result, FloatRange)
    assert result.low == pytest.approx(0.01)
    assert result.high == pytest.approx(0.5)
    assert result.log is True


def test_get_search_space__include_restricts_to_requested_fields() -> None:
    """get_search_space with include returns only the requested field names."""
    hp = XGBoostHyperParams(n_estimators=IntRange(100, 800, tune=True), max_depth=IntRange(1, 10, tune=True))
    assert set(get_search_space(hp, include={"n_estimators"}).keys()) == {"n_estimators"}


@pytest.mark.parametrize(
    "bad_include",
    [
        pytest.param({"nonexistent"}, id="field-missing-from-class"),
        pytest.param({"objective"}, id="field-has-no-tune-annotation"),
    ],
)
def test_get_search_space__raises_key_error_for_invalid_include(bad_include: set[str]) -> None:
    """get_search_space raises KeyError when include contains an absent or non-tunable field."""
    with pytest.raises(KeyError):
        get_search_space(XGBoostHyperParams(n_estimators=IntRange(100, 800, tune=True)), include=bad_include)


# _suggest_hyperparam_value


@pytest.mark.parametrize(
    ("tuning_range", "suggest_method", "call_args", "call_kwargs"),
    [
        (FloatRange(0.1, 0.9), "suggest_float", ("param", 0.1, 0.9), {"log": False}),
        (FloatRange(0.01, 0.5, log=True), "suggest_float", ("param", 0.01, 0.5), {"log": True}),
        (IntRange(10, 100), "suggest_int", ("param", 10, 100), {"log": False}),
        (CategoricalRange(("a", "b")), "suggest_categorical", ("param", ["a", "b"]), {}),
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
        FloatRange(None, 0.9),
        FloatRange(0.1, None),
        IntRange(None, 100),
        CategoricalRange(None),
    ],
)
def test_suggest_hyperparam_value__returns_none_for_incomplete_range(
    incomplete_range: FloatRange | IntRange | CategoricalRange,
) -> None:
    """_suggest_hyperparam_value returns None when bounds or choices are missing."""
    # Arrange
    trial = MagicMock(spec=optuna.Trial)

    # Act
    result = _suggest_hyperparam_value(trial, "param", incomplete_range)

    # Assert
    assert result is None
    trial.suggest_float.assert_not_called()
    trial.suggest_int.assert_not_called()
    trial.suggest_categorical.assert_not_called()


# apply_trial_suggestions


def test_apply_trial_suggestions__updates_all_fields_from_trial() -> None:
    """apply_trial_suggestions applies all trial suggestions and returns an updated HyperParams."""
    # Arrange
    hp = XGBoostHyperParams()
    trial = MagicMock(spec=optuna.Trial)
    trial.suggest_float.return_value = 0.1
    trial.suggest_int.return_value = 200
    trial.suggest_categorical.return_value = "lossguide"
    space: dict[str, Any] = {
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
    trial.suggest_float.assert_called_once_with("learning_rate", 0.01, 0.5, log=True)
    trial.suggest_int.assert_called_once_with("n_estimators", 50, 500, log=False)
    trial.suggest_categorical.assert_called_once_with("grow_policy", ["depthwise", "lossguide"])


def test_apply_trial_suggestions__skips_fields_with_none_bounds() -> None:
    """apply_trial_suggestions leaves fields unchanged when their range has None bounds."""
    hp = XGBoostHyperParams()
    trial = MagicMock(spec=optuna.Trial)

    result = apply_trial_suggestions(trial, {"n_estimators": IntRange(None, None)}, hp)

    assert result.n_estimators == hp.n_estimators
    trial.suggest_int.assert_not_called()


# TuningConfigMixin


def test_tuning_config_mixin__discovers_all_tunable_fields() -> None:
    """get_model_tuning_info returns one ModelTuningInfo per TunableHyperParams field with a non-empty search space."""
    config = _config(xgboost_hyperparams=XGBoostHyperParams(n_estimators=IntRange(50, 500, tune=True)))
    tunable = config.get_model_tuning_info()

    assert len(tunable) == 1
    assert tunable[0].model_hyperparams_field_name == "xgboost_hyperparams"
    assert "n_estimators" in tunable[0].search_space


def test_tuning_config_mixin__returns_empty_when_no_tunable_ranges() -> None:
    """get_model_tuning_info returns an empty list when no TunableHyperParams field has tune=True."""
    assert _config().get_model_tuning_info() == []


# TuningResult


@pytest.mark.parametrize(
    ("best_params", "expected_repr"),
    [
        ({"alpha": 0.5, "n_iter": 42}, "TuningResult(2 params tuned)"),
        ({}, "TuningResult(no tuning)"),
    ],
)
def test_tuning_result__repr_reflects_param_count(best_params: dict[str, Any], expected_repr: str) -> None:
    """TuningResult.__repr__ summarises the count of tuned parameters."""
    # Arrange
    study = MagicMock(spec=optuna.Study)
    study.best_params = best_params
    result = TuningResult(
        workflow=MagicMock(),
        fit_result=None,
        study=study,
        best_config=MagicMock(),
    )

    # Act & Assert
    assert repr(result) == expected_repr


# run_optuna_study


@pytest.mark.parametrize("n_trials", [1, 3])
def test_run_optuna_study__runs_exactly_n_trials(n_trials: int) -> None:
    """run_optuna_study produces a study containing exactly n_trials completed trials."""

    # Arrange
    def objective(trial: optuna.Trial) -> float:
        return trial.suggest_float("x", 0.0, 1.0)

    # Act
    study = run_optuna_study(objective, n_trials=n_trials, seed=0)

    # Assert
    assert isinstance(study, optuna.Study)
    assert len(study.trials) == n_trials


@pytest.mark.parametrize(
    ("direction", "expected_direction_name"),
    [
        ("maximize", "MAXIMIZE"),
        ("minimize", "MINIMIZE"),
    ],
)
def test_run_optuna_study__respects_direction(direction: str, expected_direction_name: str) -> None:
    """run_optuna_study creates a study with the specified optimisation direction."""

    # Arrange
    def objective(trial: optuna.Trial) -> float:
        return trial.suggest_float("x", 0.0, 1.0)

    # Act
    study = run_optuna_study(objective, n_trials=1, seed=0, direction=direction)  # type: ignore[arg-type]

    # Assert
    assert study.direction.name == expected_direction_name


# _reconstruct_best_config


def test_reconstruct_best_config__single_model_applies_best_params() -> None:
    """_reconstruct_best_config updates the single hyperparams field with best_params values."""
    config = _config()
    info = ModelTuningInfo(
        model_hyperparams_field_name="xgboost_hyperparams",
        tunable_hyperparams=config.xgboost_hyperparams,
        search_space={"n_estimators": IntRange(50, 500, tune=True)},
    )
    study = MagicMock(spec=optuna.Study)
    study.best_params = {"n_estimators": 300}

    best_config = _reconstruct_best_config(config, [info], study)

    assert best_config.xgboost_hyperparams.n_estimators == 300
    assert best_config.xgboost_hyperparams.max_depth == config.xgboost_hyperparams.max_depth  # unchanged


def test_reconstruct_best_config__multi_model_parses_dotted_trial_keys() -> None:
    """_reconstruct_best_config routes 'field.param' keys to the correct HP group."""
    config = _config()
    info_xgb = ModelTuningInfo(
        model_hyperparams_field_name="xgboost_hyperparams",
        tunable_hyperparams=config.xgboost_hyperparams,
        search_space={"n_estimators": IntRange(50, 500, tune=True)},
    )
    info_gblinear = ModelTuningInfo(
        model_hyperparams_field_name="gblinear_hyperparams",
        tunable_hyperparams=config.gblinear_hyperparams,
        search_space={"n_estimators": IntRange(50, 500, tune=True)},
    )
    study = MagicMock(spec=optuna.Study)
    study.best_params = {"xgboost_hyperparams.n_estimators": 300, "gblinear_hyperparams.n_estimators": 150}

    best_config = _reconstruct_best_config(config, [info_xgb, info_gblinear], study)

    assert best_config.xgboost_hyperparams.n_estimators == 300
    assert best_config.gblinear_hyperparams.n_estimators == 150


# tune


def test_tune__raises_when_no_tunable_hyperparams() -> None:
    """tune raises ValueError when the config exposes no tune=True hyperparameter ranges."""
    with pytest.raises(ValueError, match="No tunable hyperparameters"):
        tune(_config(), MagicMock(), MagicMock())


def test_tune__returns_best_config_study_and_params() -> None:
    """tune returns a best_config, completed study, and best_params dict after n_trials trials."""
    config = _config(xgboost_hyperparams=XGBoostHyperParams(n_estimators=IntRange(50, 500, tune=True)))
    create_workflow = MagicMock(return_value=_make_mock_workflow())

    best_config, study, best_params = tune(config, MagicMock(), create_workflow)

    assert isinstance(best_config, ForecastingWorkflowConfig)
    assert isinstance(study, optuna.Study)
    assert len(study.trials) == 2
    assert "n_estimators" in best_params
    assert 50 <= best_params["n_estimators"] <= 500


# fit_with_tuning


def test_fit_with_tuning__returns_tuning_result_with_best_config_and_study() -> None:
    """fit_with_tuning returns a TuningResult holding the best config, study, and fitted workflow."""
    config = _config(xgboost_hyperparams=XGBoostHyperParams(n_estimators=IntRange(50, 500, tune=True)))
    create_workflow = MagicMock(return_value=_make_mock_workflow())

    result = fit_with_tuning(config, MagicMock(), create_workflow)

    assert isinstance(result, TuningResult)
    assert isinstance(result.study, optuna.Study)
    assert isinstance(result.best_config, ForecastingWorkflowConfig)
    assert result.workflow is not None
    # n_trials training runs during tuning + 1 final fit with the best config
    assert create_workflow.call_count == 2 + 1
