# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""Hyperparameter tuning via Optuna.

Requires ``optuna`` (available via ``pip install openstef-models[tuning]``).
Import this module only when tuning is needed.

Key public API:
- ``HyperparameterTuner`` — Pydantic model that orchestrates Bayesian tuning.
- ``run_optuna_study`` — standalone utility for custom objectives.
- ``TuningResult`` — result container for tuned config + study.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

from pydantic import ConfigDict, Field, SkipValidation

try:
    import optuna
except ImportError as _err:
    from openstef_core.exceptions import MissingExtraError

    raise MissingExtraError("optuna", "openstef-models[tuning]") from _err

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.mixins.predictor import HyperParams
from openstef_core.param_ranges import (
    CategoricalRange,
    FloatRange,
    IntRange,
    ModelTuningInfo,
    TuningRange,
)
from openstef_core.types import QuantileOrGlobal
from openstef_models.models.forecasting_model import ModelFitResult
from openstef_models.workflows.custom_forecasting_workflow import CustomForecastingWorkflow


def _get_model_tuning_info(config: BaseConfig) -> list[ModelTuningInfo]:
    """Return one ``ModelTuningInfo`` per ``HyperParams`` field with a non-empty search space.

    Args:
        config: Configuration to introspect for tunable hyperparameter fields.
    """
    result: list[ModelTuningInfo] = []
    for field_name in type(config).model_fields:
        value = getattr(config, field_name)
        if isinstance(value, HyperParams):
            space = value.get_search_space()
            if space:
                result.append(ModelTuningInfo(field_name=field_name, hyperparams=value, search_space=space))
    return result


class _TrialEntry(NamedTuple):
    """One entry in the combined search space, keyed by Optuna trial-key."""

    model_hyperparams_field_name: str
    hyperparam_name: str
    tuning_range: TuningRange


def _suggest_hyperparam_value(
    trial: optuna.Trial,
    trial_key: str,
    tuning_range: TuningRange,
) -> bool | int | float | str | None:
    """Suggest a value for *trial_key* using the appropriate Optuna API.

    Returns:
        Suggested value, or ``None`` when the range has missing bounds.
    """
    if isinstance(tuning_range, FloatRange) and tuning_range.low is not None and tuning_range.high is not None:
        return trial.suggest_float(trial_key, tuning_range.low, tuning_range.high, log=tuning_range.log)
    if isinstance(tuning_range, IntRange) and tuning_range.low is not None and tuning_range.high is not None:
        return trial.suggest_int(trial_key, tuning_range.low, tuning_range.high, log=tuning_range.log)
    if isinstance(tuning_range, CategoricalRange) and tuning_range.choices is not None:
        return trial.suggest_categorical(trial_key, list(tuning_range.choices))
    return None


def apply_trial_suggestions[HP: BaseConfig](
    trial: optuna.Trial,
    space: dict[str, TuningRange],
    current: HP,
) -> HP:
    """Create an updated config by applying Optuna trial suggestions.

    Returns:
        Copy of *current* with suggested values applied.
    """
    updates: dict[str, Any] = {}
    for hyperparam_name, tuning_range in space.items():
        value = _suggest_hyperparam_value(trial, hyperparam_name, tuning_range)
        if value is not None:
            updates[hyperparam_name] = value
    return current.model_copy(update=updates)


def run_optuna_study(
    objective: Callable[[optuna.Trial], float],
    n_trials: int,
    seed: int | None = 42,
    direction: Literal["maximize", "minimize"] = "maximize",
    study_name: str = "hyperparameter_tuning",
) -> optuna.Study:
    """Run a Bayesian hyperparameter optimisation study.

    Returns:
        Completed ``optuna.Study`` with all trial results.
    """
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        study_name=study_name,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


def _build_hp_updates(
    model_tuning_info: list[ModelTuningInfo],
    per_field: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build a config-level update dict by applying suggested values per HP group.

    Returns:
        Mapping of config field name to updated ``HyperParams`` instance.
    """
    return {
        tf.field_name: tf.hyperparams.model_copy(update=per_field[tf.field_name])
        for tf in model_tuning_info
        if tf.field_name in per_field
    }


class _TuningObjective(BaseConfig):
    """Callable Optuna objective encapsulating the tuning context."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    combined_space: SkipValidation[dict[str, _TrialEntry]] = Field(
        description="Combined search space keyed by Optuna trial-key."
    )
    model_tuning_info: list[ModelTuningInfo] = Field(description="Tuning metadata per HyperParams field.")
    config: SkipValidation[BaseConfig] = Field(description="Base configuration to permute per trial.")
    train_dataset: SkipValidation[TimeSeriesDataset] = Field(description="Training data for each trial.")
    create_workflow: SkipValidation[Callable[..., CustomForecastingWorkflow]] = Field(
        description="Factory that builds a workflow from a config."
    )
    target_quantile: QuantileOrGlobal = Field(description="Quantile (or 'global') for metric evaluation.")
    metric_name: str = Field(description="Name of the metric to optimise.")

    def __call__(self, trial: optuna.Trial) -> float:
        """Evaluate a single Optuna trial.

        Returns:
            Score to optimise, or ``-inf`` on failure.
        """
        per_field: dict[str, dict[str, Any]] = {}
        for trial_key, trial_entry in self.combined_space.items():
            value = _suggest_hyperparam_value(trial, trial_key, trial_entry.tuning_range)
            if value is not None:
                per_field.setdefault(trial_entry.model_hyperparams_field_name, {})[trial_entry.hyperparam_name] = value

        tuned_config = self.config.model_copy(update=_build_hp_updates(self.model_tuning_info, per_field))
        trial_workflow = self.create_workflow(tuned_config)
        trial_result = trial_workflow.fit(self.train_dataset)
        if trial_result is None:
            return float("-inf")
        metrics = trial_result.metrics_val if trial_result.metrics_val is not None else trial_result.metrics_train
        score = metrics.get_metric(quantile=self.target_quantile, metric_name=self.metric_name)
        return float(score) if score is not None else float("-inf")


@dataclass(repr=False)
class TuningResult:
    """Result container for ``fit_with_tuning``."""

    workflow: CustomForecastingWorkflow
    fit_result: ModelFitResult | None
    study: optuna.Study
    best_config: BaseConfig

    def __repr__(self) -> str:
        """Show tuned parameter count for quick inspection.

        Returns:
            Human-readable summary like ``TuningResult(3 params tuned)``.
        """
        n = len(self.study.best_params)
        return f"TuningResult({n} params tuned)" if n else "TuningResult(no tuning)"


def _reconstruct_best_config[ConfigT: BaseConfig](
    config: ConfigT,
    model_tuning_info_list: list[ModelTuningInfo],
    study: optuna.Study,
) -> ConfigT:
    """Apply the best trial params back to *config*.

    Returns:
        Updated config with best hyperparameter values.
    """
    multi = len(model_tuning_info_list) > 1
    per_field_best: dict[str, dict[str, Any]] = {}
    for trial_key, value in study.best_params.items():
        if multi and "." in trial_key:
            field_name, hyperparam_name = trial_key.split(".", 1)
        else:
            field_name = model_tuning_info_list[0].field_name
            hyperparam_name = trial_key
        per_field_best.setdefault(field_name, {})[hyperparam_name] = value
    return config.model_copy(update=_build_hp_updates(model_tuning_info_list, per_field_best))


class HyperparameterTuner[ConfigT: BaseConfig](BaseConfig):
    """Bayesian hyperparameter tuner powered by Optuna.

    Orchestrates an Optuna study over the tunable search spaces declared on
    ``HyperParams`` fields of the provided *config*.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: SkipValidation[ConfigT] = Field(description="Configuration with tunable HyperParams fields.")
    train_dataset: SkipValidation[TimeSeriesDataset] = Field(description="Training data for each trial.")
    create_workflow: SkipValidation[Callable[[ConfigT], CustomForecastingWorkflow]] = Field(
        description="Factory that builds a workflow from a config."
    )
    target_quantile: QuantileOrGlobal = Field(description="Quantile (or 'global') for metric evaluation.")
    metric_name: str = Field(description="Name of the metric to optimise.")
    direction: Literal["maximize", "minimize"] = Field(
        default="maximize", description="Optimisation direction for the metric."
    )
    n_trials: int = Field(default=20, description="Number of Optuna trials.")
    seed: int | None = Field(default=42, description="Random seed for reproducibility.")
    study_name: str = Field(default="hyperparameter_tuning", description="Optuna study name.")

    def tune(self) -> tuple[ConfigT, optuna.Study, dict[str, Any]]:
        """Run the Optuna study and return the best config.

        Returns:
            ``(best_config, study, best_params)`` tuple.

        Raises:
            ValueError: If no tunable fields are found.
        """
        model_tuning_info = _get_model_tuning_info(self.config)
        if not model_tuning_info:
            msg = "No tunable hyperparameters found. Pass TuningRange(tune=True) in the HyperParams constructor."
            raise ValueError(msg)

        multi = len(model_tuning_info) > 1
        combined_space: dict[str, _TrialEntry] = {}
        for tf in model_tuning_info:
            for hp_name, tr in tf.search_space.items():
                trial_key = f"{tf.field_name}.{hp_name}" if multi else hp_name
                combined_space[trial_key] = _TrialEntry(tf.field_name, hp_name, tr)

        objective = _TuningObjective(
            combined_space=combined_space,
            model_tuning_info=model_tuning_info,
            config=self.config,
            train_dataset=self.train_dataset,
            create_workflow=self.create_workflow,
            target_quantile=self.target_quantile,
            metric_name=self.metric_name,
        )
        study = run_optuna_study(
            objective=objective,
            n_trials=self.n_trials,
            seed=self.seed,
            direction=self.direction,
            study_name=self.study_name,
        )
        best_config = _reconstruct_best_config(self.config, model_tuning_info, study)
        return best_config, study, study.best_params

    def fit_with_tuning(self) -> TuningResult:
        """Tune, then fit a final workflow with the best config.

        Returns:
            ``TuningResult`` containing the fitted workflow, study, and best config.
        """
        best_config, study, _ = self.tune()
        workflow = self.create_workflow(best_config)
        result = workflow.fit(self.train_dataset)
        return TuningResult(workflow=workflow, fit_result=result, study=study, best_config=best_config)


__all__ = [
    "HyperparameterTuner",
    "TuningResult",
    "apply_trial_suggestions",
    "run_optuna_study",
]
