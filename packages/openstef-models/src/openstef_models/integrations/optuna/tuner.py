# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Hyperparameter tuner and supporting types."""

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


class _SearchSpaceEntry(NamedTuple):
    """Maps an Optuna trial key back to its source HyperParams field and parameter."""

    config_field: str
    param_name: str
    range: TuningRange


def _suggest_value(
    trial: optuna.Trial,
    trial_key: str,
    tuning_range: TuningRange,
) -> bool | int | float | str | None:
    """Suggest a value for *trial_key* using the appropriate Optuna API.

    Returns:
        Suggested value, or ``None`` when the range is incomplete.
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

    Useful for custom objectives outside of ``HyperparameterTuner``.

    Returns:
        Copy of *current* with trial-suggested values applied.
    """
    updates: dict[str, Any] = {}
    for param_name, tuning_range in space.items():
        value = _suggest_value(trial, param_name, tuning_range)
        if value is not None:
            updates[param_name] = value
    return current.model_copy(update=updates)


@dataclass(repr=False)
class TuningResult:
    """Result container for ``HyperparameterTuner.fit_with_tuning``."""

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


class HyperparameterTuner[ConfigT: BaseConfig](BaseConfig):
    """Bayesian hyperparameter tuner powered by Optuna.

    Orchestrates an Optuna study over the tunable search spaces declared on
    ``HyperParams`` fields of the provided *config*.  Override methods
    prefixed with ``_`` to customise study creation, objective evaluation,
    or parameter suggestion.
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

    def _discover_search_spaces(self) -> list[ModelTuningInfo]:
        """Scan *config* for ``HyperParams`` fields with non-empty search spaces.

        Returns:
            One ``ModelTuningInfo`` per tunable ``HyperParams`` field.
        """
        result: list[ModelTuningInfo] = []
        for field_name in type(self.config).model_fields:
            value = getattr(self.config, field_name)
            if isinstance(value, HyperParams):
                space = value.get_search_space()
                if space:
                    result.append(ModelTuningInfo(field_name=field_name, hyperparams=value, search_space=space))
        return result

    def _build_combined_space(  # noqa: PLR6301
        self, model_tuning_info: list[ModelTuningInfo]
    ) -> dict[str, _SearchSpaceEntry]:
        """Merge per-field search spaces into a single dict keyed by Optuna trial key.

        Returns:
            Combined search space mapping trial keys to ``_SearchSpaceEntry``.
        """
        multi = len(model_tuning_info) > 1
        combined: dict[str, _SearchSpaceEntry] = {}
        for info in model_tuning_info:
            for param_name, tuning_range in info.search_space.items():
                trial_key = f"{info.field_name}.{param_name}" if multi else param_name
                combined[trial_key] = _SearchSpaceEntry(info.field_name, param_name, tuning_range)
        return combined

    def _create_study(self) -> optuna.Study:
        """Create and configure the Optuna study.

        Override to use a different sampler, pruner, or storage backend.

        Returns:
            Configured ``optuna.Study`` ready for optimisation.
        """
        return optuna.create_study(
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            study_name=self.study_name,
        )

    def _suggest_value(  # noqa: PLR6301
        self,
        trial: optuna.Trial,
        trial_key: str,
        tuning_range: TuningRange,
    ) -> bool | int | float | str | None:
        """Suggest a single parameter value via the Optuna trial API.

        Override to add custom suggestion logic or constraints.

        Returns:
            Suggested value, or ``None`` when the range is incomplete.
        """
        return _suggest_value(trial, trial_key, tuning_range)

    def _evaluate_trial(
        self,
        trial: optuna.Trial,
        combined_space: dict[str, _SearchSpaceEntry],
        model_tuning_info: list[ModelTuningInfo],
    ) -> float:
        """Score a single Optuna trial.

        Override to change how trials are evaluated (e.g. cross-validation).

        Returns:
            Metric score for the trial (lower / higher is better depending on *direction*).

        Raises:
            ValueError: If ``metric_name`` is not found in the evaluation metrics.
        """
        per_field: dict[str, dict[str, Any]] = {}
        for trial_key, entry in combined_space.items():
            value = self._suggest_value(trial, trial_key, entry.range)
            if value is not None:
                per_field.setdefault(entry.config_field, {})[entry.param_name] = value

        tuned_config = self.config.model_copy(
            update={
                info.field_name: info.hyperparams.model_copy(update=per_field[info.field_name])
                for info in model_tuning_info
                if info.field_name in per_field
            }
        )
        workflow = self.create_workflow(tuned_config)
        fit_result = workflow.fit(self.train_dataset)
        if fit_result is None:
            return float("-inf")
        metrics = fit_result.metrics_val if fit_result.metrics_val is not None else fit_result.metrics_train
        score = metrics.get_metric(quantile=self.target_quantile, metric_name=self.metric_name)
        if score is None:
            available = metrics.to_flat_dict()
            msg = (
                f"Metric {self.metric_name!r} (quantile={self.target_quantile!r}) not found. "
                f"Available metrics: {sorted(available)}"
            )
            raise ValueError(msg)
        return float(score)

    def _reconstruct_best_config(
        self,
        model_tuning_info: list[ModelTuningInfo],
        study: optuna.Study,
    ) -> ConfigT:
        """Apply the best trial params back to the original config.

        Returns:
            Config copy with the best trial's parameters applied.
        """
        multi = len(model_tuning_info) > 1
        per_field_best: dict[str, dict[str, Any]] = {}
        for trial_key, value in study.best_params.items():
            if multi and "." in trial_key:
                field_name, param_name = trial_key.split(".", 1)
            else:
                field_name = model_tuning_info[0].field_name
                param_name = trial_key
            per_field_best.setdefault(field_name, {})[param_name] = value
        return self.config.model_copy(
            update={
                info.field_name: info.hyperparams.model_copy(update=per_field_best[info.field_name])
                for info in model_tuning_info
                if info.field_name in per_field_best
            }
        )

    def tune(self) -> tuple[ConfigT, optuna.Study, dict[str, Any]]:
        """Run the Optuna study and return the best config.

        Returns:
            ``(best_config, study, best_params)`` tuple.

        Raises:
            ValueError: If no tunable fields are found.
        """
        model_tuning_info = self._discover_search_spaces()
        if not model_tuning_info:
            msg = "No tunable hyperparameters found. Pass TuningRange(tune=True) in the HyperParams constructor."
            raise ValueError(msg)

        combined_space = self._build_combined_space(model_tuning_info)
        study = self._create_study()

        def objective(trial: optuna.Trial) -> float:
            return self._evaluate_trial(trial, combined_space, model_tuning_info)

        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        best_config = self._reconstruct_best_config(model_tuning_info, study)
        return best_config, study, study.best_params

    def fit_with_tuning(self) -> TuningResult:
        """Tune, then fit a final workflow with the best config.

        Returns:
            ``TuningResult`` with the fitted workflow, study, and best config.
        """
        best_config, study, _ = self.tune()
        workflow = self.create_workflow(best_config)
        result = workflow.fit(self.train_dataset)
        return TuningResult(workflow=workflow, fit_result=result, study=study, best_config=best_config)


__all__ = [
    "HyperparameterTuner",
    "TuningResult",
    "apply_trial_suggestions",
]
