# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Hyperparameter tuner and supporting types."""

from collections import defaultdict
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
from openstef_core.mixins.param_ranges import (
    CategoricalRange,
    FloatRange,
    IntRange,
    ModelTuningInfo,
    TuningRange,
)
from openstef_core.mixins.predictor import HyperParams
from openstef_core.types import QuantileOrGlobal
from openstef_models.workflows.custom_forecasting_workflow import CustomForecastingWorkflow


class _SearchSpaceEntry(NamedTuple):
    """Maps an Optuna trial key back to its source HyperParams field and parameter."""

    config_field: str
    param_name: str
    range: TuningRange


def _collect_available_metric_names(config: BaseConfig) -> set[str] | None:
    """Extract declared metric names from *config* if it has ``evaluation_metrics``.

    Returns:
        Metric name set, or ``None`` when the config doesn't expose metric providers.
    """
    providers = getattr(config, "evaluation_metrics", None)
    if providers is None:
        return None

    names: set[str] = set()
    for provider in providers:
        if hasattr(provider, "metric_names"):
            names.update(provider.metric_names)
    return names or None


@dataclass(repr=False)
class TuningResult[ConfigT: BaseConfig]:
    """Result container for ``HyperparameterTuner.fit_with_tuning``."""

    best_config: ConfigT
    study: optuna.Study

    def __repr__(self) -> str:  # noqa: D105  # self-explanatory
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
    n_jobs: int = Field(
        default=1,
        description="Number of parallel trial workers. Set >1 when the underlying model is single-threaded.",
    )
    seed: int | None = Field(default=42, description="Random seed for reproducibility.")
    study_name: str = Field(default="hyperparameter_tuning", description="Optuna study name.")

    def _discover_search_spaces(self) -> list[ModelTuningInfo]:
        """Scan *config* for ``HyperParams`` fields with non-empty search spaces.

        Returns:
            One ``ModelTuningInfo`` per tunable ``HyperParams`` field.
        """
        return [
            ModelTuningInfo(field_name=field_name, hyperparams=value, search_space=space)
            for field_name in type(self.config).model_fields
            if isinstance(value := getattr(self.config, field_name), HyperParams)
            and (space := value.get_search_space())
        ]

    @staticmethod
    def _build_combined_space(model_tuning_info: list[ModelTuningInfo]) -> dict[str, _SearchSpaceEntry]:
        """Merge per-field search spaces into a single dict keyed by Optuna trial key.

        When multiple HyperParams groups are tuned, trial keys are prefixed with the
        field name (e.g. ``"xgboost_hyperparams.learning_rate"``) to avoid collisions.

        Returns:
            Combined search space mapping trial keys to ``_SearchSpaceEntry``.
        """
        multi = len(model_tuning_info) > 1
        return {
            (f"{info.field_name}.{param_name}" if multi else param_name): _SearchSpaceEntry(
                info.field_name, param_name, tuning_range
            )
            for info in model_tuning_info
            for param_name, tuning_range in info.search_space.items()
        }

    @property
    def _trial_base(self) -> ConfigT:
        """Config with tracking disabled so each trial trains from scratch.

        Strips ``mlflow_storage`` when present — without this, the MLflow callback
        can reuse a previously stored model and short-circuit training, making trials
        non-comparable and invalidating the optimisation.

        Override to disable additional callbacks or change trial-specific settings.

        Returns:
            Config copy suitable for use inside ``_evaluate_trial``.
        """
        if hasattr(self.config, "mlflow_storage"):
            return self.config.model_copy(update={"mlflow_storage": None})  # type: ignore[return-value]
        return self.config

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

    @staticmethod
    def suggest_value(
        trial: optuna.Trial,
        trial_key: str,
        tuning_range: TuningRange,
    ) -> bool | int | float | str | None:
        """Suggest a single parameter value via the Optuna trial API.

        Override to add custom suggestion logic or constraints.

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

    def _evaluate_trial(
        self,
        trial: optuna.Trial,
        combined_space: dict[str, _SearchSpaceEntry],
        model_tuning_info: list[ModelTuningInfo],
    ) -> float:
        """Score a single Optuna trial.

        Suggests HP values, builds a config copy, fits a workflow, and extracts the
        configured metric.  Override to change how trials are evaluated (e.g.
        cross-validation).

        Returns:
            Metric score for the trial (lower / higher is better depending on *direction*).

        Raises:
            ValueError: If ``metric_name`` is not found in the evaluation metrics.
        """
        # Group suggested values by their owning HyperParams field
        per_field: dict[str, dict[str, Any]] = defaultdict(dict)
        for trial_key, entry in combined_space.items():
            value = self.suggest_value(trial, trial_key, entry.range)
            if value is not None:
                per_field[entry.config_field][entry.param_name] = value

        # Replace each HyperParams instance with a copy containing the trial's suggestions
        tuned_config = self._trial_base.model_copy(
            update={
                info.field_name: info.hyperparams.model_copy(update=per_field[info.field_name])
                for info in model_tuning_info
                if info.field_name in per_field
            }
        )

        # Create a workflow and train from the trial config
        workflow = self.create_workflow(tuned_config)
        fit_result = workflow.fit(self.train_dataset)
        if fit_result is None:
            return float("-inf") if self.direction == "maximize" else float("inf")

        # Prefer validation metrics; fall back to training metrics
        metrics = fit_result.metrics_val if fit_result.metrics_val is not None else fit_result.metrics_train
        score = metrics.get_metric(quantile=self.target_quantile, metric_name=self.metric_name)
        if score is None:
            available = sorted(metrics.to_flat_dict())
            msg = (
                f"Metric {self.metric_name!r} (quantile={self.target_quantile!r}) not found. "
                f"Available metrics: {available}"
            )
            raise ValueError(msg)
        return float(score)

    @staticmethod
    def _reconstruct_best_config(
        config: BaseConfig,
        model_tuning_info: list[ModelTuningInfo],
        study: optuna.Study,
    ) -> BaseConfig:
        """Apply the best trial params back to the original config.

        Returns:
            Config copy with the best trial's parameters applied.
        """
        multi = len(model_tuning_info) > 1
        per_field: dict[str, dict[str, Any]] = defaultdict(dict)
        for trial_key, value in study.best_params.items():
            if multi and "." in trial_key:
                field_name, param_name = trial_key.split(".", 1)
            else:
                # Single-model shortcut — params aren't prefixed
                field_name = model_tuning_info[0].field_name
                param_name = trial_key
            per_field[field_name][param_name] = value

        return config.model_copy(
            update={
                info.field_name: info.hyperparams.model_copy(update=per_field[info.field_name])
                for info in model_tuning_info
                if info.field_name in per_field
            }
        )

    def _validate_metric_name(self) -> None:
        """Eagerly check that ``metric_name`` is valid for the configured evaluation metrics.

        Only performs the check when the *config* exposes ``evaluation_metrics``
        with providers that declare ``metric_names``.  Silently skips otherwise
        (validation will still occur at trial time).

        Raises:
            ValueError: If *metric_name* is not among the declared provider metric names.
        """
        known = _collect_available_metric_names(self.config)
        if known is not None and self.metric_name not in known:
            msg = (
                f"Metric {self.metric_name!r} is not provided by the configured evaluation_metrics. "
                f"Available: {sorted(known)}"
            )
            raise ValueError(msg)

    def tune(
            self,
        ) -> tuple[ConfigT, optuna.Study]:
        """Run the Optuna study and return the best config.

        Returns:
            ``(best_config, study)`` tuple.

        Raises:
            ValueError: If no tunable fields are found or *metric_name* is invalid.
        """
        self._validate_metric_name()

        model_tuning_info = self._discover_search_spaces()
        if not model_tuning_info:
            msg = "No tunable hyperparameters found. Pass TuningRange(tune=True) in the HyperParams constructor."
            raise ValueError(msg)

        combined_space = self._build_combined_space(model_tuning_info)
        study = self._create_study()

        def objective(trial: optuna.Trial) -> float:
            return self._evaluate_trial(trial, combined_space, model_tuning_info)

        study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs, show_progress_bar=True)
        best_config = self._reconstruct_best_config(self.config, model_tuning_info, study)
        return best_config, study  # type: ignore[return-value]  # ConfigT narrowing not expressible

    def fit_with_tuning(self) -> TuningResult[ConfigT]:
        """Tune, then fit a final workflow with the best config.

        Returns:
            ``TuningResult`` with the best config and Optuna study.
        """
        best_config, study = self.tune()
        return TuningResult(best_config=best_config, study=study)


__all__ = [
    "HyperparameterTuner",
    "TuningResult",
]
