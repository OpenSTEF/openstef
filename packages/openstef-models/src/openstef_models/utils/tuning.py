# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""Hyperparameter tuning utilities for OpenSTEF models.

Provides dataclasses for describing hyperparameter search spaces, helper functions to
extract and merge search spaces from annotated HyperParams classes, and a thin wrapper
around Optuna for running Bayesian hyperparameter optimisation studies.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, Self, cast, runtime_checkable

import optuna
from pydantic import BaseModel, PrivateAttr, model_validator

from openstef_core.mixins import HyperParams

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic.fields import FieldInfo

    from openstef_core.datasets import TimeSeriesDataset
    from openstef_core.types import QuantileOrGlobal
    from openstef_models.mixins.model_serializer import ModelIdentifier
    from openstef_models.models.forecasting_model import ModelFitResult
    from openstef_models.workflows.custom_forecasting_workflow import CustomForecastingWorkflow


@dataclass(frozen=True)
class FloatRange:
    """Search space metadata for continuous float hyperparameters.

    Attach to a ``HyperParams`` field via ``Annotated`` to declare the
    range that a hyperparameter tuner should explore.

    Args:
        low: Lower bound of the search interval (inclusive).
        high: Upper bound of the search interval (inclusive).
        log: When ``True`` the sampler draws on a log scale, which is
            recommended for parameters like learning rates and
            regularisation coefficients.

    Example:
        >>> learning_rate: Annotated[float, FloatRange(0.01, 0.5, log=True)] = 0.3
    """

    low: float | None
    high: float | None
    log: bool = False
    tune: bool = False


@dataclass(frozen=True)
class IntRange:
    """Search space metadata for discrete integer hyperparameters.

    Attach to a ``HyperParams`` field via ``Annotated`` to declare the
    integer range that a hyperparameter tuner should explore.

    Args:
        low: Minimum integer value (inclusive).
        high: Maximum integer value (inclusive).
        log: When ``True`` the sampler draws on a log scale.

    Example:
        >>> n_estimators: Annotated[int, IntRange(50, 500)] = 100
    """

    low: int | None
    high: int | None
    log: bool = False
    tune: bool = False


@dataclass(frozen=True)
class CategoricalRange:
    """Search space metadata for categorical hyperparameters.

    Attach to a ``HyperParams`` field via ``Annotated`` to list the
    discrete choices that a hyperparameter tuner should explore.

    Args:
        choices: Tuple of allowed values for the parameter.

    Example:
        >>> tree_method: Annotated[str, CategoricalRange(("hist", "approx"))] = "hist"
    """

    choices: tuple[Any, ...] | None
    tune: bool = False


#: Union alias for any single-parameter search space descriptor.
TuningRange = FloatRange | IntRange | CategoricalRange


class TunableHyperParams(HyperParams):
    """HyperParams subclass that accepts ``TuningRange`` objects as field values.

    Pass a :class:`FloatRange`, :class:`IntRange`, or :class:`CategoricalRange` as the
    value for any field during construction.  The range is stored in the private
    ``_instance_ranges`` attribute and the field itself keeps its declared default value.
    ``None`` for ``low`` / ``high`` / ``choices`` falls back to the class-level
    ``Annotated`` metadata when the search space is resolved.

    This means the tuning search space lives **on the HyperParams instance itself** — no
    separate dict is needed.

    Example::

        hp = XGBoostHyperParams(
            n_estimators=IntRange(100, 800, tune=True),
            learning_rate=FloatRange(None, None, log=True, tune=True),   # → class default [0.01, 0.5]
        )
        # hp.n_estimators == 100  (the class default; the IntRange was extracted)
        # get_search_space(hp) → {'n_estimators': IntRange(100, 800), 'learning_rate': FloatRange(0.01, 0.5)}
    """

    _instance_ranges: dict[str, TuningRange] = PrivateAttr(  # pyright: ignore[reportUnknownVariableType,reportIncompatibleVariableOverride]
        default_factory=dict
    )

    @property
    def instance_ranges(self) -> dict[str, TuningRange]:
        """Public view of the per-instance tuning ranges extracted at construction."""
        return self._instance_ranges

    @model_validator(mode="wrap")
    @classmethod
    def _extract_tuning_ranges(
        cls,
        data: dict[str, object] | object,
        handler: Callable[[dict[str, object] | object], TunableHyperParams],
    ) -> TunableHyperParams:
        """Strip TuningRange values from the input dict and store them as instance metadata.

        Returns:
            A new :class:`TunableHyperParams` instance with TuningRange values removed
            from the fields and stored in the private ``_instance_ranges`` attribute.
        """
        instance_ranges: dict[str, TuningRange] = {}
        if isinstance(data, dict):
            cleaned: dict[str, Any] = {}
            for key, value in cast("dict[str, object]", data).items():
                if isinstance(value, (FloatRange, IntRange, CategoricalRange)):
                    instance_ranges[key] = value
                    # Keep the key absent: Pydantic uses the declared field default
                else:
                    cleaned[key] = value
            data = cleaned
        result: TunableHyperParams = handler(data)
        if instance_ranges and result.__pydantic_private__ is not None:
            result._instance_ranges = instance_ranges
        return result


@dataclass(frozen=True)
class ModelTuningInfo:
    """Dataclass for model specific hyperparameter info.

    Ensures that search_space cannot be empty.

    Attributes:
        model_hyperparams_field_name: Name of the field on the config object
            (e.g. ``"xgboost_hyperparams"``).
        tunable_hyperparams: The ``TunableHyperParams`` instance to update with
            trial suggestions.
        search_space: Pre-computed, non-empty mapping of
            parameter name → :class:`TuningRange`.
    """

    model_hyperparams_field_name: str
    tunable_hyperparams: TunableHyperParams
    search_space: dict[str, TuningRange]

    def __post_init__(self) -> None:
        """Validate that search_space is non-empty.

        Raises:
            ValueError: If ``search_space`` is empty.
        """
        if not self.search_space:
            msg = (
                f"search_space for '{self.model_hyperparams_field_name}' must not be empty. "
                "Pass TuningRange(tune=True) objects in the HyperParams constructor."
            )
            raise ValueError(msg)


@runtime_checkable
class TunableWorkflowConfig(Protocol):
    """Structural requirements for workflow configs for tuning.

    This protocol is used for type checking of different configs cross-package, for example
    ForecastingWorkflowConfig and EnsembleForecastingWorkflowConfig.
    """

    model_id: ModelIdentifier
    optuna_n_trials: int
    optuna_seed: int | None

    @property
    def model_selection_metric(self) -> tuple[QuantileOrGlobal, str, Any]:
        """Metric used to select the best trial: (quantile, metric_name, direction)."""
        ...

    def get_model_tuning_info(self) -> list[ModelTuningInfo]:
        """Return TunableField with model_hyperparams_field_name, hyperparams_instance and search_space for tuning.

        Can be inherited from TuningConfigMixin.
        """
        ...

    def model_copy(self, *, update: dict[str, Any]) -> Self:
        """Return a copy of the config with the given fields updated."""
        ...


def _get_class_range(field_info: FieldInfo) -> TuningRange | None:
    """Return the first TuningRange found in a Pydantic FieldInfo's metadata."""
    for meta in field_info.metadata:
        if isinstance(meta, (FloatRange, IntRange, CategoricalRange)):
            return meta
    return None


def _merge_numerical_range[T: (FloatRange, IntRange)](override: T, class_range: TuningRange | None) -> T:
    """Merge a FloatRange or IntRange override with the class-level default.

    Returns:
        A new instance of the same type as *override* with ``None`` bounds filled
        from *class_range*.
    """
    cr = class_range if isinstance(class_range, type(override)) else None
    low = override.low if override.low is not None else (cr.low if cr is not None else None)
    high = override.high if override.high is not None else (cr.high if cr is not None else None)
    return replace(override, low=low, high=high)


def _merge_categorical_range(override: CategoricalRange, class_range: TuningRange | None) -> CategoricalRange:
    """Merge a CategoricalRange override with the class-level CategoricalRange default.

    Returns:
        A new :class:`CategoricalRange` with ``None`` choices filled from *class_range*.
    """
    cr = class_range if isinstance(class_range, CategoricalRange) else None
    cr_choices = cr.choices if cr is not None else None
    choices = override.choices if override.choices is not None else cr_choices
    return CategoricalRange(choices=choices, tune=override.tune)


def _merge_range(override: TuningRange, class_range: TuningRange | None) -> TuningRange:
    """Merge *override* with *class_range*, filling ``None`` from the class defaults.

    For ``FloatRange`` / ``IntRange``, ``None`` values for ``low`` or ``high`` are filled
    in from *class_range*.  For ``CategoricalRange``, ``None`` for ``choices`` falls back
    to *class_range*.  The ``tune`` flag always comes from *override*.

    Returns:
        A new :class:`TuningRange` with ``None`` bounds merged from *class_range*.
    """
    if isinstance(override, FloatRange):
        return _merge_numerical_range(override, class_range)
    if isinstance(override, IntRange):
        return _merge_numerical_range(override, class_range)
    return _merge_categorical_range(override, class_range)


def get_search_space(
    hyperparams: BaseModel,
    include: set[str] | None = None,
) -> dict[str, TuningRange]:
    """Extract the effective tunable search space from a *HyperParams* instance.

    Reads per-instance ``TuningRange`` objects stored in ``_instance_ranges``
    (set by passing ranges directly in the constructor of a
    :class:`TunableHyperParams` subclass) and merges them with the class-level
    ``Annotated`` metadata.  ``None`` bounds fall back to the class-level defaults.
    Only fields where the resulting ``tune`` flag is ``True`` are included.

    Args:
        hyperparams: A :class:`TunableHyperParams` (or plain ``HyperParams``) instance.
        include: If given, restrict the output to exactly these field names.  A
            ``KeyError`` is raised immediately for any name that is absent or has no
            ``tune=True`` annotation (catches typos early).

    Returns:
        Mapping of hyperparam field-name → effective :class:`TuningRange` for all tunable fields.

    Raises:
        KeyError: If ``include`` is specified and any requested field name is not
            present in the tunable search space.

    Example::

        hp = XGBoostHyperParams(
            n_estimators=IntRange(100, 800, tune=True),
            learning_rate=FloatRange(None, None, log=True, tune=True),
        )
        space = get_search_space(hp)
        # {'n_estimators': IntRange(100, 800), 'learning_rate': FloatRange(0.01, 0.5, log=True)}
    """
    # Per-instance ranges take precedence over class-level annotations
    instance_ranges: dict[str, TuningRange] = {}
    if isinstance(hyperparams, TunableHyperParams):
        instance_ranges = hyperparams.instance_ranges

    result: dict[str, TuningRange] = {}
    for hyperparam_name, field_info in type(hyperparams).model_fields.items():
        class_range = _get_class_range(field_info)
        override = instance_ranges.get(hyperparam_name)

        if override is not None:
            if not override.tune:
                continue
            result[hyperparam_name] = _merge_range(override, class_range)
        elif class_range is not None and class_range.tune:
            result[hyperparam_name] = class_range

    if include is not None:
        missing = include - result.keys()
        if missing:
            msg = (
                f"Fields {sorted(missing)!r} not found in the tunable search space. "
                "Check that they exist on the HyperParams class and were passed as "
                "TuningRange(tune=True) in the constructor."
            )
            raise KeyError(msg)
        result = {k: result[k] for k in include}

    return result


def apply_trial_suggestions[HP: BaseModel](
    trial: optuna.Trial,
    space: dict[str, TuningRange],
    current: HP,
) -> HP:
    """Create an updated *HyperParams* using Optuna trial suggestions.

    Args:
        trial: Optuna trial object for suggesting values.
        space: Search space returned by :func:`get_search_space`.
        current: Current ``HyperParams`` instance to copy-and-update.

    Returns:
        A new ``HyperParams`` instance with the suggested values applied.
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
    """Run a Bayesian hyperparameter optimisation study using Optuna.

    Args:
        objective: Callable that receives an :class:`optuna.Trial` and returns a
            ``float`` score to optimise.
        n_trials: Number of trials to evaluate.
        seed: Random seed for the TPE sampler (``None`` disables seeding).
        direction: ``"maximize"`` or ``"minimize"``.
        study_name: Human-readable label for the study.

    Returns:
        Completed :class:`optuna.Study` with all trial results.
    """
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        study_name=study_name,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


class TuningConfigMixin:
    """Mixin for get_model_tuning_info for workflow configs.

    Discovers tunable fields by reflecting over model_fields and returning a TunableField for every field whose value
    is a TunableHyperParams instance with a non-empty search space.
    """

    def get_model_tuning_info(self) -> list[ModelTuningInfo]:
        """Return one ModelTuningInfo per active tunable hyperparameter group for a model."""
        result: list[ModelTuningInfo] = []
        model_fields: dict[str, Any] = cast(dict[str, Any], getattr(type(self), "model_fields", {}))
        for field_name in model_fields:
            value = getattr(self, field_name)
            if isinstance(value, TunableHyperParams):  # checks if the config field contains tunable hyperparams
                space = get_search_space(value)
                if space:
                    result.append(
                        ModelTuningInfo(
                            model_hyperparams_field_name=field_name,
                            tunable_hyperparams=value,
                            search_space=space,
                        )
                    )
        return result


@dataclass(repr=False)
class TuningResult:
    """Result of a :func:`fit_with_tuning` call.

    Attributes:
        workflow: The fitted :class:`CustomForecastingWorkflow`.
        fit_result: The :class:`ModelFitResult` from the final training run, or
            ``None`` if fitting was skipped (e.g. by an MLflow callback).
        study: The completed :class:`optuna.Study`.  Raw best parameter values
            are available via ``study.best_params``.
        best_config: The workflow config updated with the best hyperparameters
            found during tuning.
    """

    workflow: CustomForecastingWorkflow
    fit_result: ModelFitResult | None
    study: optuna.Study
    best_config: TunableWorkflowConfig

    def __repr__(self) -> str:
        """Return a string representation of the TuningResult."""
        n = len(self.study.best_params)
        return f"TuningResult({n} params tuned)" if n else "TuningResult(no tuning)"


class _TrialEntry(NamedTuple):
    """One entry in the combined search space, keyed by Optuna trial-key.

    Attributes:
        model_hyperparams_field_name: Field on the config holding the hyperparams
            group for a specific model, e.g. ``"xgboost_hyperparams"``.
        hyperparam_name: Individual parameter within that group, e.g. ``"n_estimators"``.
        tuning_range: Defines the search space for this parameter.
    """

    model_hyperparams_field_name: str
    hyperparam_name: str
    tuning_range: TuningRange


def _suggest_hyperparam_value(
    trial: optuna.Trial,
    trial_key: str,
    tuning_range: TuningRange,
) -> bool | int | float | str | None:
    """Suggest a value for *trial_key* using the appropriate Optuna API.

    Returns ``None`` when the range is incomplete (missing bounds or choices)
    so the caller can skip updating that parameter.

    Returns:
        The suggested value, or ``None`` if the range has no usable bounds.
    """
    if isinstance(tuning_range, FloatRange) and tuning_range.low is not None and tuning_range.high is not None:
        return trial.suggest_float(trial_key, tuning_range.low, tuning_range.high, log=tuning_range.log)
    if isinstance(tuning_range, IntRange) and tuning_range.low is not None and tuning_range.high is not None:
        return trial.suggest_int(trial_key, tuning_range.low, tuning_range.high, log=tuning_range.log)
    if isinstance(tuning_range, CategoricalRange) and tuning_range.choices is not None:
        return trial.suggest_categorical(trial_key, list(tuning_range.choices))
    return None


def _build_hp_updates(
    model_tuning_info: list[ModelTuningInfo],
    per_field: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build a config-level update dict by applying *per_field* values to each HP group.

    Returns:
        Mapping of config field name → updated :class:`TunableHyperParams` instance.
    """
    return {
        tf.model_hyperparams_field_name: tf.tunable_hyperparams.model_copy(
            update=per_field[tf.model_hyperparams_field_name]
        )
        for tf in model_tuning_info
        if tf.model_hyperparams_field_name in per_field
    }


class _TuningObjective:
    """Callable Optuna objective that encapsulates the context for a tuning run."""

    def __init__(
        self,
        combined_space: dict[str, _TrialEntry],
        model_tuning_info: list[ModelTuningInfo],
        config: TunableWorkflowConfig,
        train_dataset: TimeSeriesDataset,
        create_workflow: Callable[..., CustomForecastingWorkflow],
        target_quantile: QuantileOrGlobal,
        metric_name: str,
    ) -> None:
        """Store the tuning context."""
        self._combined_space = combined_space
        self._model_tuning_info = model_tuning_info
        self._config = config
        self._train_dataset = train_dataset
        self._create_workflow = create_workflow
        self._target_quantile: QuantileOrGlobal = target_quantile
        self._metric_name = metric_name

    def __call__(self, trial: optuna.Trial) -> float:
        """Evaluate a single Optuna trial.

        Returns:
            Score to maximise, or ``-inf`` on failure.
        """
        per_field: dict[str, dict[str, Any]] = {}
        for trial_key, trial_entry in self._combined_space.items():
            value = _suggest_hyperparam_value(trial, trial_key, trial_entry.tuning_range)
            if value is not None:
                per_field.setdefault(trial_entry.model_hyperparams_field_name, {})[trial_entry.hyperparam_name] = value

        tuned_config = self._config.model_copy(update=_build_hp_updates(self._model_tuning_info, per_field))

        trial_workflow = self._create_workflow(tuned_config)
        trial_result = trial_workflow.fit(self._train_dataset)
        if trial_result is None:
            return float("-inf")
        metrics = trial_result.metrics_val if trial_result.metrics_val is not None else trial_result.metrics_train
        score = metrics.get_metric(quantile=self._target_quantile, metric_name=self._metric_name)
        return float(score) if score is not None else float("-inf")


def tune[ConfigT: TunableWorkflowConfig](
    config: ConfigT,
    train_dataset: TimeSeriesDataset,
    create_workflow: Callable[[ConfigT], CustomForecastingWorkflow],
) -> tuple[ConfigT, optuna.Study, dict[str, Any]]:
    """Generic hyperparameter tuning for any TunableWorkflowConfig.

    Args:
        config: Any config implementing TunableWorkflowConfig.
        train_dataset: Dataset used for all trial fit calls.
        create_workflow: Factory that builds a CustomForecastingWorkflow from config.

    Returns:
        (best_config, study, best_params)

    Raises:
        ValueError: If no hyperparameter field has tune=True ranges.
    """
    model_tuning_info = config.get_model_tuning_info()
    if not model_tuning_info:
        msg = (
            f"No tunable hyperparameters found on config '{config.model_id}'. "
            "Pass TuningRange(tune=True) objects as field values in the hyperparams constructor."
        )
        raise ValueError(msg)

    target_quantile, metric_name, _ = config.model_selection_metric

    # Aggregate search spaces across tunable hyperparam fields.
    # Use prefixes to avoid collisions.
    multi = len(model_tuning_info) > 1
    combined_space: dict[
        str, _TrialEntry
    ] = {}  # trial_key -> (model_hyperparams_field_name, hyperparam_name, tuning_range)
    for tf in model_tuning_info:
        for hyperparam_name, tuning_range in tf.search_space.items():
            trial_key = f"{tf.model_hyperparams_field_name}.{hyperparam_name}" if multi else hyperparam_name
            combined_space[trial_key] = _TrialEntry(tf.model_hyperparams_field_name, hyperparam_name, tuning_range)

    # Build and run the Optuna study
    objective = _TuningObjective(
        combined_space=combined_space,
        model_tuning_info=model_tuning_info,
        config=config,
        train_dataset=train_dataset,
        create_workflow=create_workflow,
        target_quantile=target_quantile,
        metric_name=metric_name,
    )
    study = run_optuna_study(
        objective=objective,
        n_trials=config.optuna_n_trials,
        seed=config.optuna_seed,
        study_name=f"tuning_{config.model_id}",
    )

    # Reconstruct the best config by applying the best parameters per field
    best_config = _reconstruct_best_config(config, model_tuning_info, study)
    return best_config, study, study.best_params


def _reconstruct_best_config[ConfigT: TunableWorkflowConfig](
    config: ConfigT,
    model_tuning_info_list: list[ModelTuningInfo],
    study: optuna.Study,
) -> ConfigT:
    """Returns the best config using the optuna study results for all tunable fields.

    Args:
        config: Any config implementing TunableWorkflowConfig.
        model_tuning_info_list: list of :class: ModelTuningInfo per model,
        study: :class:`optuna.Study` completed with trial results

    Returns:
        :class:`TunableWorkflowConfig` with the best best hyperparameter values.
    """
    multi = len(model_tuning_info_list) > 1
    per_field_best: dict[str, dict[str, Any]] = {}
    for trial_key, value in study.best_params.items():
        if multi and "." in trial_key:
            model_hyperparams_field_name, hyperparam_name = trial_key.split(".", 1)
        else:
            model_hyperparams_field_name = model_tuning_info_list[0].model_hyperparams_field_name
            hyperparam_name = trial_key
        per_field_best.setdefault(model_hyperparams_field_name, {})[hyperparam_name] = value

    return config.model_copy(update=_build_hp_updates(model_tuning_info_list, per_field_best))


def fit_with_tuning[ConfigT: TunableWorkflowConfig](
    config: ConfigT, train_dataset: TimeSeriesDataset, create_workflow: Callable[[ConfigT], CustomForecastingWorkflow]
) -> TuningResult:
    """Create, tune and fit.

    Args:
        config: Any config implementing TunableWorkflowConfig.
        train_dataset: Dataset used for fit.
        create_workflow: Factory that builds a CustomForecastingWorkflow from config.

    Returns:
        :class:`TuningResult` with the fitted workflow, completed study, and best config.
    """
    best_config, study, _ = tune(config=config, train_dataset=train_dataset, create_workflow=create_workflow)
    workflow = create_workflow(best_config)
    result = workflow.fit(train_dataset)
    return TuningResult(workflow=workflow, fit_result=result, study=study, best_config=best_config)


__all__ = [
    "CategoricalRange",
    "FloatRange",
    "IntRange",
    "ModelTuningInfo",
    "TunableHyperParams",
    "TunableWorkflowConfig",
    "TuningConfigMixin",
    "TuningRange",
    "TuningResult",
    "apply_trial_suggestions",
    "fit_with_tuning",
    "get_search_space",
    "run_optuna_study",
    "tune",
]
