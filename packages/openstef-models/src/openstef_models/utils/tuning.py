# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""Hyperparameter tuning utilities for OpenSTEF models.

Provides dataclasses for describing hyperparameter search spaces, helper functions to
extract and merge search spaces from annotated HyperParams classes, and a thin wrapper
around Optuna for running Bayesian hyperparameter optimisation studies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import optuna
from pydantic import BaseModel, PrivateAttr, model_validator

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic.fields import FieldInfo

from openstef_core.mixins import HyperParams


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

    _instance_ranges: dict[str, TuningRange] = PrivateAttr(  # pyright: ignore[reportUnknownVariableType]
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
            result.__pydantic_private__["_instance_ranges"] = instance_ranges
        return result


def _get_class_range(field_info: FieldInfo) -> TuningRange | None:
    """Return the first TuningRange found in a Pydantic FieldInfo's metadata."""
    for meta in field_info.metadata:
        if isinstance(meta, (FloatRange, IntRange, CategoricalRange)):
            return meta
    return None


def _merge_range(override: TuningRange, class_range: TuningRange | None) -> TuningRange:
    """Merge *override* with *class_range*, filling ``None`` from the class defaults.

    For ``FloatRange`` / ``IntRange``, ``None`` values for ``low`` or ``high`` are filled
    in from *class_range*.  For ``CategoricalRange``, ``None`` for ``choices`` falls back
    to *class_range*.  The ``tune`` flag always comes from *override*.

    Returns:
        A new :class:`TuningRange` with ``None`` bounds merged from *class_range*.
    """
    if isinstance(override, FloatRange):
        cr = class_range if isinstance(class_range, FloatRange) else None
        return FloatRange(
            low=override.low if override.low is not None else (cr.low if cr else None),
            high=override.high if override.high is not None else (cr.high if cr else None),
            log=override.log,
            tune=override.tune,
        )
    if isinstance(override, IntRange):
        cr = class_range if isinstance(class_range, IntRange) else None
        return IntRange(
            low=override.low if override.low is not None else (cr.low if cr else None),
            high=override.high if override.high is not None else (cr.high if cr else None),
            log=override.log,
            tune=override.tune,
        )
    cr = class_range if isinstance(class_range, CategoricalRange) else None
    return CategoricalRange(
        choices=override.choices if override.choices is not None else (cr.choices if cr else None),
        tune=override.tune,
    )


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
        Mapping of field-name → effective :class:`TuningRange` for all tunable fields.

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
    for field_name, field_info in type(hyperparams).model_fields.items():
        class_range = _get_class_range(field_info)
        override = instance_ranges.get(field_name)

        if override is not None:
            if not override.tune:
                continue
            result[field_name] = _merge_range(override, class_range)
        elif class_range is not None and class_range.tune:
            result[field_name] = class_range

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


def suggest_hyperparams[HP: BaseModel](
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
    for field_name, param in space.items():
        if isinstance(param, FloatRange):
            if param.low is not None and param.high is not None:
                updates[field_name] = trial.suggest_float(field_name, param.low, param.high, log=param.log)
        elif isinstance(param, IntRange):
            if param.low is not None and param.high is not None:
                updates[field_name] = trial.suggest_int(field_name, param.low, param.high, log=param.log)
        elif param.choices is not None:
            updates[field_name] = trial.suggest_categorical(field_name, list(param.choices))
    return current.model_copy(update=updates)


def run_optuna_study(
    objective: Callable[[optuna.Trial], float],
    n_trials: int,
    seed: int | None = 42,
    direction: str = "maximize",
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


__all__ = [
    "CategoricalRange",
    "FloatRange",
    "IntRange",
    "TunableHyperParams",
    "TuningRange",
    "get_search_space",
    "run_optuna_study",
    "suggest_hyperparams",
]
