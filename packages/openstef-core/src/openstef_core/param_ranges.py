# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""Tuning range types and metadata for hyperparameter search spaces.

Provides frozen Pydantic models for annotating ``HyperParams`` fields as tunable,
plus ``ModelTuningInfo`` for grouping a hyperparameter set with its search space.
None of these types depend on Optuna — they live in ``openstef-core`` so that
anything can import them without pulling in optional dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict, model_validator

from openstef_core.base_model import BaseModel

if TYPE_CHECKING:
    from openstef_core.mixins.predictor import HyperParams


class FloatRange(BaseModel):
    """Annotate a ``HyperParams`` float field as tunable within ``[low, high]``.

    Pass ``None`` for ``low`` / ``high`` to inherit bounds from the class-level
    ``Annotated`` metadata when the search space is resolved.
    """

    model_config = ConfigDict(frozen=True)

    low: float | None = None
    high: float | None = None
    log: bool = False
    tune: bool = False

    @model_validator(mode="after")
    def _validate_bounds(self) -> FloatRange:
        if self.low is not None and self.high is not None and self.low > self.high:
            msg = f"low ({self.low}) must be <= high ({self.high})"
            raise ValueError(msg)
        return self

    def resolve(self, class_default: FloatRange | None) -> FloatRange:
        """Fill ``None`` bounds from a class-level default range."""
        if class_default is None:
            return self
        return self.model_copy(
            update={
                "low": self.low if self.low is not None else class_default.low,
                "high": self.high if self.high is not None else class_default.high,
            }
        )


class IntRange(BaseModel):
    """Annotate a ``HyperParams`` int field as tunable within ``[low, high]``."""

    model_config = ConfigDict(frozen=True)

    low: int | None = None
    high: int | None = None
    log: bool = False
    tune: bool = False

    @model_validator(mode="after")
    def _validate_bounds(self) -> IntRange:
        if self.low is not None and self.high is not None and self.low > self.high:
            msg = f"low ({self.low}) must be <= high ({self.high})"
            raise ValueError(msg)
        return self

    def resolve(self, class_default: IntRange | None) -> IntRange:
        """Fill ``None`` bounds from a class-level default range."""
        if class_default is None:
            return self
        return self.model_copy(
            update={
                "low": self.low if self.low is not None else class_default.low,
                "high": self.high if self.high is not None else class_default.high,
            }
        )


class CategoricalRange(BaseModel):
    """Annotate a ``HyperParams`` field as tunable over discrete ``choices``."""

    model_config = ConfigDict(frozen=True)

    choices: tuple[Any, ...] | None = None
    tune: bool = False

    @model_validator(mode="after")
    def _validate_choices(self) -> CategoricalRange:
        if self.choices is not None and len(self.choices) == 0:
            msg = "choices must not be empty"
            raise ValueError(msg)
        return self

    def resolve(self, class_default: CategoricalRange | None) -> CategoricalRange:
        """Fill ``None`` choices from a class-level default range."""
        if class_default is None:
            return self
        return self.model_copy(
            update={
                "choices": self.choices if self.choices is not None else class_default.choices,
            }
        )


type TuningRange = FloatRange | IntRange | CategoricalRange


class ModelTuningInfo(BaseModel):
    """Groups a hyperparameter config with its resolved search space.

    Ensures ``search_space`` is non-empty at construction time.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    field_name: str
    hyperparams: HyperParams
    search_space: dict[str, TuningRange]

    @model_validator(mode="after")
    def _validate_search_space(self) -> ModelTuningInfo:
        if not self.search_space:
            msg = f"search_space for '{self.field_name}' must not be empty"
            raise ValueError(msg)
        return self


__all__ = [
    "CategoricalRange",
    "FloatRange",
    "IntRange",
    "ModelTuningInfo",
    "TuningRange",
]
