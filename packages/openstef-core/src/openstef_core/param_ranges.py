# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""Tuning range types and metadata for hyperparameter search spaces.

Range types (``FloatRange``, ``IntRange``, ``CategoricalRange``) are frozen
dataclasses — NOT Pydantic models — because they're used as ``Annotated``
metadata on ``HyperParams`` fields.  Pydantic ``BaseModel`` instances in
``Annotated`` are interpreted as type annotations, breaking the field's
actual type.  Plain dataclasses are treated as opaque metadata.

``ModelTuningInfo`` is a frozen Pydantic model since it's not used in
``Annotated`` context.
"""

from dataclasses import dataclass, replace
from typing import Any, Self

from pydantic import ConfigDict, Field, model_validator

from openstef_core.base_model import BaseConfig, BaseModel


@dataclass(frozen=True)
class FloatRange:
    """Annotate a ``HyperParams`` float field as tunable within ``[low, high]``."""

    low: float | None = None
    high: float | None = None
    log: bool = False
    tune: bool = False

    def __post_init__(self) -> None:  # noqa: D105  # low must be <= high
        if self.low is not None and self.high is not None and self.low > self.high:
            msg = f"low ({self.low}) must be <= high ({self.high})"
            raise ValueError(msg)

    def resolve(self, class_default: Self | None) -> Self:
        """Fill ``None`` bounds from *class_default*.

        Returns:
            Resolved range.
        """
        if class_default is None:
            return self
        return replace(
            self,
            low=self.low if self.low is not None else class_default.low,
            high=self.high if self.high is not None else class_default.high,
        )


@dataclass(frozen=True)
class IntRange:
    """Annotate a ``HyperParams`` int field as tunable within ``[low, high]``."""

    low: int | None = None
    high: int | None = None
    log: bool = False
    tune: bool = False

    def __post_init__(self) -> None:  # noqa: D105  # low must be <= high
        if self.low is not None and self.high is not None and self.low > self.high:
            msg = f"low ({self.low}) must be <= high ({self.high})"
            raise ValueError(msg)

    def resolve(self, class_default: Self | None) -> Self:
        """Fill ``None`` bounds from *class_default*.

        Returns:
            Resolved range.
        """
        if class_default is None:
            return self
        return replace(
            self,
            low=self.low if self.low is not None else class_default.low,
            high=self.high if self.high is not None else class_default.high,
        )


@dataclass(frozen=True)
class CategoricalRange:
    """Annotate a ``HyperParams`` field as tunable over discrete ``choices``."""

    choices: tuple[Any, ...] | None = None
    tune: bool = False

    def __post_init__(self) -> None:  # noqa: D105  # choices must be non-empty when provided
        if self.choices is not None and len(self.choices) == 0:
            msg = "choices must not be empty"
            raise ValueError(msg)

    def resolve(self, class_default: Self | None) -> Self:
        """Fill ``None`` choices from *class_default*.

        Returns:
            Resolved range.
        """
        if class_default is None:
            return self
        return replace(
            self,
            choices=self.choices if self.choices is not None else class_default.choices,
        )


type TuningRange = FloatRange | IntRange | CategoricalRange


class ModelTuningInfo(BaseModel):
    """Groups a hyperparameter config with its resolved search space."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    field_name: str = Field(description="Name of the HyperParams field on the parent config.")
    hyperparams: BaseConfig = Field(description="The HyperParams instance that owns the search space.")
    search_space: dict[str, TuningRange] = Field(description="Resolved tuning ranges keyed by parameter name.")

    @model_validator(mode="after")
    def _validate_search_space(self) -> Self:
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
