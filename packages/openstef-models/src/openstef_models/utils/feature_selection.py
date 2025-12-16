# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""Feature selection utilities for transforms.

Provides standardized feature selection with include/exclude patterns.
Transforms use this to consistently specify which features to operate on.
"""

from typing import ClassVar, Self

from pydantic import Field

from openstef_core.base_model import BaseConfig


class FeatureSelection(BaseConfig):
    """Standardized feature selection with include/exclude patterns.

    Defines which features a transform should operate on. Features can be
    specified by inclusion (whitelist) or exclusion (blacklist), or both.
    When both are specified, inclusion is applied first, then exclusion.

    Use `FeatureSelection.ALL` to select all available features.
    """

    include: set[str] | None = Field(
        default=None,
        description=("List of feature names to include. Use None to include all features from the input dataset."),
        frozen=True,
    )
    exclude: set[str] | None = Field(
        default=None,
        description="List of feature names to exclude. Use None to exclude no features.",
        frozen=True,
    )

    ALL: ClassVar[Self]
    NONE: ClassVar[Self]

    def resolve(self, features: list[str]) -> list[str]:
        """Resolve the final list of features based on include and exclude lists.

        Args:
            features: List of all available feature names.

        Returns:
            List of feature names after applying include and exclude filters.
        """
        return [
            feature
            for feature in features
            if (self.include is None or feature in self.include)
            and (self.exclude is None or feature not in self.exclude)
        ]

    def combine(self, other: Self | None) -> Self:
        """Create a new FeatureSelection that is the union of this and another.

        Args:
            other: Another FeatureSelection instance.

        Returns:
            A new FeatureSelection instance that is the union of this and the other instance.
        """
        if other is None:
            return self

        return self.__class__(
            include=((self.include or set()) | (other.include or set())),
            exclude=((self.exclude or set()) | (other.exclude or set())),
        )


FeatureSelection.ALL = FeatureSelection(include=None, exclude=None)
FeatureSelection.NONE = FeatureSelection(include=set(), exclude=None)


def Include(*features: str) -> FeatureSelection:  # noqa: N802
    """Helper to create a FeatureSelection that includes only specified features.

    Args:
        *features: Feature names to include.

    Returns:
        FeatureSelection instance with specified features included.
    """
    return FeatureSelection(include={*features}, exclude=None)


def Exclude(*features: str) -> FeatureSelection:  # noqa: N802
    """Helper to create a FeatureSelection that excludes specified features.

    Args:
        *features: Feature names to exclude.

    Returns:
        FeatureSelection instance with specified features excluded.
    """
    return FeatureSelection(include=None, exclude={*features})
