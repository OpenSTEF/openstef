# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""Feature selection utilities for transforms.

Provides standardized feature selection with include/exclude patterns.
Transforms use this to consistently specify which features to operate on.
"""

import re
from typing import Any, ClassVar, Self, cast, override

from pydantic import Field

from openstef_core.base_model import BaseConfig


class FeatureSelection(BaseConfig):
    """Standardized feature selection with include/exclude patterns.

    Supports both exact matching and regex pattern matching for feature selection.
    Features can be specified by inclusion (whitelist) or exclusion (blacklist), or both.
    When both are specified, inclusion is applied first, then exclusion.

    Use `FeatureSelection.ALL` to select all available features.

    Example:
        >>> from openstef_models.utils.feature_selection import (
        ...     FeatureSelection,
        ...     Include,
        ...     Exclude,
        ... )
        >>>
        >>> # Select all features
        >>> all_features = FeatureSelection.ALL
        >>> all_features.resolve(['a', 'b', 'c'])
        ['a', 'b', 'c']
        >>>
        >>> # Include only specific features (exact match)
        >>> include_only = Include('a', 'b')
        >>> include_only.resolve(['a', 'b', 'c', 'd'])
        ['a', 'b']
        >>>
        >>> # Exclude specific features (exact match)
        >>> exclude_some = Exclude('b', 'd')
        >>> exclude_some.resolve(['a', 'b', 'c', 'd'])
        ['a', 'c']
        >>>
        >>> # Regex matching
        >>> regex_sel = FeatureSelection(include_regex={r'^b_.*'})
        >>> regex_sel.resolve(['b_1', 'b_2', 'c_1'])
        ['b_1', 'b_2']
        >>>
        >>> # Combine exact and regex
        >>> combined = FeatureSelection(include={'a'}, include_regex={r'^b.*'})
        >>> combined.resolve(['a', 'b1', 'b2', 'c'])
        ['a', 'b1', 'b2']
    """

    include: set[str] | None = Field(
        default=None,
        description="Set of exact feature names to include. Use None to include all features.",
        frozen=True,
    )
    include_regex: set[str] | None = Field(
        default=None,
        description="Set of regex patterns to include features. Use None to include all features.",
        frozen=True,
    )
    exclude: set[str] | None = Field(
        default=None,
        description="Set of exact feature names to exclude. Use None to exclude no features.",
        frozen=True,
    )
    exclude_regex: set[str] | None = Field(
        default=None,
        description="Set of regex patterns to exclude features. Use None to exclude no features.",
        frozen=True,
    )

    ALL: ClassVar[Self]
    NONE: ClassVar[Self]

    @staticmethod
    def _matches_regex(feature: str, patterns: set[str]) -> bool:
        """Check if a feature matches any regex pattern in the set.

        Args:
            feature: Feature name to check.
            patterns: Set of regex patterns to match against.

        Returns:
            True if feature matches any regex pattern.
        """
        return any(re.match(pattern, feature) for pattern in patterns)

    def _should_include_feature(self, feature: str) -> bool:
        """Check if a feature should be included based on include filters.

        Args:
            feature: Feature name to check.

        Returns:
            True if feature should be included.
        """
        if self.include is None and self.include_regex is None:
            return True
        exact_match = self.include is not None and feature in self.include
        regex_match = self.include_regex is not None and self._matches_regex(feature, self.include_regex)
        return exact_match or regex_match

    def _should_exclude_feature(self, feature: str) -> bool:
        """Check if a feature should be excluded based on exclude filters.

        Args:
            feature: Feature name to check.

        Returns:
            True if feature should be excluded.
        """
        if self.exclude is None and self.exclude_regex is None:
            return False
        exact_match = self.exclude is not None and feature in self.exclude
        regex_match = self.exclude_regex is not None and self._matches_regex(feature, self.exclude_regex)
        return exact_match or regex_match

    def resolve(self, features: list[str]) -> list[str]:
        """Resolve the final list of features based on include and exclude filters.

        Args:
            features: List of all available feature names.

        Returns:
            List of feature names after applying include and exclude filters.
        """
        return [
            feature
            for feature in features
            if self._should_include_feature(feature) and not self._should_exclude_feature(feature)
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

        def _union(a: set[str] | None, b: set[str] | None) -> set[str] | None:
            return None if a is None and b is None else (a or set()) | (b or set())

        return self.__class__(
            include=_union(self.include, other.include),
            include_regex=_union(self.include_regex, other.include_regex),
            exclude=_union(self.exclude, other.exclude),
            exclude_regex=_union(self.exclude_regex, other.exclude_regex),
        )

    @override
    def __setstate__(self, state: Any) -> None:  # TODO(#799): delete after stable release
        if "include_regex" not in state["__dict__"]:
            state["__dict__"]["include_regex"] = None
            cast(set[str], state["__pydantic_fields_set__"]).add("include_regex")
        if "exclude_regex" not in state["__dict__"]:
            state["__dict__"]["exclude_regex"] = None
            cast(set[str], state["__pydantic_fields_set__"]).add("exclude_regex")

        return super().__setstate__(state)


FeatureSelection.ALL = FeatureSelection(include=None, include_regex=None, exclude=None, exclude_regex=None)
FeatureSelection.NONE = FeatureSelection(include=set(), include_regex=set(), exclude=None, exclude_regex=None)


def Include(*features: str) -> FeatureSelection:  # noqa: N802
    """Helper to create a FeatureSelection that includes only specified features.

    Args:
        *features: Feature names to include.

    Returns:
        FeatureSelection instance with specified features included.
    """
    return FeatureSelection(include={*features}, include_regex=None, exclude=None, exclude_regex=None)


def Exclude(*features: str) -> FeatureSelection:  # noqa: N802
    """Helper to create a FeatureSelection that excludes specified features.

    Args:
        *features: Feature names to exclude.

    Returns:
        FeatureSelection instance with specified features excluded.
    """
    return FeatureSelection(include=None, include_regex=None, exclude={*features}, exclude_regex=None)


def IncludeRegex(*patterns: str) -> FeatureSelection:  # noqa: N802
    """Helper to create a FeatureSelection that includes features matching regex patterns.

    Args:
        *patterns: Regex patterns to include.

    Returns:
        FeatureSelection instance with specified patterns included.
    """
    return FeatureSelection(include=None, include_regex={*patterns}, exclude=None, exclude_regex=None)


def ExcludeRegex(*patterns: str) -> FeatureSelection:  # noqa: N802
    """Helper to create a FeatureSelection that excludes features matching regex patterns.

    Args:
        *patterns: Regex patterns to exclude.

    Returns:
        FeatureSelection instance with specified patterns excluded.
    """
    return FeatureSelection(include=None, include_regex=None, exclude=None, exclude_regex={*patterns})
