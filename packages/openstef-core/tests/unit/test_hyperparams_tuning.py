# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from typing import Annotated

import pytest

from openstef_core.mixins.param_ranges import CategoricalRange, FloatRange, IntRange
from openstef_core.mixins.predictor import HyperParams


class SampleHP(HyperParams):
    lr: Annotated[float, FloatRange(low=0.01, high=1.0)] = 0.3
    depth: Annotated[int, IntRange(low=1, high=15)] = 6
    method: Annotated[str, CategoricalRange(choices=("hist", "approx"))] = "hist"
    plain: float = 1.0  # not annotated with a range


def test_hyperparams_range_extracted_from_kwargs():
    """TuningRange values are extracted; field keeps its class default."""
    # Arrange / Act
    hp = SampleHP(
        lr=FloatRange(low=0.001, high=0.5, tune=True),
        depth=IntRange(tune=True),
    )

    # Assert — field values are the class defaults
    assert hp.lr == pytest.approx(0.3)
    assert hp.depth == 6


def test_hyperparams_mixed_range_and_plain_value():
    """Can pass a range for one field and a plain value for another."""
    # Arrange / Act
    hp = SampleHP(
        lr=FloatRange(low=0.001, high=0.5, tune=True),
        depth=10,  # plain override
    )

    # Assert
    assert hp.lr == pytest.approx(0.3)  # range extracted, field keeps default
    assert hp.depth == 10  # plain value applied


def test_get_search_space_returns_only_tune_true():
    """Only fields with tune=True appear in the search space."""
    # Arrange
    hp = SampleHP(
        lr=FloatRange(low=0.001, high=0.5, tune=True),
        depth=IntRange(tune=False),  # explicit tune=False
    )

    # Act
    space = hp.get_search_space()

    # Assert
    assert "lr" in space
    assert "depth" not in space


def test_get_search_space_resolves_none_bounds():
    """None bounds on instance range are resolved from class-level Annotated metadata."""
    # Arrange
    hp = SampleHP(depth=IntRange(low=None, high=None, tune=True))

    # Act
    space = hp.get_search_space()

    # Assert — resolved from class-level IntRange(1, 15)
    assert space["depth"].low == 1
    assert space["depth"].high == 15


def test_get_search_space_with_class_level_tune():
    """Class-level Annotated range with tune=True is included without instance override."""

    # Arrange
    class AutoTuneHP(HyperParams):
        lr: Annotated[float, FloatRange(low=0.01, high=1.0, tune=True)] = 0.3

    hp = AutoTuneHP()

    # Act
    space = hp.get_search_space()

    # Assert
    assert "lr" in space
    assert space["lr"].low == pytest.approx(0.01)


def test_get_search_space_include_filter():
    """include parameter restricts output to requested fields."""
    # Arrange
    hp = SampleHP(
        lr=FloatRange(tune=True),
        depth=IntRange(tune=True),
        method=CategoricalRange(choices=("hist",), tune=True),
    )

    # Act
    space = hp.get_search_space(include={"lr", "depth"})

    # Assert
    assert set(space.keys()) == {"lr", "depth"}


def test_get_search_space_include_raises_on_missing():
    """include with a non-existent field raises KeyError."""
    # Arrange
    hp = SampleHP(lr=FloatRange(tune=True))

    # Act / Assert
    with pytest.raises(KeyError, match="nonexistent"):
        hp.get_search_space(include={"lr", "nonexistent"})
