# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from typing import Annotated

import pytest

from openstef_core.param_ranges import CategoricalRange, FloatRange, IntRange


# ---------------------------------------------------------------------------
# FloatRange
# ---------------------------------------------------------------------------

def test_float_range_construction():
    # Arrange / Act
    fr = FloatRange(low=0.01, high=1.0, log=True, tune=True)

    # Assert
    assert fr.low == 0.01
    assert fr.high == 1.0
    assert fr.log is True
    assert fr.tune is True


def test_float_range_defaults():
    # Arrange / Act
    fr = FloatRange()

    # Assert
    assert fr.low is None
    assert fr.high is None
    assert fr.log is False
    assert fr.tune is False


def test_float_range_frozen():
    # Arrange
    fr = FloatRange(low=0.01, high=1.0)

    # Act / Assert
    with pytest.raises(Exception):  # noqa: B017
        fr.low = 5.0  # type: ignore[misc]


def test_float_range_validates_low_gt_high():
    # Act / Assert
    with pytest.raises(ValueError, match="low.*must be <= high"):
        FloatRange(low=10.0, high=1.0)


def test_float_range_resolve_fills_none_bounds():
    # Arrange
    override = FloatRange(low=None, high=None, log=True, tune=True)
    class_default = FloatRange(low=0.01, high=1.0)

    # Act
    resolved = override.resolve(class_default)

    # Assert
    assert resolved.low == 0.01
    assert resolved.high == 1.0
    assert resolved.log is True  # kept from override
    assert resolved.tune is True


def test_float_range_resolve_keeps_explicit_bounds():
    # Arrange
    override = FloatRange(low=0.001, high=0.5, tune=True)
    class_default = FloatRange(low=0.01, high=1.0)

    # Act
    resolved = override.resolve(class_default)

    # Assert — explicit bounds not overwritten
    assert resolved.low == 0.001
    assert resolved.high == 0.5


def test_float_range_resolve_with_none_class_default():
    # Arrange
    override = FloatRange(low=0.01, high=1.0, tune=True)

    # Act
    resolved = override.resolve(None)

    # Assert — unchanged
    assert resolved.low == 0.01
    assert resolved.high == 1.0


# ---------------------------------------------------------------------------
# IntRange
# ---------------------------------------------------------------------------

def test_int_range_construction():
    # Arrange / Act
    ir = IntRange(low=1, high=15, log=False, tune=True)

    # Assert
    assert ir.low == 1
    assert ir.high == 15


def test_int_range_validates_low_gt_high():
    # Act / Assert
    with pytest.raises(ValueError, match="low.*must be <= high"):
        IntRange(low=20, high=5)


def test_int_range_resolve_fills_none_bounds():
    # Arrange
    override = IntRange(low=None, high=None, tune=True)
    class_default = IntRange(low=1, high=15)

    # Act
    resolved = override.resolve(class_default)

    # Assert
    assert resolved.low == 1
    assert resolved.high == 15


# ---------------------------------------------------------------------------
# CategoricalRange
# ---------------------------------------------------------------------------

def test_categorical_range_construction():
    # Arrange / Act
    cr = CategoricalRange(choices=("a", "b", "c"), tune=True)

    # Assert
    assert cr.choices == ("a", "b", "c")
    assert cr.tune is True


def test_categorical_range_validates_empty_choices():
    # Act / Assert
    with pytest.raises(ValueError, match="choices must not be empty"):
        CategoricalRange(choices=())


def test_categorical_range_resolve_fills_none_choices():
    # Arrange
    override = CategoricalRange(choices=None, tune=True)
    class_default = CategoricalRange(choices=("hist", "approx"))

    # Act
    resolved = override.resolve(class_default)

    # Assert
    assert resolved.choices == ("hist", "approx")
    assert resolved.tune is True


def test_categorical_range_resolve_keeps_explicit_choices():
    # Arrange
    override = CategoricalRange(choices=("x", "y"), tune=True)
    class_default = CategoricalRange(choices=("a", "b", "c"))

    # Act
    resolved = override.resolve(class_default)

    # Assert
    assert resolved.choices == ("x", "y")
