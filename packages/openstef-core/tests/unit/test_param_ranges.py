# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import pytest

from openstef_core.mixins.param_ranges import CategoricalRange, FloatRange, IntRange


def test_float_range_frozen():
    # Arrange
    fr = FloatRange(low=0.01, high=1.0)

    # Act / Assert
    with pytest.raises(AttributeError):
        fr.low = 5.0  # ty: ignore[invalid-assignment]


def test_float_range_validates_low_gt_high():
    # Act / Assert
    with pytest.raises(ValueError, match=r"low.*must be <= high"):
        FloatRange(low=10.0, high=1.0)


def test_float_range_resolve_fills_none_bounds():
    # Arrange
    override = FloatRange(low=None, high=None, log=True, tune=True)
    class_default = FloatRange(low=0.01, high=1.0)

    # Act
    resolved = override.resolve(class_default)

    # Assert
    assert resolved.low == pytest.approx(0.01)
    assert resolved.high == pytest.approx(1.0)
    assert resolved.log is True  # kept from override
    assert resolved.tune is True


def test_float_range_resolve_keeps_explicit_bounds():
    # Arrange
    override = FloatRange(low=0.001, high=0.5, tune=True)
    class_default = FloatRange(low=0.01, high=1.0)

    # Act
    resolved = override.resolve(class_default)

    # Assert — explicit bounds not overwritten
    assert resolved.low == pytest.approx(0.001)
    assert resolved.high == pytest.approx(0.5)


def test_float_range_resolve_with_none_class_default():
    # Arrange
    override = FloatRange(low=0.01, high=1.0, tune=True)

    # Act
    resolved = override.resolve(None)

    # Assert — unchanged
    assert resolved.low == pytest.approx(0.01)
    assert resolved.high == pytest.approx(1.0)


def test_int_range_validates_low_gt_high():
    # Act / Assert
    with pytest.raises(ValueError, match=r"low.*must be <= high"):
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
