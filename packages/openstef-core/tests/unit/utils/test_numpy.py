# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the pure NumPy forecasting helpers."""

import numpy as np
import pytest

from openstef_core.utils.numpy import interpolate_quantiles, zero_fill_with_mask


def test_interpolate_quantiles_returns_exact_values_on_matching_levels() -> None:
    """Target levels that coincide with source levels are copied unchanged."""
    # Arrange
    source = [0.1, 0.5, 0.9]
    predictions = np.array([[10.0, 50.0, 90.0]])

    # Act
    result = interpolate_quantiles(predictions, source, target_quantiles=[0.1, 0.5, 0.9])

    # Assert
    np.testing.assert_array_almost_equal(result, predictions)


def test_interpolate_quantiles_linearly_interpolates_between_levels() -> None:
    """A target halfway between two source levels is their midpoint."""
    # Arrange
    source = [0.2, 0.8]
    predictions = np.array([[20.0, 80.0]])

    # Act: 0.5 is exactly halfway between 0.2 and 0.8
    result = interpolate_quantiles(predictions, source, target_quantiles=[0.5])

    # Assert
    np.testing.assert_array_almost_equal(result, [[50.0]])


def test_interpolate_quantiles_clamps_targets_below_source_range() -> None:
    """Targets below the lowest source level extrapolate as a constant."""
    # Arrange
    source = [0.2, 0.5, 0.8]
    predictions = np.array([[20.0, 50.0, 80.0]])

    # Act
    result = interpolate_quantiles(predictions, source, target_quantiles=[0.01])

    # Assert: clamped to the lowest source prediction, not extrapolated downward
    np.testing.assert_array_almost_equal(result, [[20.0]])


def test_interpolate_quantiles_clamps_targets_above_source_range() -> None:
    """Targets above the highest source level extrapolate as a constant."""
    # Arrange
    source = [0.2, 0.5, 0.8]
    predictions = np.array([[20.0, 50.0, 80.0]])

    # Act
    result = interpolate_quantiles(predictions, source, target_quantiles=[0.99])

    # Assert: clamped to the highest source prediction
    np.testing.assert_array_almost_equal(result, [[80.0]])


def test_interpolate_quantiles_warns_when_clamping_out_of_range(caplog: pytest.LogCaptureFixture) -> None:
    """A requested level beyond the source range logs a warning before clamping."""
    # Arrange
    source = [0.1, 0.5, 0.9]
    predictions = np.array([[10.0, 50.0, 90.0]])

    # Act
    with caplog.at_level("WARNING"):
        result = interpolate_quantiles(predictions, source, target_quantiles=[0.999])

    # Assert: clamped to the highest source prediction, and the user was warned
    np.testing.assert_array_almost_equal(result, [[90.0]])
    assert "outside the source range" in caplog.text


def test_interpolate_quantiles_does_not_warn_within_range(caplog: pytest.LogCaptureFixture) -> None:
    """Targets inside the source range interpolate silently."""
    # Arrange
    source = [0.1, 0.5, 0.9]
    predictions = np.array([[10.0, 50.0, 90.0]])

    # Act
    with caplog.at_level("WARNING"):
        interpolate_quantiles(predictions, source, target_quantiles=[0.3, 0.7])

    # Assert
    assert not caplog.text


def test_interpolate_quantiles_preserves_leading_dimensions() -> None:
    """Interpolation only touches the last axis; leading shape is preserved."""
    # Arrange: a horizon of four rows, each with three source levels
    source = [0.1, 0.5, 0.9]
    predictions = np.tile(np.array([10.0, 50.0, 90.0]), (4, 1))

    # Act
    result = interpolate_quantiles(predictions, source, target_quantiles=[0.1, 0.3, 0.5])

    # Assert
    assert result.shape == (4, 3)
    np.testing.assert_array_almost_equal(result[:, 0], np.full(4, 10.0))
    np.testing.assert_array_almost_equal(result[:, 2], np.full(4, 50.0))


def test_interpolate_quantiles_keeps_monotonic_output() -> None:
    """Resampling a monotone source grid yields a monotone target grid."""
    # Arrange
    source = [0.1, 0.3, 0.5, 0.7, 0.9]
    predictions = np.array([[1.0, 3.0, 5.0, 7.0, 9.0]])

    # Act
    result = interpolate_quantiles(predictions, source, target_quantiles=[0.2, 0.4, 0.6, 0.8])

    # Assert
    assert np.all(np.diff(result[0]) > 0)


def test_interpolate_quantiles_raises_when_source_not_ascending() -> None:
    """A non-ascending source grid is rejected."""
    # Arrange
    source = [0.5, 0.2, 0.8]
    predictions = np.array([[1.0, 2.0, 3.0]])

    # Act / Assert
    with pytest.raises(ValueError, match="strictly ascending"):
        interpolate_quantiles(predictions, source, target_quantiles=[0.5])


def test_interpolate_quantiles_raises_on_shape_mismatch() -> None:
    """The predictions' last axis must match the source-grid length."""
    # Arrange
    source = [0.1, 0.5, 0.9]
    predictions = np.array([[1.0, 2.0]])

    # Act / Assert
    with pytest.raises(ValueError, match="must match the number"):
        interpolate_quantiles(predictions, source, target_quantiles=[0.5])


def test_interpolate_quantiles_raises_when_too_few_source_levels() -> None:
    """At least two source levels are required to interpolate."""
    # Arrange
    source = [0.5]
    predictions = np.array([[1.0]])

    # Act / Assert
    with pytest.raises(ValueError, match="at least two levels"):
        interpolate_quantiles(predictions, source, target_quantiles=[0.5])


def test_zero_fill_with_mask_zeros_non_finite_and_flags_finite() -> None:
    """Non-finite entries become 0 in values and 0 in the mask; finite stay 1."""
    # Arrange
    values = np.array([[1.0, np.nan, 3.0], [np.inf, 5.0, -np.inf]])

    # Act
    filled, mask = zero_fill_with_mask(values)

    # Assert
    np.testing.assert_array_equal(filled, np.array([[1.0, 0.0, 3.0], [0.0, 5.0, 0.0]], dtype=np.float32))
    np.testing.assert_array_equal(mask, np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32))


def test_zero_fill_with_mask_returns_float32() -> None:
    """Both outputs are float32 regardless of input dtype."""
    # Arrange
    values = np.array([1, 2, 3], dtype=np.int64)

    # Act
    filled, mask = zero_fill_with_mask(values)

    # Assert
    assert filled.dtype == np.float32
    assert mask.dtype == np.float32
    np.testing.assert_array_equal(mask, np.ones(3, dtype=np.float32))
