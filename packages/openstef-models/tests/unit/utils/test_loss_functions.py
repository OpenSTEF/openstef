# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from openstef_core.types import Quantile
from openstef_models.utils.loss_functions import (
    arctan_loss_multi_objective,
    pinball_loss_magnitude_weighted_multi_objective,
    pinball_loss_multi_objective,
)


@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray, list[Quantile]]:
    """Fixture providing sample data for testing loss functions."""
    # Simple case: 1 sample, 2 quantiles
    y_true = np.array([1.0, 1.0])  # shape (n_samples * n_quantiles,)
    y_pred = np.array([2.0, 2.0])  # overprediction
    quantiles = [Quantile(0.1), Quantile(0.9)]
    return y_true, y_pred, quantiles


@pytest.fixture
def sample_weights() -> np.ndarray:
    """Fixture providing sample weights."""
    return np.array([2.0])


@pytest.mark.parametrize(
    ("y_pred", "sample_weight", "expected_gradient"),
    [
        (np.ones(2) * 1.5, None, np.array([0.45, 0.05])),  # overprediction, no weights
        (np.zeros(2), None, np.array([-0.05, -0.45])),  # underprediction, no weights
        (np.ones(2) * 1.5, np.array([2.0]), np.array([0.9, 0.1])),  # overprediction, with weights
        (np.zeros(2), np.array([2.0]), np.array([-0.1, -0.9])),  # underprediction, with weights
        (np.ones(2), None, np.array([0.45, 0.05])),  # zero error case
    ],
)
def test_pinball_loss_multi_objective__returns_expected_values_with_weights(
    y_pred: np.ndarray, sample_weight: np.ndarray | None, expected_gradient: np.ndarray
) -> None:
    """Test pinball loss gradients for over/under prediction with/without sample weights."""
    # Arrange
    quantiles = [Quantile(0.1), Quantile(0.9)]
    y_true = np.ones(2)

    # Act
    gradient, hessian = pinball_loss_multi_objective(y_true, y_pred, quantiles, sample_weight=sample_weight)

    # Assert
    np.testing.assert_array_almost_equal(gradient, expected_gradient)
    assert np.all(hessian > 0)


@pytest.mark.parametrize(
    ("quantiles", "y_pred", "sample_weight", "expected_gradient"),
    [
        # Different quantiles, overprediction, no weights
        ([Quantile(0.1), Quantile(0.9)], np.ones(2) * 1.5, None, np.array([0.225, 0.025])),
        ([Quantile(0.25), Quantile(0.75)], np.ones(2) * 1.5, None, np.array([0.1875, 0.0625])),
        # Test with weights (2x multiplier)
        ([Quantile(0.1), Quantile(0.9)], np.ones(2) * 1.5, np.array([2.0]), np.array([0.45, 0.05])),
        # Test larger errors produce larger gradients
        ([Quantile(0.1), Quantile(0.9)], np.ones(2) * 2.0, None, np.array([0.45, 0.05])),  # error=1.0
        # Zero error case
        ([Quantile(0.1), Quantile(0.9)], np.ones(2), None, np.array([0.0, 0.0])),
    ],
)
def test_pinball_loss_magnitude_weighted_multi_objective__returns_expected_values_with_weights(
    quantiles: list[Quantile], y_pred: np.ndarray, sample_weight: np.ndarray | None, expected_gradient: np.ndarray
) -> None:
    """Test magnitude-weighted pinball loss with different quantiles, predictions, and weights."""
    # Arrange
    y_true = np.ones(len(quantiles))

    # Act
    gradient, hessian = pinball_loss_magnitude_weighted_multi_objective(
        y_true, y_pred, quantiles, sample_weight=sample_weight
    )

    # Assert
    np.testing.assert_array_almost_equal(gradient, expected_gradient)
    assert np.all(hessian > 0)


@pytest.mark.parametrize(
    ("y_pred", "sample_weight", "expected_gradient_negative"),
    [
        (np.array([0.0, 0.0]), None, np.array([True, True])),  # underprediction, no weights
        (np.array([2.0, 2.0]), None, np.array([False, False])),  # overprediction, no weights
        (np.array([0.0, 0.0]), np.array([2.0]), np.array([True, True])),  # underprediction, with weights
        (np.array([2.0, 2.0]), np.array([2.0]), np.array([False, False])),  # overprediction, with weights
        (np.array([1.0, 1.0]), None, np.array([False, True])),  # zero error case (arctan: [0.2, -0.2])
    ],
)
def test_arctan_loss_multi_objective__returns_expected_values_with_weights(
    y_pred: np.ndarray, sample_weight: np.ndarray | None, expected_gradient_negative: np.ndarray
) -> None:
    """Test arctan loss gradients for under/over prediction with/without sample weights."""
    # Arrange
    quantiles = [Quantile(0.1), Quantile(0.9)]
    y_true = np.array([1.0, 1.0])

    # Act
    gradient, hessian = arctan_loss_multi_objective(y_true, y_pred, quantiles, sample_weight=sample_weight)

    # Assert
    for i, expected_neg in enumerate(expected_gradient_negative):
        if expected_neg:
            assert gradient[i] < 0
        else:
            assert gradient[i] > 0
    assert np.all(hessian > 0)


@pytest.mark.parametrize("s", [0.01, 0.1, 1.0])
def test_arctan_loss_multi_objective__smoothing_parameter_affects_hessian(s: float) -> None:
    """Test that different smoothing parameters s affect hessian magnitude."""
    # Arrange
    y_true = np.array([1.0, 1.0])
    y_pred = np.array([2.0, 2.0])
    quantiles = [Quantile(0.1), Quantile(0.9)]

    # Act
    _, hessian = arctan_loss_multi_objective(y_true, y_pred, quantiles, s=s)

    # Assert
    # Hessians should be positive
    assert np.all(hessian > 0)
    # For smaller s, hessian should be smaller
    if s == 0.01:
        _, hessian_large = arctan_loss_multi_objective(y_true, y_pred, quantiles, s=1.0)
        assert np.all(hessian < hessian_large)


def test_loss_functions__raise_error_on_mismatched_lengths() -> None:
    """Test that functions raise errors for mismatched input lengths."""
    # Arrange
    y_true = np.array([1.0, 1.0, 1.0])  # 3 elements
    y_pred = np.array([2.0, 2.0])  # 2 elements
    quantiles = [Quantile(0.1), Quantile(0.9)]

    # Act & Assert
    with pytest.raises(ValueError):
        pinball_loss_multi_objective(y_true, y_pred, quantiles)

    with pytest.raises(ValueError):
        pinball_loss_magnitude_weighted_multi_objective(y_true, y_pred, quantiles)

    with pytest.raises(ValueError):
        arctan_loss_multi_objective(y_true, y_pred, quantiles)


def test_pinball_loss_magnitude_weighted_multi_objective__quantile_properties() -> None:
    """Test that magnitude-weighted pinball loss respects quantile properties."""
    # Arrange
    y_true = np.array([1.0, 1.0])
    y_pred = np.array([2.0, 2.0])  # overprediction
    quantiles_low = [Quantile(0.1), Quantile(0.1)]
    quantiles_high = [Quantile(0.9), Quantile(0.9)]

    # Act
    grad_low, _ = pinball_loss_magnitude_weighted_multi_objective(y_true, y_pred, quantiles_low)
    grad_high, _ = pinball_loss_magnitude_weighted_multi_objective(y_true, y_pred, quantiles_high)

    # Assert
    # For overprediction, higher quantile should have smaller gradient magnitude
    assert np.abs(grad_high[0]) < np.abs(grad_low[0])


def test_arctan_loss_multi_objective__quantile_properties() -> None:
    """Test that arctan loss respects quantile properties."""
    # Arrange
    y_true = np.array([1.0, 1.0])
    y_pred = np.array([1.5, 1.5])  # overprediction
    quantiles_low = [Quantile(0.1), Quantile(0.1)]
    quantiles_high = [Quantile(0.9), Quantile(0.9)]

    # Act
    grad_low, _ = arctan_loss_multi_objective(y_true, y_pred, quantiles_low)
    grad_high, _ = arctan_loss_multi_objective(y_true, y_pred, quantiles_high)

    # Assert
    # For overprediction, higher quantile should have smaller gradient magnitude
    assert np.abs(grad_high[0]) < np.abs(grad_low[0])


def test_pinball_loss_multi_objective__quantile_properties() -> None:
    """Test that pinball loss respects quantile properties (higher quantile penalizes overprediction more)."""
    # Arrange
    y_true = np.array([1.0, 1.0])
    y_pred = np.array([2.0, 2.0])  # overprediction
    quantiles_low = [Quantile(0.1), Quantile(0.1)]
    quantiles_high = [Quantile(0.9), Quantile(0.9)]

    # Act
    grad_low, _ = pinball_loss_multi_objective(y_true, y_pred, quantiles_low)
    grad_high, _ = pinball_loss_multi_objective(y_true, y_pred, quantiles_high)

    # Assert
    # For overprediction, higher quantile should have smaller gradient magnitude
    assert np.abs(grad_high[0]) < np.abs(grad_low[0])


@pytest.mark.parametrize(
    ("y_true", "y_pred", "description"),
    [
        (np.array([1e-10, 1e-10]), np.array([2e-10, 2e-10]), "very_small_values"),
        (np.array([1e6, 1e6]), np.array([2e6, 2e6]), "very_large_values"),
        (np.array([1.0, 1.0]), np.array([1.0 + 1e-15, 1.0 + 1e-15]), "tiny_differences"),
    ],
)
def test_loss_functions__numerical_stability(y_true: np.ndarray, y_pred: np.ndarray, description: str) -> None:
    """Test numerical stability with extreme values."""
    # Arrange
    quantiles = [Quantile(0.1), Quantile(0.9)]

    # Act & Assert - all functions should handle extreme values without errors
    grad_pinball, hess_pinball = pinball_loss_multi_objective(y_true, y_pred, quantiles)
    grad_weighted, hess_weighted = pinball_loss_magnitude_weighted_multi_objective(y_true, y_pred, quantiles)
    grad_arctan, hess_arctan = arctan_loss_multi_objective(y_true, y_pred, quantiles)

    # All gradients should be finite
    assert np.all(np.isfinite(grad_pinball)), f"Pinball gradients not finite for {description}"
    assert np.all(np.isfinite(grad_weighted)), f"Weighted gradients not finite for {description}"
    assert np.all(np.isfinite(grad_arctan)), f"Arctan gradients not finite for {description}"

    # All hessians should be finite and positive
    assert np.all(np.isfinite(hess_pinball)), f"Pinball hessians not finite for {description}"
    assert np.all(np.isfinite(hess_weighted)), f"Weighted hessians not finite for {description}"
    assert np.all(np.isfinite(hess_arctan)), f"Arctan hessians not finite for {description}"
    assert np.all(hess_pinball > 0), f"Pinball hessians not positive for {description}"
    assert np.all(hess_weighted > 0), f"Weighted hessians not positive for {description}"
    assert np.all(hess_arctan > 0), f"Arctan hessians not positive for {description}"


def test_pinball_loss_multi_objective__medium_length_with_varied_weights() -> None:
    """Test pinball loss with medium-length arrays and varied sample weights."""
    # Arrange
    quantiles = [Quantile(0.1), Quantile(0.5), Quantile(0.9)]
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * len(quantiles))  # 15 elements
    y_pred = np.array([1.2, 1.8, 3.1, 4.2, 4.8] * len(quantiles))  # mixed over/under prediction
    sample_weights = np.array([1.0, 2.0, 0.5, 3.0, 1.5])

    # Act
    grad_weighted, hess_weighted = pinball_loss_multi_objective(y_true, y_pred, quantiles, sample_weight=sample_weights)
    grad_unweighted, _ = pinball_loss_multi_objective(y_true, y_pred, quantiles)

    # Assert
    assert len(grad_weighted) == len(y_true)
    assert len(hess_weighted) == len(y_true)
    assert np.all(hess_weighted > 0)
    # Weighted results should differ from unweighted
    assert not np.allclose(grad_weighted, grad_unweighted)


def test_pinball_loss_magnitude_weighted_multi_objective__medium_length_with_varied_weights() -> None:
    """Test magnitude-weighted pinball loss with medium-length arrays and varied sample weights."""
    # Arrange
    quantiles = [Quantile(0.25), Quantile(0.75)]
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * len(quantiles))  # 10 elements
    y_pred = np.array([1.5, 1.5, 3.5, 4.5, 4.5] * len(quantiles))  # varied errors
    sample_weights = np.array([2.0, 1.0, 3.0, 0.5, 1.5])

    # Act
    grad_weighted, hess_weighted = pinball_loss_magnitude_weighted_multi_objective(
        y_true, y_pred, quantiles, sample_weight=sample_weights
    )
    grad_unweighted, _ = pinball_loss_magnitude_weighted_multi_objective(y_true, y_pred, quantiles)

    # Assert
    assert len(grad_weighted) == len(y_true)
    assert len(hess_weighted) == len(y_true)
    assert np.all(hess_weighted > 0)
    # Larger errors should have larger gradient magnitudes (magnitude-weighted property)
    error_magnitudes = np.abs(y_pred - y_true)
    grad_magnitudes = np.abs(grad_unweighted)
    # Compare gradients for same quantile positions
    for q_idx in range(len(quantiles)):
        q_errors = error_magnitudes[q_idx :: len(quantiles)]
        q_grads = grad_magnitudes[q_idx :: len(quantiles)]
        if len(np.unique(q_errors)) > 1:  # Only test if errors differ
            # Larger errors should generally have larger gradients
            max_error_idx = np.argmax(q_errors)
            min_error_idx = np.argmin(q_errors)
            assert q_grads[max_error_idx] >= q_grads[min_error_idx]


def test_arctan_loss_multi_objective__medium_length_with_varied_weights() -> None:
    """Test arctan loss with medium-length arrays and varied sample weights."""
    # Arrange
    quantiles = [Quantile(0.1), Quantile(0.9)]
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * len(quantiles))  # 10 elements
    y_pred = np.array([0.8, 2.2, 2.9, 4.1, 5.2] * len(quantiles))  # mixed predictions
    sample_weights = np.array([1.5, 0.8, 2.5, 1.2, 3.0])

    # Act
    grad_weighted, hess_weighted = arctan_loss_multi_objective(y_true, y_pred, quantiles, sample_weight=sample_weights)
    grad_unweighted, _ = arctan_loss_multi_objective(y_true, y_pred, quantiles)

    # Assert
    assert len(grad_weighted) == len(y_true)
    assert len(hess_weighted) == len(y_true)
    assert np.all(hess_weighted > 0)
    # Weighted results should differ from unweighted
    assert not np.allclose(grad_weighted, grad_unweighted)


def test_loss_functions__cross_function_consistency() -> None:
    """Test that all loss functions have consistent gradient signs for identical inputs."""
    # Arrange
    quantiles = [Quantile(0.1), Quantile(0.9)]
    test_cases = [
        (np.array([1.0, 1.0]), np.array([0.5, 0.5])),  # underprediction
        (np.array([1.0, 1.0]), np.array([1.5, 1.5])),  # overprediction
        (np.array([1.0, 1.0]), np.array([1.0, 1.0])),  # zero error
    ]

    for y_true, y_pred in test_cases:
        # Act
        grad_pinball, hess_pinball = pinball_loss_multi_objective(y_true, y_pred, quantiles)
        grad_weighted, hess_weighted = pinball_loss_magnitude_weighted_multi_objective(y_true, y_pred, quantiles)
        grad_arctan, hess_arctan = arctan_loss_multi_objective(y_true, y_pred, quantiles)

        # Assert - gradient signs should be consistent across functions
        for i in range(len(grad_pinball)):
            pinball_sign = np.sign(grad_pinball[i])
            weighted_sign = np.sign(grad_weighted[i])
            arctan_sign = np.sign(grad_arctan[i])

            # For zero errors, some functions may have slightly different behavior
            if not np.isclose(y_true[i % len(y_true)], y_pred[i % len(y_pred)]):
                assert pinball_sign == weighted_sign, (
                    f"Gradient sign mismatch between pinball and weighted at index {i}"
                )
                # Note: arctan may have slightly different zero-error behavior, so we're more lenient
                if abs(grad_pinball[i]) > 1e-10:  # Only check sign consistency for non-negligible gradients
                    assert pinball_sign == arctan_sign or abs(grad_arctan[i]) < 1e-6, (
                        f"Gradient sign mismatch between pinball and arctan at index {i}"
                    )

        # All hessians should be positive
        assert np.all(hess_pinball > 0)
        assert np.all(hess_weighted > 0)
        assert np.all(hess_arctan > 0)
