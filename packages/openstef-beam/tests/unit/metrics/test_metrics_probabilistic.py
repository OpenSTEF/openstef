# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from collections.abc import Sequence

import numpy as np
import pytest

from openstef_beam.metrics import crps, mean_absolute_calibration_error, rcrps


# CRPS Test Cases
@pytest.mark.parametrize(
    ("y_true", "y_pred", "quantiles", "expected"),
    [
        # Perfect deterministic forecast (single quantile)
        pytest.param(
            np.array([5.0]),
            np.array([[5.0]]),
            np.array([0.5]),
            0.0,
            id="crps_perfect_deterministic",
        ),
        # Deterministic forecast with fixed offset # CRPS = |5-3| = 2
        pytest.param(
            np.array([5.0]),
            np.array([[3.0]]),
            np.array([0.5]),
            2.0,
            id="crps_offset_deterministic",
        ),
        # Uniform distribution forecast (0-10 range)
        pytest.param(
            np.array([5.0]),
            np.array([np.linspace(0, 10, 11)]),
            np.linspace(0, 1, 11),
            8 / 11,
            id="crps_uniform_distribution",
        ),
        # Two-point distribution (equally likely at 2 and 4)
        pytest.param(
            np.array([3.0]),
            np.array([[2.0, 4.0]]),
            np.array([0.5, 0.5]),
            1.0,
            id="crps_two_point_distribution",
        ),
    ],
)
def test_crps(y_true: Sequence[float], y_pred: Sequence[float], quantiles: Sequence[float], expected: float) -> None:
    # Act
    result = crps(np.array(y_true), np.array(y_pred), np.array(quantiles))

    # Assert
    assert np.isclose(result, expected, rtol=1e-8), f"Expected {expected} but got {result}"


# rCRPS Test Cases
@pytest.mark.parametrize(
    ("y_true", "y_pred", "quantiles", "lower_q", "upper_q", "sample_weights", "expected"),
    [
        # Normalized CRPS with non-zero range,
        pytest.param(
            [0.0, 10.0],  # Range = 10;
            [[5.0]],  # Single quantile forecast at 5.0
            [0.5],
            0.0,
            1.0,
            None,
            0.5,  # CRPS = |5-0| + |5-10| / 2 = 5 → 5/10 = 0.5
            id="rcrps_simple_normalization",
        ),
        # Zero range case
        pytest.param(
            [5.0, 5.0, 5.0],
            [[4.0, 5.0, 6.0]],
            [0.25, 0.5, 0.75],
            0.05,
            0.95,
            None,
            np.nan,
            id="rcrps_zero_range",
        ),
        # Non-zero range with explicit sample weights
        pytest.param(
            [0.0, 10.0, 20.0],  # Range = 20;
            [[5.0]],  # Single quantile forecast at 5.0
            [0.5],
            0.0,
            1.0,
            [0.1, 0.5, 1.0],
            0.5625,  # Weighted CRPS numerator = 0.1*5 + 0.5*5 + 1*15 = 18 → avg = 18/1.6 = 11.25 → 11.25/20 = 0.5625
            id="rcrps_simple_normalization_sample_weights",
        ),
        # Custom sample weights (explicit array matching previous config)
        pytest.param(
            [0.0, 10.0, 20.0],  # Range = 20;
            [[5.0]],  # Single quantile forecast at 5.0
            [0.5],
            0.0,
            1.0,
            [0.3, 0.3, 0.9],
            0.55,  # Weighted CRPS numerator = 0.3*5 + 0.3*5 + 0.9*15 = 16.5 → avg = 16.5/1.5 = 11 → 11/20 = 0.55
            id="rcrps_custom_sample_weights",
        ),
    ],
)
def test_rcrps(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    quantiles: Sequence[float],
    lower_q: float,
    upper_q: float,
    sample_weights: Sequence[float] | None,
    expected: float,
) -> None:
    # Arrange
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    quantiles_arr = np.array(quantiles)
    weights_arg = np.array(sample_weights) if sample_weights is not None else None

    # Act
    result = rcrps(
        y_true_arr,
        y_pred_arr,
        quantiles_arr,
        lower_quantile=lower_q,
        upper_quantile=upper_q,
        sample_weights=weights_arg,
    )

    # Assert
    if np.isnan(expected):
        assert np.isnan(result), f"Expected NaN but got {result}"
    else:
        assert np.isclose(result, expected, rtol=1e-8), f"Expected {expected} but got {result}"


def test_mean_absolute_calibration_error() -> None:
    """Test the mean_absolute_calibration_error with a sample input."""

    # Sample true and predicted values
    y_true = np.array([0.1, 0.2, 0.3, 0.4])
    y_pred = np.repeat(np.array([0.09, 0.19, 0.31, 0.41])[:, np.newaxis], 3, axis=1)
    quantiles = np.array([0.1, 0.5, 0.9])

    # Compute the mean calibration error
    result = mean_absolute_calibration_error(y_true=y_true, y_pred=y_pred, quantiles=quantiles)

    assert isinstance(result, float)
    assert result == (0.4 + 0.4) / 3  # observed probabilities are 0.5, 0.5, 0.5 vs 0.1, 0.5, 0.9 quantiles
