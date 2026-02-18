# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import pytest

from openstef_meta.utils.pinball_errors import calculate_pinball_errors


@pytest.fixture
def index() -> pd.DatetimeIndex:
    return pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"])


@pytest.fixture
def y_true(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series([10.0, 20.0, 30.0, 40.0], index=index)


def test_perfect_predictions_zero_loss(y_true: pd.Series):
    """When predictions match actual values exactly, pinball loss is zero everywhere."""
    # Act
    result = calculate_pinball_errors(y_true, y_true, quantile=0.5)

    # Assert
    assert (result == 0).all()


def test_under_prediction_penalized_by_quantile(y_true: pd.Series, index: pd.DatetimeIndex):
    """Under-prediction (y_true > y_pred) is penalized by quantile * error."""
    # Arrange
    y_pred = pd.Series([5.0, 15.0, 25.0, 35.0], index=index)  # all under-predict by 5
    quantile = 0.9

    # Act
    result = calculate_pinball_errors(y_true, y_pred, quantile=quantile)

    # Assert — errors = y_true - y_pred = 5, pinball = 0.9 * 5 = 4.5
    expected = pd.Series([4.5, 4.5, 4.5, 4.5], index=index)
    pd.testing.assert_series_equal(result, expected)


def test_over_prediction_penalized_by_complement(y_true: pd.Series, index: pd.DatetimeIndex):
    """Over-prediction (y_true < y_pred) is penalized by (1 - quantile) * |error|."""
    # Arrange
    y_pred = pd.Series([15.0, 25.0, 35.0, 45.0], index=index)  # all over-predict by 5
    quantile = 0.9

    # Act
    result = calculate_pinball_errors(y_true, y_pred, quantile=quantile)

    # Assert — errors = y_true - y_pred = -5, pinball = (0.9 - 1) * (-5) = 0.5
    expected = pd.Series([0.5, 0.5, 0.5, 0.5], index=index)
    pd.testing.assert_series_equal(result, expected)


def test_median_quantile_symmetric(y_true: pd.Series, index: pd.DatetimeIndex):
    """At quantile 0.5, under- and over-prediction penalties are symmetric."""
    # Arrange
    y_under = pd.Series([5.0, 15.0, 25.0, 35.0], index=index)
    y_over = pd.Series([15.0, 25.0, 35.0, 45.0], index=index)

    # Act
    loss_under = calculate_pinball_errors(y_true, y_under, quantile=0.5)
    loss_over = calculate_pinball_errors(y_true, y_over, quantile=0.5)

    # Assert
    pd.testing.assert_series_equal(loss_under, loss_over)


def test_result_preserves_index(y_true: pd.Series, index: pd.DatetimeIndex):
    """Output Series has the same index as the input y_true."""
    # Arrange
    y_pred = pd.Series([10.0, 20.0, 30.0, 40.0], index=index)

    # Act
    result = calculate_pinball_errors(y_true, y_pred, quantile=0.5)

    # Assert
    pd.testing.assert_index_equal(result.index, index)


def test_pinball_loss_is_non_negative(y_true: pd.Series, index: pd.DatetimeIndex):
    """Pinball loss should always be >= 0 for any quantile."""
    # Arrange
    rng = np.random.default_rng(42)
    y_pred = pd.Series(rng.normal(25, 15, size=len(y_true)), index=index)

    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        # Act
        result = calculate_pinball_errors(y_true, y_pred, quantile=q)

        # Assert
        assert (result >= 0).all(), f"Negative pinball loss found at quantile {q}"
