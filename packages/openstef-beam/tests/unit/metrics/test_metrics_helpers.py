# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from collections.abc import Sequence

import numpy as np
import pytest

from openstef_beam.metrics.metrics_helpers import SampleWeighter, represented_interval_weights, skill_score
from openstef_core.types import Q


@pytest.mark.parametrize(
    ("quantiles", "expected"),
    [
        # Three symmetric quantiles: the median represents the [0.3, 0.7] interval while each tail represents 0.3.
        pytest.param([0.1, 0.5, 0.9], [0.3, 0.4, 0.3], id="three_symmetric"),
        # Unequally spaced quantiles: the outer quantiles represent wider probability intervals than the inner ones.
        pytest.param([0.05, 0.1, 0.5, 0.9, 0.95], [0.075, 0.225, 0.4, 0.225, 0.075], id="unequal_spacing"),
        # Equally spaced quantiles: each quantile represents the same probability interval of 0.2.
        pytest.param([0.1, 0.3, 0.5, 0.7, 0.9], [0.2, 0.2, 0.2, 0.2, 0.2], id="equal_spacing"),
    ],
)
def test_represented_interval_weights(quantiles: Sequence[float], expected: Sequence[float]) -> None:
    # Act
    weights = represented_interval_weights([Q(q) for q in quantiles])

    # Assert
    assert np.allclose(weights, expected)
    assert np.isclose(weights.sum(), 1.0)  # weights form a partition of the unit interval


@pytest.mark.parametrize(
    ("scale", "expected"),
    [
        pytest.param(4.0, [0.25, 0.5, 1.0], id="scalar_scale"),
        pytest.param([2.0, 4.0, 8.0], [0.5, 0.5, 0.5], id="vector_scale"),
    ],
)
def test_sample_weighter_calculate_weights_returns_clipped_weights(
    scale: float | Sequence[float], expected: Sequence[float]
) -> None:
    # Arrange
    weighter = SampleWeighter()
    values = np.array([1.0, 2.0, -4.0])

    # Act
    weights = weighter.calculate_weights(values, np.asarray(scale))

    # Assert
    assert np.allclose(weights, expected)


@pytest.mark.parametrize(
    ("score", "reference_score", "expected"),
    [
        # A perfect forecast has no error left to remove, so all of the reference error is skill.
        pytest.param(0.0, 2.0, 100.0, id="perfect_forecast"),
        # Matching the reference forecast means no skill was added.
        pytest.param(2.0, 2.0, 0.0, id="equal_to_reference"),
        # Halving the reference error is half of the way towards a perfect forecast.
        pytest.param(1.0, 2.0, 50.0, id="twice_as_accurate"),
        # Doubling the reference error is as far below the reference as a perfect forecast is above it.
        pytest.param(4.0, 2.0, -100.0, id="worse_than_reference"),
        # A reference without error cannot be improved upon, so skill is undefined.
        pytest.param(1.0, 0.0, np.nan, id="zero_reference"),
        pytest.param(np.nan, 2.0, np.nan, id="nan_score"),
        pytest.param(1.0, np.nan, np.nan, id="nan_reference"),
    ],
)
def test_skill_score_returns_expected_value(score: float, reference_score: float, expected: float) -> None:
    # Arrange: score pairs are supplied by the parametrization

    # Act
    result = skill_score(score, reference_score)

    # Assert
    if np.isnan(expected):
        assert np.isnan(result)
    else:
        assert result == pytest.approx(expected)
