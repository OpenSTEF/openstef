# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from collections.abc import Sequence

import numpy as np
import pytest

from openstef_beam.metrics.metrics_helpers import represented_interval_weights
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
