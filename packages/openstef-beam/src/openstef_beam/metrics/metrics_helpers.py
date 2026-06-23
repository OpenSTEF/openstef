# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Helper functions shared between probabilistic and deterministic metrics."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from openstef_core.types import Quantile


def represented_interval_weights(
    quantiles: Sequence[Quantile],
) -> npt.NDArray[np.floating]:
    """Calculate the probability interval each quantile represents on [0, 1].

    Interval boundaries are placed at the midpoints between consecutive quantiles,
    with the outer edges fixed at 0 and 1. Each quantile is weighted by the width of
    its interval.

    Args:
        quantiles: Quantile levels with shape (num_quantiles,). Must be sorted in
            ascending order and contain values in (0, 1).

    Returns:
        The interval weights with shape (num_quantiles,). Weights are non-negative
        and sum to 1.

    Example:
        Unevenly spaced quantiles get different weights

        >>> import numpy as np
        >>> quantiles = np.array([0.05, 0.1, 0.5, 0.9, 0.95])
        >>> represented_interval_weights(quantiles)
        array([0.075, 0.225, 0.4  , 0.225, 0.075])
    """
    q = np.asarray(quantiles, dtype=float).reshape(-1)

    boundaries = np.empty(len(q) + 1, dtype=float)
    boundaries[0] = 0.0
    boundaries[-1] = 1.0
    boundaries[1:-1] = 0.5 * (q[:-1] + q[1:])

    return np.diff(boundaries)
