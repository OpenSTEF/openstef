# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Helper functions shared between probabilistic and deterministic metrics."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from openstef_core.types import Quantile


def equals_zero(number: float) -> bool:
    """Check if a floating-point number is effectively zero, considering floating-point precision."""
    return bool(np.isclose(number, 0.0, rtol=1e-09, atol=1e-09))


def skill_score(score: float, reference_score: float, *, as_percentage: bool = True) -> float:
    """Calculate the skill of a score relative to a reference score.

    A skill score expresses how much better a forecast is than a reference forecast.
    It is only meaningful for non-negative metrics where lower values are better,
    such as CRPS or MAE.

    Interpretation:

    - Equal to the reference: 0
    - Perfect forecast (score of 0): 1 (or 100%)
    - Twice as accurate as the reference: 0.5 (or 50%)
    - Worse than the reference: negative

    Args:
        score: The score of the evaluated forecast.
        reference_score: The score of the reference forecast.
        as_percentage: Whether to express the result as a percentage instead of a fraction.

    Returns:
        The skill score. Returns NaN if either input is NaN or if the reference
        score is zero.

    Example:
        Compare a forecast against a reference forecast

        >>> skill_score(0.5, 1.0)
        50.0
        >>> skill_score(1.0, 1.0)
        0.0
        >>> skill_score(2.0, 1.0)
        -100.0
        >>> skill_score(0.5, 1.0, as_percentage=False)
        0.5
    """
    if np.isnan(score) or np.isnan(reference_score) or equals_zero(reference_score):
        return float("NaN")

    skill = 1.0 - score / reference_score
    return float(skill * 100.0) if as_percentage else float(skill)


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
