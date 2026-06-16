# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Quantile-grid resampling for probabilistic forecasts.

Some models emit a fixed, model-native grid of quantile levels while callers
request an arbitrary set of quantiles. This module provides a single pure NumPy
helper that resamples predictions from one quantile grid onto another, kept
dependency-free and unit tested in isolation.
"""

from collections.abc import Sequence

import numpy as np

#: Minimum number of source quantile levels required to interpolate between.
_MIN_SOURCE_QUANTILES = 2

__all__ = ["interpolate_quantiles"]


def interpolate_quantiles(
    predictions: np.ndarray,
    source_quantiles: Sequence[float],
    target_quantiles: Sequence[float],
) -> np.ndarray:
    """Resample quantile predictions onto a new quantile grid.

    Performs piecewise-linear interpolation across the quantile dimension (the
    last axis of *predictions*). Target levels outside the source range are
    clamped to the nearest source prediction (constant extrapolation), which
    keeps the resampled values within the predicted envelope.

    Args:
        predictions: Array of shape ``(..., n_source)`` whose last axis holds
            predictions for each level in *source_quantiles*, in the same order.
        source_quantiles: Strictly ascending quantile levels the model emits.
        target_quantiles: Quantile levels to resample onto. Any order.

    Returns:
        Array of shape ``(..., n_target)`` with predictions for each level in
        *target_quantiles*, in the same order.

    Raises:
        ValueError: If *source_quantiles* is not strictly ascending, or its
            length does not match the last axis of *predictions*.
    """
    src = np.asarray(source_quantiles, dtype=np.float64)
    tgt = np.asarray(target_quantiles, dtype=np.float64)

    if src.ndim != 1 or src.shape[0] < _MIN_SOURCE_QUANTILES:
        msg = "source_quantiles must be a 1-D sequence with at least two levels."
        raise ValueError(msg)
    if predictions.shape[-1] != src.shape[0]:
        msg = (
            f"predictions last axis ({predictions.shape[-1]}) must match the number "
            f"of source quantiles ({src.shape[0]})."
        )
        raise ValueError(msg)
    if np.any(np.diff(src) <= 0):
        msg = "source_quantiles must be strictly ascending."
        raise ValueError(msg)

    # Bracket each target level by the adjacent source levels, clamping the
    # endpoints so out-of-range targets extrapolate as constants.
    upper = np.clip(np.searchsorted(src, tgt, side="left"), 1, src.shape[0] - 1)
    lower = upper - 1

    weight = (tgt - src[lower]) / (src[upper] - src[lower])
    weight = np.clip(weight, 0.0, 1.0)

    low_values = predictions[..., lower]
    high_values = predictions[..., upper]
    return low_values * (1.0 - weight) + high_values * weight
