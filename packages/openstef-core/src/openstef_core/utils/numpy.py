# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Pure NumPy helpers for probabilistic forecasting."""

import logging
from collections.abc import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def zero_fill_with_mask(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split an array into a zero-filled copy and a finiteness mask.

    Non-finite entries (``NaN`` or infinities) are replaced with ``0.0`` in the
    returned values; the mask is ``1.0`` exactly where the original entry was
    finite. This is the common "feed raw values, tell the model which are real"
    step for masked model inputs.

    Args:
        values: Array of any shape.

    Returns:
        Tuple ``(filled, mask)``, both ``float32`` and the same shape as
        *values*. ``filled`` has non-finite entries zeroed; ``mask`` is ``1.0``
        where *values* was finite, ``0.0`` otherwise.
    """
    finite = np.isfinite(values)
    filled = np.where(finite, values, np.float32(0.0)).astype(np.float32)
    return filled, finite.astype(np.float32)


def interpolate_quantiles(
    predictions: np.ndarray,
    source_quantiles: Sequence[float],
    target_quantiles: Sequence[float],
) -> np.ndarray:
    """Resample quantile predictions onto a new quantile grid.

    Performs piecewise-linear interpolation across the quantile dimension (the
    last axis of *predictions*). Target levels outside the source range are
    clamped to the nearest source prediction (constant extrapolation), which
    keeps the resampled values within the predicted envelope. Because the model
    cannot say anything beyond its most extreme level, a request for, say, q0.999
    against a model whose highest level is q0.99 returns the q0.99 prediction; a
    warning is logged whenever this clamping happens.

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

    min_levels = 2  # need at least two source levels to interpolate between
    if src.ndim != 1 or src.shape[0] < min_levels:
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

    out_of_range = tgt[(tgt < src[0]) | (tgt > src[-1])]
    if out_of_range.size:
        logger.warning(
            "Target quantile level(s) %s lie outside the source range [%s, %s]; "
            "clamping to the nearest source quantile (constant extrapolation).",
            np.unique(out_of_range).tolist(),
            src[0],
            src[-1],
        )

    # Bracket each target level by the adjacent source levels, clamping the
    # endpoints so out-of-range targets extrapolate as constants.
    upper = np.clip(np.searchsorted(src, tgt, side="left"), 1, src.shape[0] - 1)
    lower = upper - 1

    weight = (tgt - src[lower]) / (src[upper] - src[lower])
    weight = np.clip(weight, 0.0, 1.0)

    low_values = predictions[..., lower]
    high_values = predictions[..., upper]
    return low_values * (1.0 - weight) + high_values * weight


__all__ = ["interpolate_quantiles", "zero_fill_with_mask"]
