# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Forecast mean and standard deviation calculated from available quantiles."""

from typing import override

import numpy as np

from openstef_core.datasets import ForecastDataset
from openstef_core.mixins import Transform

_MIN_QUANTILES_FOR_INTEGRATION = 2


class QuantileMomentsAdder(Transform[ForecastDataset, ForecastDataset]):
    """Add forecast mean and standard deviation over the available probability range.

    Apply after quantile crossing correction. Each row uses its finite quantiles
    and their actual probability levels. The trapezoidal rule integrates the
    quantile values for the mean and the squared deviations for the variance.
    Both integrals are divided by the span from the lowest to the highest valid
    probability. Missing interior points are skipped and the tails are not extrapolated.

    Args:
        min_valid_quantiles: Minimum number of finite quantiles per row. Two
            points are needed to form an integration interval. Both outputs
            are NaN when the row has too few points, has no positive probability
            span, or still contains a quantile crossing.
    """

    def __init__(self, min_valid_quantiles: int = _MIN_QUANTILES_FOR_INTEGRATION) -> None:
        """Set the minimum number of finite points used to estimate the moments."""
        if min_valid_quantiles < _MIN_QUANTILES_FOR_INTEGRATION:
            raise ValueError("min_valid_quantiles must be at least 2.")
        self.min_valid_quantiles = min_valid_quantiles

    @property
    @override
    def is_fitted(self) -> bool:
        return True

    @override
    def fit(self, data: ForecastDataset) -> None:
        pass  # No historical data is needed.

    @override
    def transform(self, data: ForecastDataset) -> ForecastDataset:
        quantiles = sorted(data.quantiles)
        probabilities = np.asarray(quantiles, dtype=float)
        columns = [quantile.format() for quantile in quantiles]
        predictions = data.data[columns].to_numpy(dtype=float, na_value=np.nan)
        means = np.full(len(predictions), np.nan)
        stdevs = np.full(len(predictions), np.nan)
        if predictions.shape[1] >= self.min_valid_quantiles:
            valid = np.isfinite(predictions)
            counts = valid.sum(axis=1)

            # Pack finite values in probability order, then repeat the last valid
            # point so padding contributes zero probability width to the integral.
            order = np.argsort(~valid, axis=1, kind="stable")
            last_valid = np.maximum(counts - 1, 0)
            positions = np.minimum(np.arange(len(quantiles)), last_valid[:, None])
            columns_by_row = np.take_along_axis(order, positions, axis=1)
            levels = probabilities[columns_by_row]
            values = np.take_along_axis(np.where(valid, predictions, 0.0), columns_by_row, axis=1)

            mass = levels[:, -1] - levels[:, 0]
            crossings = (values[:, 1:] < values[:, :-1]).any(axis=1)
            rows = (counts >= self.min_valid_quantiles) & (mass > 0) & ~crossings
            levels = levels[rows]
            values = values[rows]
            mass = mass[rows]

            # Center at the first value to preserve zero variance for constant rows.
            anchor = values[:, 0]
            shifted = values - anchor[:, None]
            mean_shifted = np.trapezoid(shifted, x=levels, axis=1) / mass
            variance = np.trapezoid((shifted - mean_shifted[:, None]) ** 2, x=levels, axis=1) / mass
            means[rows] = anchor + mean_shifted
            stdevs[rows] = np.sqrt(variance)

        return data.pipe_pandas(
            lambda df: df.assign(**{data.forecast_mean_column: means, data.standard_deviation_column: stdevs})
        )


__all__ = ["QuantileMomentsAdder"]
