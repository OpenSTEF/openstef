# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Forecast mean and standard deviation calculated from available quantiles."""

from typing import override

import numpy as np

from openstef_core.datasets import ForecastDataset
from openstef_core.mixins import Transform


class QuantileMoments(Transform[ForecastDataset, ForecastDataset]):
    """Estimate mean and standard deviation over each row's valid probability range.

    Integrates the quantile values and their squared deviations using the
    trapezoidal rule. Apply after quantile crossing correction. Rows with
    fewer than two finite quantiles or remaining crossings yield NaN.
    """

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
        min_points = 2

        for index, row in enumerate(predictions):
            valid = np.isfinite(row)
            values = row[valid]
            if len(values) < min_points or np.any(values[1:] < values[:-1]):
                continue

            levels = probabilities[valid]
            mass = levels[-1] - levels[0]
            if mass <= 0:
                continue

            # Shift first to preserve exact zero variance for constant forecasts.
            shifted = values - values[0]
            mean_shifted = np.trapezoid(shifted, x=levels) / mass
            means[index] = values[0] + mean_shifted
            variance = np.trapezoid((shifted - mean_shifted) ** 2, x=levels) / mass
            stdevs[index] = np.sqrt(variance)

        return data.pipe_pandas(
            lambda df: df.assign(**{data.forecast_mean_column: means, data.standard_deviation_column: stdevs})
        )


__all__ = ["QuantileMoments"]
