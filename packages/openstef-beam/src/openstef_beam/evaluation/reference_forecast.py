# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Reference forecasts for skill metrics, built from data available at each forecast cutoff."""

import math
from datetime import timedelta

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import ForecastDataset
from openstef_core.types import Quantile


class ReferenceForecastConfig(BaseConfig):
    """Settings for reference forecasts used by skill score metrics."""

    context_length: timedelta = Field(
        default=timedelta(days=14),
        description="Measurement history used for each reference forecast",
    )
    weight_scale_quantile: Quantile = Field(
        default=Quantile(0.99),
        description="Absolute context quantile used to scale sample weights",
    )
    minimum_coverage: float = Field(
        default=0.9,
        description="Minimum fraction of expected context samples required",
    )


class ConstantQuantileReference:
    """Constant quantile forecast derived from a context series.

    Example:
        >>> import pandas as pd
        >>> from openstef_core.types import Quantile
        >>> measurements = pd.Series(
        ...     [0.0, 10.0, 20.0],
        ...     index=pd.date_range("2025-05-01", periods=3, freq="15min"),
        ... )
        >>> index = pd.date_range("2025-05-16", periods=2, freq="15min")
        >>> ConstantQuantileReference.predict(measurements, index, [Quantile(0.5)])
        array([[10.],
               [10.]])
    """

    @staticmethod
    def predict(
        measurements: pd.Series,
        index: pd.DatetimeIndex,
        quantiles: list[Quantile],
    ) -> npt.NDArray[np.floating]:
        """Repeat context quantiles at the requested timestamps."""
        values = np.asarray(measurements.quantile(quantiles), dtype=float)
        return np.tile(values, (len(index), 1))


def build_reference_forecast(
    config: ReferenceForecastConfig,
    measurements: pd.Series,
    subset: ForecastDataset,
    issue_times: pd.Series,
) -> ForecastDataset:
    """Build a reference forecast aligned to an evaluation subset.

    Samples whose forecasts were issued at the same time share one context ending at that
    time, so the reference follows the forecaster's own cadence and never sees later
    measurements. Contexts below the coverage threshold remain NaN.

    Args:
        config: Reference context and coverage settings.
        measurements: Observations used to fit each context; missing values are excluded.
        subset: Forecasts to evaluate.
        issue_times: Time each forecast in the subset was issued, indexed like the subset.

    Returns:
        Reference quantiles and context weight scales aligned to the subset.
    """
    index = subset.index
    quantiles = subset.quantiles
    quantile_columns = [quantile.format() for quantile in quantiles]

    data = pd.DataFrame(np.nan, index=index, columns=[*quantile_columns, "weight_scale"])
    minimum_samples = math.ceil(config.minimum_coverage * config.context_length / subset.sample_interval)

    for issue_time, group in issue_times.groupby(issue_times):
        context = measurements.loc[issue_time - config.context_length : issue_time].dropna()
        if context.empty or len(context) < minimum_samples:
            continue

        group_index = pd.DatetimeIndex(group.index)
        data.loc[group_index, quantile_columns] = ConstantQuantileReference.predict(context, group_index, quantiles)
        data.loc[group_index, "weight_scale"] = np.quantile(np.abs(context.to_numpy()), q=config.weight_scale_quantile)

    return ForecastDataset(data=data, sample_interval=subset.sample_interval, target_column="weight_scale")


__all__ = [
    "ConstantQuantileReference",
    "ReferenceForecastConfig",
    "build_reference_forecast",
]
