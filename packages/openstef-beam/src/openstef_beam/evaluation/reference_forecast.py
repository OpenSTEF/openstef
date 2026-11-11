# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Reference forecasts for skill metrics, built from data available at each forecast cutoff."""

import math
from datetime import datetime, timedelta

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field

from openstef_beam.evaluation.models.window import Filtering
from openstef_core.base_model import BaseConfig
from openstef_core.datasets import ForecastDataset
from openstef_core.types import AvailableAt, Quantile


class ReferenceForecastConfig(BaseConfig):
    """Settings for reference forecasts used by skill score metrics."""

    context_length: timedelta = Field(
        default=timedelta(days=14),
        description="Measurement history used for each reference forecast",
    )
    update_interval: timedelta = Field(
        default=timedelta(hours=6),
        description="Reference recomputation cadence; ideally matches the forecast cadence",
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
    filtering: Filtering,
) -> ForecastDataset:
    """Build a reference forecast aligned to an evaluation subset.

    Group timestamps by their availability cutoff, rounded down to ``update_interval``.
    Each group uses the context ending at its earliest cutoff; groups below the coverage
    threshold remain NaN.

    Args:
        config: Reference context, cadence, and coverage settings.
        measurements: Observations used to fit each context; missing values are excluded.
        subset: Forecasts to evaluate.
        filtering: Rule for determining when forecast inputs became available.

    Returns:
        Reference quantiles and context weight scales aligned to the subset.
    """
    index = subset.index
    quantiles = subset.quantiles
    quantile_columns = [quantile.format() for quantile in quantiles]
    if isinstance(filtering, AvailableAt):
        cutoffs = pd.Series(filtering.apply_index(index), index=index)
    else:
        cutoffs = pd.Series(index - filtering.value, index=index)

    data = pd.DataFrame(np.nan, index=index, columns=[*quantile_columns, "weight_scale"])
    minimum_samples = math.ceil(config.minimum_coverage * config.context_length / subset.sample_interval)

    for _, group in cutoffs.groupby(cutoffs.dt.floor(config.update_interval)):
        cutoff: datetime = group.min()
        context = measurements.loc[cutoff - config.context_length : cutoff].dropna()
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
