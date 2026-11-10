# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Reference forecasts for skill score based evaluation metrics.

Skill scores express forecast quality relative to a simple reference forecast rather
than on an absolute scale. This module builds such reference forecasts from the
measurements that were available at the moment each forecast was made.

The reference is recomputed at a fixed cadence, mirroring how forecasts are produced
operationally: a forecast made on a given day can only use measurements up to its
availability cutoff.
"""

from datetime import datetime, timedelta
from typing import override

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field

from openstef_beam.evaluation.models.window import Filtering
from openstef_core.base_model import BaseConfig
from openstef_core.datasets import ForecastDataset
from openstef_core.types import AvailableAt, Quantile


class ReferenceForecast(BaseConfig):
    """Base class for reference forecasts used by skill score metrics.

    Subclasses implement predict() to derive quantile predictions from the
    measurements of a single context window.
    """

    def predict(
        self,
        measurements: pd.Series,
        index: pd.DatetimeIndex,
        quantiles: list[Quantile],
    ) -> npt.NDArray[np.floating]:
        """Predict quantile values for the given timestamps.

        Args:
            measurements: Observed values of the context window, without missing values.
            index: Timestamps to predict for.
            quantiles: Quantile levels to predict, in ascending order.

        Returns:
            Predicted quantiles with shape (num_timestamps, num_quantiles).
        """
        msg = "Subclasses should implement this method."
        raise NotImplementedError(msg)


class ConstantQuantileReference(ReferenceForecast):
    """Reference forecast that repeats the quantiles of the context window.

    Predicts a constant distribution derived from the observed values of the context
    window, ignoring any time-dependent structure. This is the simplest non-trivial
    probabilistic baseline: any skill over it comes from predicting variation rather
    than the overall load level and spread.

    Example:
        >>> import pandas as pd
        >>> from openstef_core.types import Quantile
        >>> measurements = pd.Series(
        ...     [0.0, 10.0, 20.0],
        ...     index=pd.date_range("2025-05-01", periods=3, freq="15min"),
        ... )
        >>> index = pd.date_range("2025-05-16", periods=2, freq="15min")
        >>> ConstantQuantileReference().predict(measurements, index, [Quantile(0.5)])
        array([[10.],
               [10.]])
    """

    @override
    def predict(
        self,
        measurements: pd.Series,
        index: pd.DatetimeIndex,
        quantiles: list[Quantile],
    ) -> npt.NDArray[np.floating]:
        values = np.asarray(measurements.quantile(list(quantiles)), dtype=float)
        return np.tile(values, (len(index), 1))


class ReferenceForecastConfig(BaseConfig):
    """Configuration of the reference forecast that skill score metrics compare against."""

    forecast: ReferenceForecast = Field(
        default_factory=ConstantQuantileReference,
        description="Reference forecast fitted on the measurements preceding each forecast moment",
    )
    context_length: timedelta = Field(
        default=timedelta(days=14),
        description="Duration of measurements each reference forecast is based on",
    )
    update_interval: timedelta = Field(
        default=timedelta(days=1),
        description="Cadence at which reference forecasts are recomputed. Should match the cadence "
        "at which the evaluated forecasts were made",
    )
    weight_scale_quantile: Quantile = Field(
        default=Quantile(0.99),
        description="Additional quantile predicted by the reference forecast, used by sample weighted "
        "skill score metrics to anchor their weights on the context period",
    )


def build_reference_forecast(
    config: ReferenceForecastConfig,
    measurements: pd.Series,
    subset: ForecastDataset,
    filtering: Filtering,
) -> ForecastDataset:
    """Build the reference forecast for an evaluation subset.

    Timestamps are grouped by the moment their input data became available, rounded
    down to the update interval. Each group is predicted by a reference fitted on the
    measurements of the context window preceding that moment. To avoid using
    measurements that were not yet available, a group's context window ends at the
    earliest availability cutoff within that group.

    Args:
        config: Settings controlling the reference forecast and its context windows.
        measurements: Observed values over the evaluation period and the context
            preceding it, without missing values.
        subset: Evaluation subset to build the reference forecast for.
        filtering: Filtering criteria the subset was selected with, determining when
            the input data of each forecast became available.

    Returns:
        Reference predictions aligned to the subset, NaN where the context window
        contained no measurements. Covers the quantiles of the subset plus the
        configured weight scale quantile.
    """
    index = subset.index
    quantiles = sorted({*subset.quantiles, config.weight_scale_quantile})
    if isinstance(filtering, AvailableAt):
        cutoffs = pd.Series(filtering.apply_index(index), index=index)
    else:
        cutoffs = pd.Series(index - filtering.value, index=index)

    positions = pd.Series(np.arange(len(index)), index=index)
    predictions = np.full((len(index), len(quantiles)), np.nan)

    for _, group in cutoffs.groupby(cutoffs.dt.floor(config.update_interval)):
        cutoff: datetime = group.min()
        context = measurements.loc[cutoff - config.context_length : cutoff]
        if context.empty:
            continue

        group_index = pd.DatetimeIndex(group.index)
        predictions[positions[group_index].to_numpy()] = config.forecast.predict(
            measurements=context,
            index=group_index,
            quantiles=quantiles,
        )

    return ForecastDataset.from_quantile_predictions(
        predictions=predictions,
        index=index,
        quantiles=quantiles,
        sample_interval=subset.sample_interval,
    )


__all__ = [
    "ConstantQuantileReference",
    "ReferenceForecast",
    "ReferenceForecastConfig",
    "build_reference_forecast",
]
