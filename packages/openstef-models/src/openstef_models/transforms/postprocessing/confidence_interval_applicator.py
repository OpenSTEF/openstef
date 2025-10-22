# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Confidence interval generation for probabilistic forecasts.

This module provides transforms for adding quantile predictions to forecasts based on
learned hour-specific uncertainty patterns from validation data.
"""

from datetime import datetime
from typing import Any, Self, cast, override

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr
from scipy import stats

from openstef_core.datasets import ForecastDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.mixins import State, Transform
from openstef_core.types import LeadTime, Quantile
from openstef_core.utils.invariants import not_none


class ConfidenceIntervalApplicator(BaseModel, Transform[ForecastDataset, ForecastDataset]):
    """Add quantile predictions to forecasts based on learned uncertainty patterns.

    This transform learns hour-specific uncertainty from validation errors and applies
    it to new predictions to generate probabilistic forecasts.

    How it works:
        1. **Learning phase (fit)**:
           - Computes validation errors for each hour of day (0-23)
           - Calculates standard deviation for each hour
           - For multi-horizon data, stores separate std for each horizon

        2. **Prediction phase (transform)**:
           - Looks up appropriate std based on prediction hour
           - For multi-horizon: interpolates std using exponential decay
           - Converts std to quantiles assuming normal distribution:
             quantile_value = median + z_score * std
             (e.g., P10 = median - 1.28*std, P90 = median + 1.28*std)

    The exponential decay interpolation (for multi-horizon) uses:
        sigma(t) = a * (1 - exp(-t/tau)) + b
    where t is hours ahead, tau = far_horizon/4. This reflects the pattern
    of uncertainty growing quickly at first then leveling off.

    Args:
        quantiles: Quantiles to generate (e.g., [0.1, 0.5, 0.9]).
            If None, returns predictions unchanged.

    Invariants:
        - Validation data must span multiple days for reliable hourly statistics
        - Generated quantiles satisfy ordering: P10 <= P50 <= P90 (for typical quantiles)

    Example:
        >>> from openstef_core.types import LeadTime, Quantile
        >>> from datetime import timedelta
        >>> applicator = ConfidenceIntervalApplicator(
        ...     quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)]
        ... )
        >>> applicator.fit((validation_data, validation_predictions))  # doctest: +SKIP
        >>> result = applicator.transform((new_input_data, new_predictions))  # doctest: +SKIP
        >>> list(result.data.columns)  # doctest: +SKIP
        ['quantile_P10', 'quantile_P50', 'quantile_P90']

    Note:
        Assumes forecast errors follow a normal distribution. This works well
        for energy forecasting but may not suit all domains.
    """

    quantiles: list[Quantile] | None = Field(default=None)

    _standard_deviation: pd.DataFrame = PrivateAttr(default_factory=pd.DataFrame)
    _is_fitted: bool = PrivateAttr(default=False)

    @property
    def standard_deviation(self) -> pd.DataFrame:
        """Hour-specific standard deviations learned during fit.

        Raises:
            NotFittedError: If fit() has not been called.
        """
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)
        return self._standard_deviation

    @property
    @override
    def is_fitted(self) -> bool:
        return self._is_fitted

    @override
    def fit(self, data: ForecastDataset) -> None:
        if data.target_series is None:
            raise ValueError("Input data must contain target series for error computation.")

        # Compute hourly standard deviation for each horizon
        std_by_horizon: dict[str, pd.Series] = {}
        for horizon in data.horizons or [LeadTime.from_string("PT0H")]:
            horizon_data = data.select_horizon(horizon)
            # Compute hourly standard deviation for each horizon
            errors = not_none(horizon_data.target_series) - horizon_data.median_series
            # Group by hour and compute std
            std_by_horizon[str(horizon)] = errors.pipe(_calculate_hourly_std)

        # Group by hour and compute std
        self._standard_deviation = pd.DataFrame(std_by_horizon)
        self._is_fitted = True

    def _compute_stdev_series(self, forecast: ForecastDataset) -> pd.Series:
        horizon_strs = self._standard_deviation.columns.to_list()
        horizons = [LeadTime.from_string(h) for h in horizon_strs]

        # Single horizon: direct lookup by hour
        if len(horizons) == 1:
            return _apply_hourly_stdev(index=forecast.index, stdev_pivot=self._standard_deviation, horizon=horizons[0])

        # Multi-horizon: interpolate between near and far
        return _apply_interpolated_stdev(
            index=forecast.index,
            forecast_start=forecast.forecast_start,
            stdev_pivot=self._standard_deviation,
            horizon_near=min(horizons),
            horizon_far=max(horizons),
        )

    @staticmethod
    def _add_quantiles_from_stdev(
        forecast: ForecastDataset, stdev_series: pd.Series, quantiles: list[Quantile]
    ) -> ForecastDataset:
        # quantile = median + z_score * std (normal distribution assumption)
        data = forecast.data.copy(deep=False)
        for quantile in quantiles:
            data[quantile.format()] = forecast.median_series + stats.norm.ppf(quantile) * stdev_series  # pyright: ignore[reportUnknownMemberType]

        return forecast._copy_with_data(data=data)  # noqa: SLF001 - safe - invariant preserved

    @override
    def transform(self, data: ForecastDataset) -> ForecastDataset:
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)

        if self.quantiles is None:
            return data

        # Compute standard deviation series
        stdev_series = self._compute_stdev_series(data)

        # Add quantiles based on standard deviation
        return self._add_quantiles_from_stdev(forecast=data, stdev_series=stdev_series, quantiles=self.quantiles)

    @override
    def to_state(self) -> State:
        return cast(
            State,
            {
                "standard_deviation": self._standard_deviation.to_dict(orient="tight"),  # pyright: ignore[reportUnknownMemberType]
                "is_fitted": self._is_fitted,
            },
        )

    @override
    def from_state(self, state: State) -> Self:
        state_dict = cast(dict[str, Any], state)
        self._standard_deviation = pd.DataFrame.from_dict(state_dict["standard_deviation"], orient="tight")
        self._is_fitted = state_dict["is_fitted"]
        return self


def _calculate_hourly_std(errors: pd.Series) -> pd.Series:
    # Group errors by hour (0-23) and compute std for each hour
    return (
        errors.groupby(cast(pd.DatetimeIndex, errors.index).hour)  # pyright: ignore[reportUnknownMemberType]
        .std()
        .reindex(range(24))
        .rename("stdev")
        .rename_axis("hour")
    )


def _apply_hourly_stdev(index: pd.DatetimeIndex, stdev_pivot: pd.DataFrame, horizon: LeadTime) -> pd.Series:
    # Look up std for each timestamp based on its hour of day
    return pd.Series(
        data=[stdev_pivot.loc[hour, str(horizon)] for hour in index.hour],
        index=index,
        name="stdev",
    )


def _apply_interpolated_stdev(
    index: pd.DatetimeIndex,
    forecast_start: datetime,
    stdev_pivot: pd.DataFrame,
    horizon_near: LeadTime,
    horizon_far: LeadTime,
) -> pd.Series:
    # Interpolate std between near/far horizons using exponential decay
    time_ahead_hours = (index - forecast_start).total_seconds() / 3600.0

    # Vectorized lookup: get all near/far stdev values at once
    stdev_near = stdev_pivot.loc[index.hour, str(horizon_near)].to_numpy()  # type: ignore
    stdev_far = stdev_pivot.loc[index.hour, str(horizon_far)].to_numpy()  # type: ignore

    interpolated = _interpolate_stdev_exponential_decay(
        time_ahead=time_ahead_hours.to_numpy(),
        stdev_near=stdev_near,
        stdev_far=stdev_far,
        horizon_near=horizon_near.to_hours(),
        horizon_far=horizon_far.to_hours(),
    )

    return pd.Series(data=interpolated, index=index, name="stdev")


def _interpolate_stdev_exponential_decay(
    time_ahead: np.ndarray | float,
    stdev_near: np.ndarray | float,
    stdev_far: np.ndarray | float,
    horizon_near: float,
    horizon_far: float,
) -> np.ndarray | float:
    # Exponential decay model: sigma(t) = a * (1 - exp(-t/tau)) + b
    # where tau = far_horizon/4 is the decay time constant
    tau = horizon_far / 4.0
    denominator = (1 - np.exp(-horizon_far / tau)) - (1 - np.exp(-horizon_near / tau))
    a = (stdev_far - stdev_near) / denominator
    b = stdev_near - a * (1 - np.exp(-horizon_near / tau))
    value = a * (1 - np.exp(-time_ahead / tau)) + b
    return np.clip(value, stdev_near, stdev_far)


__all__ = ["ConfidenceIntervalApplicator"]
