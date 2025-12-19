# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Testing utilities for comparing pandas objects.

Provides matcher classes for use in test assertions when comparing pandas
DataFrames and Series with equality semantics.
"""

from datetime import datetime, timedelta
from typing import Any, override

import numpy as np
import pandas as pd

from openstef_core.datasets import TimeSeriesDataset


class IsSamePandas:
    """Utility class to allow comparison of pandas DataFrames in assertion / calls."""

    def __init__(self, pandas_obj: pd.DataFrame | pd.Series):
        """Matcher to check if two DataFrames are equal."""
        self.pandas_obj = pandas_obj

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self.pandas_obj)) and self.pandas_obj.equals(other)  # type: ignore

    @override
    def __hash__(self) -> int:
        return hash(self.pandas_obj)


def assert_timeseries_equal(actual: TimeSeriesDataset, expected: TimeSeriesDataset):
    """Assert that two TimeSeriesDataset objects are equal."""
    pd.testing.assert_frame_equal(actual.data, expected.data)
    assert actual.sample_interval == expected.sample_interval, (  # noqa: S101 - exception - testing utility
        f"Sample intervals differ: {actual.sample_interval} != {expected.sample_interval}"
    )


def create_timeseries_dataset(
    index: pd.DatetimeIndex,
    available_ats: pd.Series | list[datetime] | pd.DatetimeIndex | None = None,
    horizons: pd.Series | list[timedelta] | None = None,
    sample_interval: timedelta = timedelta(hours=1),
    **kwargs: pd.Series | list[Any] | pd.DatetimeIndex,
) -> TimeSeriesDataset:
    """Create a TimeSeriesDataset for testing purposes.

    Args:
        index: Datetime index for the dataset.
        available_ats: Optional available_at timestamps for each data point.
        horizons: Optional forecast horizons for each data point.
        sample_interval: Time interval between consecutive samples.
        **kwargs: Additional columns to include in the dataset.

    Returns:
        TimeSeriesDataset with the specified structure.
    """
    data = kwargs
    if available_ats is not None:
        data["available_at"] = available_ats
    elif horizons is not None:
        data["horizon"] = horizons

    return TimeSeriesDataset(data=pd.DataFrame(data=data, index=index), sample_interval=sample_interval)


def create_synthetic_forecasting_dataset(  # noqa: PLR0913, PLR0917 - complex function - testing utility
    start: datetime = datetime.fromisoformat("2025-01-01T00:00:00+00:00"),  # noqa: B008
    length: timedelta = timedelta(days=30 * 9),
    sample_interval: timedelta = timedelta(hours=1),
    random_seed: int = 42,
    wind_influence: float | None = -0.2,
    temp_influence: float | None = 0.3,
    radiation_influence: float | None = -0.2,
    stochastic_influence: float | None = 0.1,
    other_components: dict[str, float] | None = None,
) -> TimeSeriesDataset:
    """Create synthetic forecasting dataset for testing.

    Generates time series data with configurable components influencing load.

    Args:
        start: Start datetime for the dataset.
        length: Total duration of the dataset.
        sample_interval: Time interval between consecutive samples.
        random_seed: Random seed for reproducible random components.
        wind_influence: Coefficient for wind speed component on load.
        temp_influence: Coefficient for temperature component on load.
        radiation_influence: Coefficient for radiation component on load.
        stochastic_influence: Coefficient for random noise component.
        other_components: Additional components with their influence coefficients.

    Returns:
        TimeSeriesDataset containing synthetic load and component data.
    """
    timestamps = pd.date_range(start=start, periods=length // sample_interval, freq=sample_interval, tz="UTC")

    # Build load as a combination of various components
    component_influence = other_components or {}
    if wind_influence is not None:
        component_influence["windspeed"] = wind_influence
    if temp_influence is not None:
        component_influence["temperature"] = temp_influence
    if radiation_influence is not None:
        component_influence["radiation"] = radiation_influence
    if stochastic_influence is not None:
        component_influence["stochastic"] = stochastic_influence

    rng = np.random.default_rng(random_seed)
    load = pd.Series(np.zeros(len(timestamps)), index=timestamps, name="load")
    components: dict[str, pd.Series] = {}
    for component_name, influence in component_influence.items():
        component = pd.Series(rng.standard_normal(size=len(timestamps)), index=timestamps, name=component_name)
        load += component * influence
        components[component_name] = component

    return TimeSeriesDataset(
        data=pd.DataFrame(
            data={
                "load": load,
                **components,
            },
            index=timestamps,
        ),
        sample_interval=sample_interval,
    )
