from datetime import timedelta
from typing import Any

import pandas as pd

from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset


def create_test_timeseries_dataset(
    index: pd.DatetimeIndex,
    available_ats: pd.Series | pd.DatetimeIndex | None = None,
    horizons: pd.Series | None = None,
    sample_interval: timedelta = timedelta(hours=1),
    **kwargs: pd.Series | list[Any] | pd.DatetimeIndex,
) -> TimeSeriesDataset:
    data = kwargs
    if available_ats is not None:
        data["available_at"] = available_ats
    elif horizons is not None:
        data["horizon"] = horizons
    else:
        raise ValueError("Either available_ats or horizons must be provided")

    return TimeSeriesDataset(data=pd.DataFrame(data=data, index=index), sample_interval=sample_interval)
