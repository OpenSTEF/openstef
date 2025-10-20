from datetime import timedelta

import pandas as pd

from openstef_core.datasets.mixins import TimeSeriesMixin, VersionedTimeSeriesMixin
from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.types import LeadTime


def calculate_time_coverage(dataset: TimeSeriesMixin) -> timedelta:
    """Calculates the total time span covered by the dataset.

    This method computes the total duration represented by the dataset
    based on its unique timestamps and sample interval.

    Returns:
        timedelta: Total time coverage of the dataset.
    """
    return len(dataset.index.unique()) * dataset.sample_interval


def versioned_to_horizon_split(
    dataset: VersionedTimeSeriesMixin,
    horizons: list[LeadTime],
) -> TimeSeriesDataset:
    horizon_dfs = pd.concat(
        objs=[
            dataset.filter_by_lead_time(lead_time=horizon).select_version().data.assign(horizon=horizon.value)
            for horizon in horizons
        ]
    )
    return TimeSeriesDataset(
        data=horizon_dfs,
        sample_interval=dataset.sample_interval,
    )
