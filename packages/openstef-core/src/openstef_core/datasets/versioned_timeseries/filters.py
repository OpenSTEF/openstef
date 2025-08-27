# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Filtering functions for versioned time series datasets.

Provides utilities to filter versioned forecasts by availability time, lead time,
and other temporal criteria. Essential for creating realistic evaluation scenarios
that respect information availability constraints in operational forecasting.
"""

import pandas as pd

from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.datasets.versioned_timeseries.dataset import VersionedTimeSeriesDataset
from openstef_core.types import AvailableAt, LeadTime


def filter_by_available_at(
    dataset: VersionedTimeSeriesDataset,
    available_at: AvailableAt,
) -> TimeSeriesDataset:
    """Filters dataset to only include data available at a specific point in time.

    Guarantees:
    - Returns only data points available before the cutoff time
    - For each timestamp, only the latest available version is kept
    - Result is ordered and indexed by timestamp

    Args:
        dataset: The dataset to filter
        available_at: Availability constraint with lag information

    Returns:
        DataFrame with timestamp as index
    """
    data = dataset.data
    cutoff = data[dataset.timestamp_column].dt.floor("D") - pd.Timedelta(available_at.lag_from_day)
    return TimeSeriesDataset(
        data=(
            data[data[dataset.available_at_column] <= cutoff]
            .sort_values(dataset.available_at_column, ascending=False)
            .drop_duplicates(subset=dataset.timestamp_column, keep="first")
            .drop(columns=[dataset.available_at_column])
            .set_index(dataset.timestamp_column)
            .sort_index()
        ),
        sample_interval=dataset.sample_interval,
    )


def filter_by_lead_time(
    dataset: VersionedTimeSeriesDataset,
    lead_time: LeadTime,
) -> TimeSeriesDataset:
    """Filters dataset to include data available with specified lead time or greater.

    Guarantees:
    - Returns only data points available at or after the cutoff (timestamp - lead_time)
    - For each timestamp, only the latest available version is kept
    - Result is ordered and indexed by timestamp

    Args:
        dataset: The dataset to filter
        lead_time: Lead time constraint

    Returns:
        DataFrame with timestamp as index
    """
    data = dataset.data
    cutoff = data[dataset.timestamp_column] - pd.Timedelta(lead_time.value)
    return TimeSeriesDataset(
        data=(
            data[data[dataset.available_at_column] <= cutoff]
            .sort_values(dataset.available_at_column, ascending=False)
            .drop_duplicates(subset=dataset.timestamp_column, keep="first")
            .drop(columns=[dataset.available_at_column])
            .set_index(dataset.timestamp_column)
            .sort_index()
        ),
        sample_interval=dataset.sample_interval,
    )


def filter_by_latest_lead_time(
    dataset: VersionedTimeSeriesDataset,
) -> TimeSeriesDataset:
    """Selects the latest available version of each timestamp in the dataset.

    Guarantees:
    - For each timestamp, only the latest available version is kept
    - Result is ordered and indexed by timestamp

    Args:
        dataset: The dataset to process

    Returns:
        DataFrame with timestamp as index
    """
    return TimeSeriesDataset(
        data=(
            dataset.data.sort_values(dataset.available_at_column, ascending=False)
            .drop_duplicates(subset=dataset.timestamp_column, keep="first")
            .drop(columns=[dataset.available_at_column])
            .set_index(dataset.timestamp_column)
            .sort_index()
        ),
        sample_interval=dataset.sample_interval,
    )
