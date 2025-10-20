# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Shared fixtures for versioned timeseries dataset tests."""

from datetime import timedelta

import pandas as pd
import pytest

from openstef_core.datasets.versioned_timeseries_dataset import VersionedTimeSeriesDataset
from openstef_core.datasets.versioned_timeseries_dataset_part import VersionedTimeSeriesPart


@pytest.fixture
def dataset_a_data() -> pd.DataFrame:
    """Sample dataset A data for testing."""
    return pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2023-01-01T10:00:00",
            "2023-01-01T11:00:00",
            "2023-01-01T12:00:00",
            "2023-01-01T13:00:00",
        ]),
        "available_at": pd.to_datetime([
            "2023-01-01T10:05:00",
            "2023-01-01T11:05:00",
            "2023-01-01T12:05:00",
            "2023-01-01T13:05:00",
        ]),
        "feature_a": [20.0, 22.0, 24.0, 23.0],
    })


@pytest.fixture
def dataset_b_data() -> pd.DataFrame:
    """Sample dataset B data for testing."""
    return pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2023-01-01T10:00:00",
            "2023-01-01T11:00:00",
            "2023-01-01T12:00:00",
            "2023-01-01T13:00:00",
            "2023-01-01T14:00:00",  # Additional timestamp with delayed availability
        ]),
        "available_at": pd.to_datetime([
            "2023-01-01T10:10:00",
            "2023-01-01T11:10:00",
            "2023-01-01T12:10:00",
            "2023-01-01T13:10:00",
            "2023-01-01T14:15:00",
        ]),
        "feature_b": [100.0, 120.0, 110.0, 105.0, 125.0],
    })


@pytest.fixture
def dataset_part_a(dataset_a_data: pd.DataFrame) -> VersionedTimeSeriesPart:
    """Create dataset A part."""
    return VersionedTimeSeriesPart(
        data=dataset_a_data,
        sample_interval=timedelta(hours=1),
    )


@pytest.fixture
def dataset_part_b(dataset_b_data: pd.DataFrame) -> VersionedTimeSeriesPart:
    """Create dataset B part."""
    return VersionedTimeSeriesPart(
        data=dataset_b_data,
        sample_interval=timedelta(hours=1),
    )


@pytest.fixture
def combined_dataset(
    dataset_part_a: VersionedTimeSeriesPart, dataset_part_b: VersionedTimeSeriesPart
) -> VersionedTimeSeriesDataset:
    """Create a combined dataset with dataset A and B."""
    return VersionedTimeSeriesDataset([dataset_part_a, dataset_part_b])


# Additional fixtures for dataset part tests (shared between part and dataset tests)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Sample data with duplicates for testing dataset parts."""
    return pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2023-01-01T10:00:00",
            "2023-01-01T11:00:00",
            "2023-01-01T12:00:00",
            "2023-01-01T10:00:00",  # Duplicate
        ]),
        "available_at": pd.to_datetime([
            "2023-01-01T10:30:00",
            "2023-01-01T11:30:00",
            "2023-01-01T12:30:00",
            "2023-01-01T10:45:00",  # Later availability for duplicate
        ]),
        "value": [1.0, 2.0, 3.0, 1.5],
    })
