# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets.versioned_timeseries_accessors import (
    ConcatenatedVersionedTimeSeries,
    ConcatMode,
    RestrictedHorizonVersionedTimeSeries,
    VersionedTimeSeriesAccessors,
)
from openstef_core.datasets.versioned_timeseries_dataset import VersionedTimeseriesDataset
from openstef_core.exceptions import TimeSeriesValidationError

if TYPE_CHECKING:
    from openstef_core.datasets.mixins import VersionedTimeSeriesMixin


@pytest.fixture
def simple_dataset() -> VersionedTimeseriesDataset:
    return VersionedTimeseriesDataset(
        data=pd.DataFrame({
            "timestamp": [
                datetime.fromisoformat("2024-01-01T10:00:00"),
                datetime.fromisoformat("2024-01-01T11:00:00"),
                datetime.fromisoformat("2024-01-01T12:00:00"),
            ],
            "available_at": [
                datetime.fromisoformat("2024-01-01T10:05:00"),
                datetime.fromisoformat("2024-01-01T11:05:00"),
                datetime.fromisoformat("2024-01-01T12:05:00"),
            ],
            "value": [10.0, 20.0, 30.0],
        }),
        sample_interval=timedelta(hours=1),
    )


@pytest.fixture
def partial_dataset() -> VersionedTimeseriesDataset:
    # Create a dataset with some missing timestamps
    data = pd.DataFrame({
        "timestamp": [
            datetime.fromisoformat("2024-01-01T10:00:00"),
            # Gap at 11:00
            datetime.fromisoformat("2024-01-01T12:00:00"),
        ],
        "available_at": [
            datetime.fromisoformat("2024-01-01T10:05:00"),
            datetime.fromisoformat("2024-01-01T12:05:00"),
        ],
        "third_value": [1000.0, 3000.0],
    })

    return VersionedTimeseriesDataset(data=data, sample_interval=timedelta(hours=1))


def test_restricted_horizon_basic(simple_dataset: VersionedTimeseriesDataset):
    # Arrange
    horizon = datetime.fromisoformat("2024-01-01T12:00:00")
    start = datetime.fromisoformat("2024-01-01T10:00:00")
    end = datetime.fromisoformat("2024-01-01T13:00:00")

    expected = pd.DataFrame(
        {"value": [10.0, 20.0, np.nan]},
        index=pd.DatetimeIndex(["2024-01-01T10:00:00", "2024-01-01T11:00:00", "2024-01-01T12:00:00"]),
    )

    # Act
    transform = RestrictedHorizonVersionedTimeSeries(simple_dataset, horizon)
    window = transform.get_window(start, end)

    # Assert
    assert transform.feature_names == simple_dataset.feature_names
    assert transform.sample_interval == simple_dataset.sample_interval
    pd.testing.assert_frame_equal(window, expected, check_freq=False)


def test_restricted_horizon_validation(simple_dataset: VersionedTimeseriesDataset):
    # Arrange
    horizon = datetime.fromisoformat("2024-01-01T12:00:00")
    start = datetime.fromisoformat("2024-01-01T10:00:00")
    end = datetime.fromisoformat("2024-01-01T13:00:00")
    invalid_available = datetime.fromisoformat("2024-01-01T13:00:00")  # After horizon

    # Act & Assert
    transform = RestrictedHorizonVersionedTimeSeries(simple_dataset, horizon)
    with pytest.raises(ValueError, match=r"Available before .* is greater than the horizon"):
        transform.get_window(start, end, invalid_available)


# ConcatenatedVersionedTimeSeries Tests
@pytest.mark.parametrize(
    ("mode", "expected_length"),
    [
        pytest.param("left", 3, id="left_join"),
        pytest.param("inner", 3, id="inner_join"),
        pytest.param("outer", 3, id="outer_join"),
    ],
)
def test_concatenated_versioned_timeseries_basic(
    simple_dataset: VersionedTimeseriesDataset, mode: ConcatMode, expected_length: int
):
    # Arrange
    start = datetime.fromisoformat("2024-01-01T10:00:00")
    end = datetime.fromisoformat("2024-01-01T13:00:00")
    second_dataset = VersionedTimeseriesDataset(
        data=simple_dataset.data.rename(columns={"value": "other_value"}),
        sample_interval=simple_dataset.sample_interval,
    )

    # Act
    transform = ConcatenatedVersionedTimeSeries([simple_dataset, second_dataset], mode=mode)
    window = transform.get_window(start, end)

    # Assert
    assert len(window) == expected_length
    assert set(transform.feature_names) == {"value", "other_value"}
    assert transform.sample_interval == simple_dataset.sample_interval
    assert all(col in window.columns for col in ["value", "other_value"])


def test_concatenated_versioned_timeseries_with_partial(
    simple_dataset: VersionedTimeseriesDataset, partial_dataset: VersionedTimeseriesDataset
):
    # Arrange
    start = datetime.fromisoformat("2024-01-01T10:00:00")
    end = datetime.fromisoformat("2024-01-01T13:00:00")

    # Act
    transform = ConcatenatedVersionedTimeSeries([simple_dataset, partial_dataset], mode="outer")
    window = transform.get_window(start, end)

    # Assert
    assert len(window) == 3  # Should include all timestamps from both datasets
    assert set(transform.feature_names) == {"value", "third_value"}
    assert pd.isna(window.loc[pd.Timestamp.fromisoformat("2024-01-01T11:00:00"), "third_value"])
    assert window.loc[pd.Timestamp.fromisoformat("2024-01-01T11:00:00"), "value"] == 20.0


def test_concatenated_versioned_timeseries_validation_error():
    # Arrange
    invalid_datasets: list[VersionedTimeSeriesMixin] = []

    # Act & Assert
    with pytest.raises(ValueError, match="At least two datasets are required"):
        ConcatenatedVersionedTimeSeries(invalid_datasets, mode="left")


def test_concatenated_versioned_timeseries_overlapping_features(simple_dataset: VersionedTimeseriesDataset):
    # Arrange - both datasets have "value" column

    # Act & Assert
    with pytest.raises(TimeSeriesValidationError, match=r"Datasets have overlapping feature names"):
        ConcatenatedVersionedTimeSeries([simple_dataset, simple_dataset], mode="left")


# VersionedTimeSeriesAccessors Factory Tests
def test_versioned_timeseries_accessors_factory(simple_dataset: VersionedTimeseriesDataset):
    # Arrange
    horizon = datetime.fromisoformat("2024-01-01T12:00:00")
    second_dataset = VersionedTimeseriesDataset(
        data=simple_dataset.data.rename(columns={"value": "other_value"}),
        sample_interval=simple_dataset.sample_interval,
    )

    # Act
    horizon_transform = VersionedTimeSeriesAccessors.restrict_horizon(simple_dataset, horizon)
    concat_transform = VersionedTimeSeriesAccessors.concat_featurewise([simple_dataset, second_dataset], mode="left")

    # Assert
    assert isinstance(horizon_transform, RestrictedHorizonVersionedTimeSeries)
    assert isinstance(concat_transform, ConcatenatedVersionedTimeSeries)
    assert horizon_transform.horizon == horizon
