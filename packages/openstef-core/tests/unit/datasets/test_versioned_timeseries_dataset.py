# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for VersionedTimeSeriesDataset parquet serialization."""

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.exceptions import TimeSeriesValidationError
from openstef_core.testing import create_timeseries_dataset
from openstef_core.types import AvailableAt, LeadTime


@pytest.fixture
def dataset_part_a() -> TimeSeriesDataset:
    return create_timeseries_dataset(
        feature_a=[20.0, 22.0, 24.0, 23.0],
        available_ats=pd.to_datetime([
            "2023-01-01T10:05:00",
            "2023-01-01T11:05:00",
            "2023-01-01T12:05:00",
            "2023-01-01T13:05:00",
        ]),
        index=pd.to_datetime([
            "2023-01-01T10:00:00",
            "2023-01-01T11:00:00",
            "2023-01-01T12:00:00",
            "2023-01-01T13:00:00",
        ]),
        sample_interval=timedelta(hours=1),
    )


@pytest.fixture
def dataset_part_b() -> TimeSeriesDataset:
    return create_timeseries_dataset(
        feature_b=[100.0, 120.0, 110.0, 105.0, 125.0],
        available_ats=pd.to_datetime([
            "2023-01-01T10:10:00",
            "2023-01-01T11:10:00",
            "2023-01-01T12:10:00",
            "2023-01-01T13:10:00",
            "2023-01-01T14:15:00",
        ]),
        index=pd.to_datetime([
            "2023-01-01T10:00:00",
            "2023-01-01T11:00:00",
            "2023-01-01T12:00:00",
            "2023-01-01T13:00:00",
            "2023-01-01T14:00:00",  # Additional timestamp with delayed availability
        ]),
        sample_interval=timedelta(hours=1),
    )


@pytest.fixture
def versioned_dataset() -> VersionedTimeSeriesDataset:
    index = pd.to_datetime([
        "2023-01-01T10:00:00",
        "2023-01-01T10:00:00",
        "2023-01-01T11:00:00",
        "2023-01-01T11:00:00",
        "2023-01-01T12:00:00",
        "2023-01-01T12:00:00",
        "2023-01-01T13:00:00",
        "2023-01-01T14:00:00",
    ])

    part_a = create_timeseries_dataset(
        index=index,
        available_ats=pd.to_datetime([
            "2023-01-01T10:20:00",
            "2023-01-01T10:05:00",
            "2023-01-01T11:30:00",
            "2023-01-01T11:12:00",
            "2023-01-01T12:45:00",
            "2023-01-01T12:10:00",
            "2023-01-01T13:15:00",
            "2023-01-01T14:20:00",
        ]),
        feature_a=[100, 150, 200, 250, 275, 300, 330, 360],
    )

    part_b = create_timeseries_dataset(
        index=index,
        available_ats=pd.to_datetime([
            "2023-01-01T10:40:00",
            "2023-01-01T10:18:00",
            "2023-01-01T11:50:00",
            "2023-01-01T11:22:00",
            "2023-01-01T12:20:00",
            "2023-01-01T12:08:00",
            "2023-01-01T13:10:00",
            "2023-01-01T14:12:00",
        ]),
        feature_b=[1.0, 1.6, 2.0, 2.6, 3.2, 3.8, 4.2, 4.8],
    )

    return VersionedTimeSeriesDataset([part_a, part_b])


def test_initialization_with_multiple_parts(dataset_part_a: TimeSeriesDataset, dataset_part_b: TimeSeriesDataset):
    # Act
    dataset = VersionedTimeSeriesDataset([dataset_part_a, dataset_part_b])

    # Assert
    assert len(dataset.data_parts) == 2
    assert dataset.sample_interval == timedelta(hours=1)
    assert sorted(dataset.feature_names) == ["feature_a", "feature_b"]
    assert isinstance(dataset.index, pd.DatetimeIndex)
    assert len(dataset.index) == 5  # Union of both parts' indices


def test_initialization_with_empty_parts():
    # Act & Assert
    with pytest.raises(TimeSeriesValidationError, match="At least one data part must be provided"):
        VersionedTimeSeriesDataset([])


def test_initialization_with_mismatched_sample_intervals(
    dataset_part_a: TimeSeriesDataset, dataset_part_b: TimeSeriesDataset
):
    # Arrange
    dataset_part_b._sample_interval = timedelta(minutes=30)  # Different interval

    # Act & Assert
    with pytest.raises(TimeSeriesValidationError, match="Datasets have different sample intervals"):
        VersionedTimeSeriesDataset([dataset_part_a, dataset_part_b])


def test_initialization_with_overlapping_features(dataset_part_a: TimeSeriesDataset):
    # Act & Assert
    with pytest.raises(TimeSeriesValidationError, match="Datasets have overlapping feature names"):
        VersionedTimeSeriesDataset([dataset_part_a, dataset_part_a])


def test_filter_by_range(versioned_dataset: VersionedTimeSeriesDataset):
    """Filter dataset to a specific timestamp window."""
    # Arrange
    start = pd.Timestamp("2023-01-01T10:00:00")
    end = pd.Timestamp("2023-01-01T13:00:00")

    # Act
    filtered = versioned_dataset.filter_by_range(start=start, end=end)

    # Assert - keeps 10:00-12:00 window, retaining both versions per timestamp in lead-time order
    np.testing.assert_array_equal(filtered.data_parts[0].data["feature_a"].values, [100, 150, 200, 250, 275, 300])
    np.testing.assert_array_equal(filtered.data_parts[1].data["feature_b"].values, [1.0, 1.6, 2.0, 2.6, 3.2, 3.8])


def test_filter_by_available_before(versioned_dataset: VersionedTimeSeriesDataset):
    """Filter dataset based on absolute availability cut-off."""
    # Arrange
    cutoff = pd.Timestamp("2023-01-01T12:30:00")

    # Act
    filtered = versioned_dataset.filter_by_available_before(cutoff)

    # Assert - drops forecasts arriving after 12:30 (feature_a value 275 and later)
    np.testing.assert_array_equal(filtered.data_parts[0].data["feature_a"].values, [100, 150, 200, 250, 300])
    np.testing.assert_array_equal(filtered.data_parts[1].data["feature_b"].values, [1.0, 1.6, 2.0, 2.6, 3.2, 3.8])


def test_filter_by_available_at(versioned_dataset: VersionedTimeSeriesDataset):
    """Filter dataset using relative availability definition."""
    # Arrange
    available_at = AvailableAt(timedelta(hours=-13))

    # Act
    filtered = versioned_dataset.filter_by_available_at(available_at)

    # Assert - keeps forecasts available before 13:00 while retaining both versions per timestamp
    np.testing.assert_array_equal(filtered.data_parts[0].data["feature_a"].values, [100, 150, 200, 250, 275, 300])
    np.testing.assert_array_equal(filtered.data_parts[1].data["feature_b"].values, [1.0, 1.6, 2.0, 2.6, 3.2, 3.8])


def test_filter_by_lead_time(versioned_dataset: VersionedTimeSeriesDataset):
    """Filter dataset based on latest permissible lead time."""
    # Act
    filtered = versioned_dataset.filter_by_lead_time(lead_time=LeadTime(timedelta(minutes=-10)))

    # Assert - keeps only data with lead_time >= -10 minutes (available at or before 10 minutes after timestamp)
    # Part A: row 1 (10:00/10:05, lead=-5min) and row 5 (12:00/12:10, lead=-10min) → values 150, 300
    # Part B: row 5 (12:00/12:08, lead=-8min) and row 6 (13:00/13:10, lead=-10min) → values 3.8, 4.2
    assert filtered.data_parts[0].data["feature_a"].tolist() == [150, 300]
    assert filtered.data_parts[1].data["feature_b"].tolist() == [3.8, 4.2]


def test_select_version(versioned_dataset: VersionedTimeSeriesDataset):
    """Select latest available version across all data parts."""
    # Act
    selected = versioned_dataset.select_version()

    # Assert - older forecasts (100, 200, 275 for feature_a) are dropped in favor of fresher updates
    pd.testing.assert_index_equal(selected.data.index, versioned_dataset.index)
    assert sorted(selected.feature_names) == ["feature_a", "feature_b"]
    np.testing.assert_array_equal(selected.data["feature_a"].values, [100, 200, 275, 330, 360])
    np.testing.assert_array_equal(selected.data["feature_b"].values, [1.0, 2.0, 3.2, 4.2, 4.8])
    assert not selected.is_versioned


def test_to_horizons_returns_expected_labels_when_splitting_versions():
    """Verify that to_horizons slices forecasts at requested lead times."""
    # Arrange - dataset with clearly differentiated horizons encoded in string labels
    timestamps = pd.date_range("2025-01-01T10:00:00", periods=4, freq="1h")
    index = pd.DatetimeIndex(timestamps.tolist() * 3, name="timestamp")
    available_ats = pd.to_datetime(
        [ts - pd.Timedelta(hours=1) for ts in timestamps]
        + [ts - pd.Timedelta(hours=2) for ts in timestamps]
        + [ts - pd.Timedelta(hours=24) for ts in timestamps]
    )
    feature_values = [
        "short_h1",
        "short_h2",
        "short_h3",
        "short_h4",
        "medium_h1",
        "medium_h2",
        "medium_h3",
        "medium_h4",
        "long_h1",
        "long_h2",
        "long_h3",
        "long_h4",
    ]
    sample_interval = timedelta(hours=1)
    dataset = VersionedTimeSeriesDataset([
        create_timeseries_dataset(
            index=index,
            available_ats=available_ats,
            feature=feature_values,
            sample_interval=sample_interval,
        )
    ])

    horizons = [
        LeadTime.from_string("PT1H"),
        LeadTime.from_string("PT2H"),
        LeadTime.from_string("PT24H"),
    ]

    # Act
    horizon_dataset = dataset.to_horizons(horizons)

    # Assert - dataset stays versioned with all timestamps represented per horizon
    assert isinstance(horizon_dataset, TimeSeriesDataset)
    assert horizon_dataset.sample_interval == sample_interval
    assert horizon_dataset.is_versioned
    assert len(horizon_dataset.data) == len(horizons) * 4
    assert set(horizon_dataset.data["horizon"].unique()) == {h.value for h in horizons}

    expected_labels = {
        horizons[0]: "short",
        horizons[1]: "medium",
        horizons[2]: "long",
    }
    for horizon, label in expected_labels.items():
        # Assert - only matching horizon rows remain after filtering and version selection
        horizon_mask = horizon_dataset.data["horizon"] == horizon.value
        values = horizon_dataset.data.loc[horizon_mask, "feature"].tolist()
        assert values == [f"{label}_h{i}" for i in range(1, 5)]


def test_filter_by_available_before_realistic_energy_forecasting_scenario():
    """Integration-style check using realistic forecast and actual availability patterns."""
    # Arrange - simulate hourly forecasts ready 30 minutes ahead and actuals with 1-hour delay
    timestamps = pd.date_range("2023-01-01T10:00:00", periods=6, freq="1h")
    forecast_part = create_timeseries_dataset(
        index=timestamps,
        available_ats=timestamps - pd.Timedelta(minutes=30),
        forecast=[100, 95, 105, 110, 98, 102],
    )
    actual_part = create_timeseries_dataset(
        index=timestamps,
        available_ats=timestamps + pd.Timedelta(hours=1),
        actual=[102, 97, 108, 107, 96, 104],
    )
    dataset = VersionedTimeSeriesDataset([forecast_part, actual_part])

    # Act - select what is known by noon
    available_before_noon = pd.Timestamp("2023-01-01T12:00:00")
    available_at_noon = dataset.filter_by_available_before(available_before_noon).select_version()

    # Assert - full horizon preserved while later forecasts remain NaN
    assert len(available_at_noon.data) == 6
    assert available_at_noon.data["forecast"].iloc[:3].notna().all()
    assert available_at_noon.data["forecast"].iloc[3:].isna().all()

    # Assert - first two actuals are available (delivered by 11:00 and 12:00), later ones still pending
    assert available_at_noon.data["actual"].iloc[:2].notna().all()
    assert available_at_noon.data["actual"].iloc[2:].isna().all()


def test_parquet_roundtrip(tmp_path: Path):
    """Test that parquet roundtrip preserves data, structure, and version selection."""
    # Arrange - weather and load data with different availability times
    weather_data = pd.DataFrame(
        {
            "temperature": [20.0, 21.5, 19.0],
            "wind_speed": [5.0, 6.2, 4.8],
            "available_at": pd.date_range("2025-01-01T10:05", periods=3, freq="1h"),
        },
        index=pd.date_range("2025-01-01T10:00", periods=3, freq="1h"),
    )
    load_data = pd.DataFrame(
        {
            "load": [100.0, 110.0, 105.0],
            "available_at": pd.date_range("2025-01-02T10:05", periods=3, freq="1h"),
        },
        index=pd.date_range("2025-01-01T10:00", periods=3, freq="1h"),
    )

    original = VersionedTimeSeriesDataset(
        data_parts=[
            TimeSeriesDataset(weather_data, sample_interval=timedelta(hours=1)),
            TimeSeriesDataset(load_data, sample_interval=timedelta(hours=1)),
        ]
    )

    # Act
    parquet_path = tmp_path / "versioned_dataset.parquet"
    original.to_parquet(parquet_path)
    loaded = VersionedTimeSeriesDataset.read_parquet(parquet_path)

    # Assert - metadata preserved
    assert loaded.sample_interval == original.sample_interval
    assert loaded.is_versioned == original.is_versioned
    assert loaded.feature_names == original.feature_names
    assert len(loaded.data_parts) == len(original.data_parts)

    # Assert - each part's data preserved
    for orig_part, loaded_part in zip(original.data_parts, loaded.data_parts, strict=True):
        assert loaded_part.feature_names == orig_part.feature_names
        assert loaded_part.sample_interval == orig_part.sample_interval
        pd.testing.assert_frame_equal(
            loaded_part.data,
            orig_part.data,
            check_freq=False,
        )
