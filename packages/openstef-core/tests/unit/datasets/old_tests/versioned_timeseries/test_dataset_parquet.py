# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for VersionedTimeSeriesDataset parquet serialization."""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from openstef_core.datasets.versioned_timeseries_dataset import VersionedTimeSeriesDataset
from openstef_core.datasets.versioned_timeseries_dataset_part import VersionedTimeSeriesPart


def test_parquet_roundtrip(tmp_path: Path):
    """Test saving and loading VersionedTimeSeriesDataset with multiple parts."""
    # Arrange - realistic scenario: weather and load data with different availability
    weather_data = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01T10:00", periods=3, freq="1h"),
        "available_at": pd.date_range("2025-01-01T10:05", periods=3, freq="1h"),
        "temperature": [20.0, 21.5, 19.0],
        "wind_speed": [5.0, 6.2, 4.8],
    })
    load_data = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01T10:00", periods=3, freq="1h"),
        "available_at": pd.date_range("2025-01-01T11:00", periods=3, freq="1h"),
        "load": [100.0, 110.0, 105.0],
    })

    original = VersionedTimeSeriesDataset([
        VersionedTimeSeriesPart(weather_data, sample_interval=timedelta(hours=1)),
        VersionedTimeSeriesPart(load_data, sample_interval=timedelta(hours=1)),
    ])

    # Act
    parquet_path = tmp_path / "versioned_dataset.parquet"
    original.to_parquet(parquet_path)
    loaded = VersionedTimeSeriesDataset.read_parquet(parquet_path)

    # Assert - structure
    assert loaded.sample_interval == timedelta(hours=1)
    assert sorted(loaded.feature_names) == ["load", "temperature", "wind_speed"]
    assert len(loaded.data_parts) == len(original.data_parts)

    # Assert - data parts preserved
    for orig_part, load_part in zip(original.data_parts, loaded.data_parts, strict=True):
        assert sorted(load_part.feature_names) == sorted(orig_part.feature_names)
        assert load_part.timestamp_column == orig_part.timestamp_column
        assert load_part.available_at_column == orig_part.available_at_column
        pd.testing.assert_frame_equal(
            load_part.data.sort_values(by=["timestamp", "available_at"]),
            orig_part.data.sort_values(by=["timestamp", "available_at"]),
            check_dtype=False,
        )

    # Assert - version selection works correctly
    cutoff = datetime.fromisoformat("2025-01-01T10:30:00")
    orig_snapshot = original.select_version(available_before=cutoff)
    loaded_snapshot = loaded.select_version(available_before=cutoff)
    pd.testing.assert_frame_equal(
        loaded_snapshot.data.sort_index(),
        orig_snapshot.data.sort_index(),
        check_freq=False,
    )
