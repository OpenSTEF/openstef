# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for MultiHorizon dataset."""

from datetime import timedelta
from pathlib import Path

import pandas as pd

from openstef_core.datasets import MultiHorizon, TimeSeriesDataset
from openstef_core.types import LeadTime


def test_parquet_roundtrip(tmp_path: Path):
    """Test saving and loading MultiHorizon with different columns per horizon."""
    # Arrange - realistic scenario: shorter horizons have more features
    data_1h = pd.DataFrame(
        {"load": [100, 110, 120], "temperature": [20.0, 21.5, 19.0], "wind_speed": [5.0, 6.2, 4.8]},
        index=pd.date_range("2025-01-01", periods=3, freq="1h"),
    )
    data_6h = pd.DataFrame(
        {"load": [95, 105, 115], "temperature": [19.5, 20.0, 18.5]},
        index=pd.date_range("2025-01-01", periods=3, freq="1h"),
    )
    data_24h = pd.DataFrame({"load": [90, 100, 110]}, index=pd.date_range("2025-01-01", periods=3, freq="1h"))

    original = MultiHorizon({
        LeadTime.from_string("PT1H"): TimeSeriesDataset(data_1h, sample_interval=timedelta(hours=1)),
        LeadTime.from_string("PT6H"): TimeSeriesDataset(data_6h, sample_interval=timedelta(hours=1)),
        LeadTime.from_string("P1D"): TimeSeriesDataset(data_24h, sample_interval=timedelta(hours=1)),
    })

    # Act
    parquet_path = tmp_path / "multi_horizon.parquet"
    original.to_parquet(parquet_path)
    loaded = MultiHorizon.read_parquet(parquet_path)

    # Assert - structure
    assert loaded.sample_interval == timedelta(hours=1)
    assert sorted(loaded.feature_names) == ["load", "temperature", "wind_speed"]
    assert set(loaded.keys()) == set(original.keys())

    # Assert - data and columns per horizon
    for horizon in original:
        assert sorted(loaded[horizon].feature_names) == sorted(original[horizon].feature_names)
        pd.testing.assert_frame_equal(
            loaded[horizon].data.sort_index(),
            original[horizon].data.sort_index(),
            check_freq=False,
        )
