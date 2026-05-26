# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the CyclicFeaturesAdder."""

from datetime import timedelta

import pandas as pd

from openstef_core.datasets import TimeSeriesDataset
from openstef_models.transforms.time_domain import CyclicFeaturesAdder


def test_cyclic_features_basic():
    """Test basic cyclic features generation."""
    # Create simple test data
    data = pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=pd.date_range("2025-01-01", periods=3, freq="h"))
    input_data = TimeSeriesDataset(data, timedelta(hours=1))

    transform = CyclicFeaturesAdder()
    result = transform.transform(input_data)

    # Check that cyclic columns are added
    assert "season_sine" in result.data.columns
    assert "season_cosine" in result.data.columns
    assert "month_sine" in result.data.columns
    assert "month_cosine" in result.data.columns
    assert "day_of_week_sine" in result.data.columns
    assert "day_of_week_cosine" in result.data.columns
    assert "time_of_day_sine" in result.data.columns
    assert "time_of_day_cosine" in result.data.columns


def test_cyclic_features_midnight_values():
    """Test cyclic values at midnight (known reference point)."""
    # Single timestamp at midnight
    data = pd.DataFrame({"value": [1.0]}, index=pd.DatetimeIndex(["2025-01-01 00:00:00"]))
    input_data = TimeSeriesDataset(data, timedelta(hours=1))

    transform = CyclicFeaturesAdder()
    result = transform.transform(input_data)

    # At midnight, time_of_day should be 0 (cosine=1, sine=0)
    time_sine = result.data.iloc[0]["time_of_day_sine"]
    time_cosine = result.data.iloc[0]["time_of_day_cosine"]

    # Values should be close to expected
    assert time_sine < 1e-10
    assert abs(time_cosine - 1.0) < 1e-10


def test_cyclic_features_custom_features():
    """Test with custom feature selection."""
    data = pd.DataFrame({"value": [1.0, 2.0]}, index=pd.date_range("2025-01-01", periods=2, freq="D"))
    input_data = TimeSeriesDataset(data, timedelta(days=1))

    # Only season and month features
    transform = CyclicFeaturesAdder(included_features=["season", "month"])
    result = transform.transform(input_data)

    # Check only requested features are present
    expected_cols = {"value", "season_sine", "season_cosine", "month_sine", "month_cosine"}
    assert set(result.data.columns) == expected_cols


def test_cyclic_features_preserves_original_data():
    """Test that original data is preserved."""
    data = pd.DataFrame(
        {"load": [100.0, 200.0], "temperature": [15.5, 16.2]}, index=pd.date_range("2025-01-01", periods=2, freq="h")
    )
    input_data = TimeSeriesDataset(data, timedelta(hours=1))

    transform = CyclicFeaturesAdder()
    result = transform.transform(input_data)

    # Original columns should be preserved
    assert "load" in result.data.columns
    assert "temperature" in result.data.columns
    assert result.data["load"].iloc[0] == 100.0
    assert result.data["temperature"].iloc[1] == 16.2


def test_cyclic_features_sine_cosine_relationship():
    """Test that sine and cosine values satisfy fundamental relationship."""
    data = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]}, index=pd.date_range("2025-01-01", periods=4, freq="6h"))
    input_data = TimeSeriesDataset(data, timedelta(hours=6))

    transform = CyclicFeaturesAdder()
    result = transform.transform(input_data)

    # Check sin²(x) + cos²(x) = 1 for all features
    for feature in ["season", "month", "day_of_week", "time_of_day"]:
        sine_col = f"{feature}_sine"
        cosine_col = f"{feature}_cosine"

        # Calculate sin² + cos² for each row
        sin_squared = result.data[sine_col] ** 2
        cos_squared = result.data[cosine_col] ** 2
        sum_squares = sin_squared + cos_squared

        # Should be very close to 1.0 for all rows
        assert all(abs(val - 1.0) < 1e-10 for val in sum_squares)


def test_cyclic_features_no_features():
    """Test with empty features list."""
    data = pd.DataFrame({"value": [1.0, 2.0]}, index=pd.date_range("2025-01-01", periods=2, freq="h"))
    input_data = TimeSeriesDataset(data, timedelta(hours=1))

    # No features
    transform = CyclicFeaturesAdder(included_features=[])
    result = transform.transform(input_data)

    # Only original data should be present
    assert list(result.data.columns) == ["value"]
    assert len(result.data) == 2
