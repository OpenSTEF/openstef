# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the DaylightFeaturesTransform."""

from datetime import timedelta

import pandas as pd
import pytest
from pydantic_extra_types.coordinate import Coordinate, Latitude, Longitude

from openstef_core.datasets import TimeSeriesDataset
from openstef_models.feature_engineering.weather_transforms import DaylightFeaturesTransform

pvlib = pytest.importorskip("pvlib")


def test_daylight_features_basic():
    """Test basic daylight features generation and data preservation."""
    # Create timezone-aware test data (required for daylight calculations)
    data = pd.DataFrame(
        {"load": [100.0, 200.0], "temperature": [15.5, 16.2]},
        index=pd.date_range("2025-06-01 12:00:00", periods=2, freq="h", tz="Europe/Amsterdam"),
    )
    input_data = TimeSeriesDataset(data, timedelta(hours=1))

    # Netherlands coordinates
    transform = DaylightFeaturesTransform(coordinate=Coordinate(latitude=Latitude(52.0), longitude=Longitude(5.0)))
    result = transform.transform(input_data)

    # Check that daylight column is added
    expected_cols = {"load", "temperature", "daylight_continuous"}
    assert set(result.data.columns) == expected_cols

    # Original columns should be preserved with exact values
    assert result.data["load"].iloc[0] == 100.0
    assert result.data["temperature"].iloc[1] == 16.2

    # Daylight values should be non-negative (physical constraint)
    assert all(result.data["daylight_continuous"] >= 0)


def test_daylight_features_daytime_values():
    """Test that daylight features have realistic values during daytime."""
    # June midday in Amsterdam should have high solar radiation
    data = pd.DataFrame(
        {"value": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2025-06-21 12:00:00", "2025-06-21 13:00:00"], tz="Europe/Amsterdam"),
    )
    input_data = TimeSeriesDataset(data, timedelta(hours=1))

    transform = DaylightFeaturesTransform(coordinate=Coordinate(latitude=Latitude(52.0), longitude=Longitude(5.0)))
    result = transform.transform(input_data)

    # Summer midday should have positive solar radiation
    daylight_values = result.data["daylight_continuous"]
    assert all(daylight_values > 0), "Expected positive daylight during summer midday"
    assert daylight_values.max() > 500, "Expected high solar radiation in June"


def test_daylight_features_nighttime_values():
    """Test that daylight features are zero during nighttime."""
    # Night time in Amsterdam should have zero solar radiation
    data = pd.DataFrame(
        {"value": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2025-06-21 02:00:00", "2025-06-21 03:00:00"], tz="Europe/Amsterdam"),
    )
    input_data = TimeSeriesDataset(data, timedelta(hours=1))

    transform = DaylightFeaturesTransform(coordinate=Coordinate(latitude=Latitude(52.0), longitude=Longitude(5.0)))
    result = transform.transform(input_data)

    # Night time should have zero or very low solar radiation
    daylight_values = result.data["daylight_continuous"]
    assert all(daylight_values < 50), "Expected low/zero daylight during night"


def test_daylight_features_different_coordinates():
    """Test daylight calculation with different geographical coordinates."""
    # Same time, different locations should give different results
    time_index = pd.DatetimeIndex(["2025-06-21 12:00:00"], tz="UTC")

    # Amsterdam vs Cape Town (southern hemisphere, winter)
    data = pd.DataFrame({"value": [1.0]}, index=time_index)
    input_data = TimeSeriesDataset(data, timedelta(hours=1))

    # Amsterdam (northern hemisphere, summer)
    amsterdam_transform = DaylightFeaturesTransform(
        coordinate=Coordinate(latitude=Latitude(52.0), longitude=Longitude(5.0))
    )
    amsterdam_result = amsterdam_transform.transform(input_data)

    # Cape Town (southern hemisphere, winter)
    capetown_transform = DaylightFeaturesTransform(
        coordinate=Coordinate(latitude=Latitude(-33.9), longitude=Longitude(18.4))
    )
    capetown_result = capetown_transform.transform(input_data)

    amsterdam_daylight = amsterdam_result.data["daylight_continuous"].iloc[0]
    capetown_daylight = capetown_result.data["daylight_continuous"].iloc[0]

    # June 21st: summer in north, winter in south - should be significantly different
    assert amsterdam_daylight != capetown_daylight, "Different coordinates should give different daylight values"
