# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the DatetimeFeaturesAdder."""

from datetime import timedelta

import pandas as pd

from openstef_core.datasets import TimeSeriesDataset
from openstef_models.transforms.time_domain import DatetimeFeaturesAdder


def test_datetime_features_basic():
    """Test basic datetime features generation and data preservation."""
    # Create test data with multiple columns and known dates
    data = pd.DataFrame(
        {"load": [100.0, 200.0], "temperature": [15.5, 16.2]},
        index=pd.DatetimeIndex(["2025-01-06", "2025-01-07"]),  # Mon, Tue
    )
    input_data = TimeSeriesDataset(data, timedelta(days=1))

    transform = DatetimeFeaturesAdder()
    result = transform.transform(input_data)

    # Check that all expected columns are added
    expected_cols = {
        "load",
        "temperature",
        "is_week_day",
        "is_weekend_day",
        "is_sunday",
        "month_of_year",
        "quarter_of_year",
    }
    assert set(result.data.columns) == expected_cols

    # Original columns should be preserved with exact values
    assert result.data["load"].iloc[0] == 100.0
    assert result.data["temperature"].iloc[1] == 16.2


def test_datetime_features_weekday_weekend():
    """Test weekday/weekend classification with known dates."""
    # Monday and Saturday to test weekday vs weekend
    data = pd.DataFrame(
        {"value": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2025-01-06", "2025-01-11"]),  # Monday, Saturday
    )
    input_data = TimeSeriesDataset(data, timedelta(days=1))

    transform = DatetimeFeaturesAdder()
    result = transform.transform(input_data)

    # Monday should be weekday, Saturday should be weekend
    assert result.data.iloc[0]["is_week_day"] == 1
    assert result.data.iloc[0]["is_weekend_day"] == 0
    assert result.data.iloc[1]["is_week_day"] == 0
    assert result.data.iloc[1]["is_weekend_day"] == 1


def test_datetime_features_sunday_detection():
    """Test Sunday detection with known date."""
    data = pd.DataFrame(
        {"value": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2025-01-11", "2025-01-12"]),  # Saturday, Sunday
    )
    input_data = TimeSeriesDataset(data, timedelta(days=1))

    transform = DatetimeFeaturesAdder()
    result = transform.transform(input_data)

    # Only Sunday should be marked as Sunday
    assert result.data.iloc[0]["is_sunday"] == 0  # Saturday
    assert result.data.iloc[1]["is_sunday"] == 1  # Sunday


def test_datetime_features_month_quarter():
    """Test month and quarter features with known dates."""
    data = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(["2025-01-15", "2025-04-15", "2025-10-15"]),  # Jan Q1, Apr Q2, Oct Q4
    )
    input_data = TimeSeriesDataset(data, timedelta(days=1))

    transform = DatetimeFeaturesAdder()
    result = transform.transform(input_data)

    # Check specific month and quarter values
    assert result.data.iloc[0]["month_of_year"] == 1
    assert result.data.iloc[0]["quarter_of_year"] == 1
    assert result.data.iloc[1]["month_of_year"] == 4
    assert result.data.iloc[1]["quarter_of_year"] == 2
    assert result.data.iloc[2]["month_of_year"] == 10
    assert result.data.iloc[2]["quarter_of_year"] == 4


def test_datetime_features_onehot_encoding():
    """Test one-hot encoding functionality."""
    data = pd.DataFrame(
        {"value": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2025-01-15", "2025-04-15"]),  # Jan, Apr
    )
    input_data = TimeSeriesDataset(data, timedelta(days=1))

    transform = DatetimeFeaturesAdder(onehot_encode=True)
    result = transform.transform(input_data)

    # Should have one-hot encoded month and quarter columns
    assert "month_1" in result.data.columns
    assert "month_4" in result.data.columns
    assert "quarter_1" in result.data.columns
    assert "quarter_2" in result.data.columns

    # Should NOT have regular month/quarter columns
    assert "month_of_year" not in result.data.columns
    assert "quarter_of_year" not in result.data.columns
