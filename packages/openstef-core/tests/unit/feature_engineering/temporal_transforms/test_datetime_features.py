# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the DatetimeFeatures transform."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.feature_engineering.temporal_transforms.datetime_features import DatetimeFeatures


@pytest.fixture
def sample_dataset() -> TimeSeriesDataset:
    """
    Create a sample TimeSeriesDataset for testing.

    Returns:
        TimeSeriesDataset: A dataset with daily frequency and sample data.
    """
    data = pd.DataFrame({"load": [100.0, 110.0, 120.0, 130.0]}, index=pd.date_range("2025-01-01", periods=4, freq="D"))
    return TimeSeriesDataset(data, timedelta(days=1))


@pytest.fixture
def week_spanning_dataset() -> TimeSeriesDataset:
    """
    Create a dataset that spans a full week for comprehensive weekday testing.

    Returns:
        TimeSeriesDataset: A dataset with 7 days starting from a Monday.
    """
    # Start from Monday 2025-01-06
    data = pd.DataFrame(
        {"load": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0]},
        index=pd.date_range("2025-01-06", periods=7, freq="D"),
    )
    return TimeSeriesDataset(data, timedelta(days=1))


@pytest.fixture
def quarterly_dataset() -> TimeSeriesDataset:
    """
    Create a dataset that spans multiple quarters for quarter testing.

    Returns:
        TimeSeriesDataset: A dataset with monthly frequency across quarters.
    """
    data = pd.DataFrame(
        {"load": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0]},
        index=pd.date_range("2025-01-01", periods=6, freq="MS"),  # Month start
    )
    return TimeSeriesDataset(data, timedelta(days=30))


def test_weekday_computation():
    """Test weekday computation with known dates."""
    # Test with specific dates where we know the weekdays
    dates = pd.DatetimeIndex([
        "2025-01-06",  # Monday (weekday=0)
        "2025-01-07",  # Tuesday (weekday=1)
        "2025-01-11",  # Saturday (weekday=5)
        "2025-01-12",  # Sunday (weekday=6)
    ])

    weekday_result = DatetimeFeatures._is_weekday(dates)
    weekend_result = DatetimeFeatures._is_weekend_day(dates)
    sunday_result = DatetimeFeatures._is_sunday(dates)

    # Check weekday values (Monday-Friday should be 1)
    expected_weekday = np.array([1, 1, 0, 0])
    np.testing.assert_array_equal(weekday_result, expected_weekday)

    # Check weekend values (Saturday-Sunday should be 1)
    expected_weekend = np.array([0, 0, 1, 1])
    np.testing.assert_array_equal(weekend_result, expected_weekend)

    # Check Sunday values (only Sunday should be 1)
    expected_sunday = np.array([0, 0, 0, 1])
    np.testing.assert_array_equal(sunday_result, expected_sunday)


def test_month_and_quarter_computation():
    """Test month and quarter computation with known dates."""
    dates = pd.DatetimeIndex([
        "2025-01-15",  # January (month=1, quarter=1)
        "2025-04-15",  # April (month=4, quarter=2)
        "2025-07-15",  # July (month=7, quarter=3)
        "2025-10-15",  # October (month=10, quarter=4)
    ])

    month_result = DatetimeFeatures._month_of_year(dates)
    quarter_result = DatetimeFeatures._quarter_of_year(dates)

    # Check month values
    expected_months = np.array([1, 4, 7, 10])
    np.testing.assert_array_equal(month_result, expected_months)

    # Check quarter values
    expected_quarters = np.array([1, 2, 3, 4])
    np.testing.assert_array_equal(quarter_result, expected_quarters)


def test_fit_creates_all_features(sample_dataset: TimeSeriesDataset):
    """Test that fit creates all expected datetime features."""
    transform = DatetimeFeatures()
    transform.fit(sample_dataset)

    expected_columns = [
        "is_week_day",
        "is_weekend_day",
        "is_sunday",
        "month_of_year",
        "quarter_of_year",
    ]

    assert not transform._datetime_features.empty
    assert list(transform._datetime_features.columns) == expected_columns
    assert len(transform._datetime_features) == len(sample_dataset.index)
    assert transform._datetime_features.index.equals(sample_dataset.index)


def test_transform_adds_features(sample_dataset: TimeSeriesDataset):
    """Test that transform adds datetime features to the dataset."""
    transform = DatetimeFeatures()
    transform.fit(sample_dataset)
    result = transform.transform(sample_dataset)

    # Check structure
    assert isinstance(result, TimeSeriesDataset)
    assert len(result.feature_names) == len(sample_dataset.feature_names) + 5
    assert result.sample_interval == sample_dataset.sample_interval

    # Check that original features are preserved
    for feature in sample_dataset.feature_names:
        assert feature in result.feature_names
        pd.testing.assert_series_equal(result.data[feature], sample_dataset.data[feature])


def test_feature_value_ranges(sample_dataset: TimeSeriesDataset):
    """Test that all datetime features are within expected ranges."""
    transform = DatetimeFeatures()
    transform.fit(sample_dataset)
    result = transform.transform(sample_dataset)

    # Test binary features (0 or 1)
    binary_columns = ["is_week_day", "is_weekend_day", "is_sunday"]
    for col in binary_columns:
        values = result.data[col]
        assert values.isin([0, 1]).all(), f"{col} has values other than 0 or 1"
        assert not values.isna().any(), f"{col} has NaN values"

    # Test month range (1 to 12)
    month_values = result.data["month_of_year"]
    assert month_values.min() >= 1, "month_of_year has values below 1"
    assert month_values.max() <= 12, "month_of_year has values above 12"
    assert not month_values.isna().any(), "month_of_year has NaN values"

    # Test quarter range (1 to 4)
    quarter_values = result.data["quarter_of_year"]
    assert quarter_values.min() >= 1, "quarter_of_year has values below 1"
    assert quarter_values.max() <= 4, "quarter_of_year has values above 4"
    assert not quarter_values.isna().any(), "quarter_of_year has NaN values"


def test_weekday_weekend_consistency(week_spanning_dataset: TimeSeriesDataset):
    """Test that weekday and weekend features are mutually exclusive."""
    transform = DatetimeFeatures()
    transform.fit(week_spanning_dataset)
    result = transform.transform(week_spanning_dataset)

    weekday_values = result.data["is_week_day"]
    weekend_values = result.data["is_weekend_day"]

    # Weekday and weekend should be mutually exclusive (sum should always be 1)
    combined = weekday_values + weekend_values
    assert (combined == 1).all(), "Weekday and weekend features are not mutually exclusive"

    # Test specific known values for the week starting Monday 2025-01-06
    expected_weekdays = [1, 1, 1, 1, 1, 0, 0]  # Mon-Fri=1, Sat-Sun=0
    expected_weekends = [0, 0, 0, 0, 0, 1, 1]  # Mon-Fri=0, Sat-Sun=1

    np.testing.assert_array_equal(weekday_values.values, expected_weekdays)
    np.testing.assert_array_equal(weekend_values.values, expected_weekends)


def test_sunday_identification(week_spanning_dataset: TimeSeriesDataset):
    """Test that Sunday is correctly identified."""
    transform = DatetimeFeatures()
    transform.fit(week_spanning_dataset)
    result = transform.transform(week_spanning_dataset)

    sunday_values = result.data["is_sunday"]

    # Only the last day (Sunday) should be marked as 1
    expected_sundays = [0, 0, 0, 0, 0, 0, 1]  # Only Sunday=1
    np.testing.assert_array_equal(sunday_values.values, expected_sundays)


def test_quarterly_features(quarterly_dataset: TimeSeriesDataset):
    """Test quarter and month features across multiple quarters."""
    transform = DatetimeFeatures()
    transform.fit(quarterly_dataset)
    result = transform.transform(quarterly_dataset)

    month_values = result.data["month_of_year"]
    quarter_values = result.data["quarter_of_year"]

    # Expected values for months 1, 2, 3, 4, 5, 6
    expected_months = [1, 2, 3, 4, 5, 6]
    expected_quarters = [1, 1, 1, 2, 2, 2]  # Q1, Q1, Q1, Q2, Q2, Q2

    np.testing.assert_array_equal(month_values.values, expected_months)
    np.testing.assert_array_equal(quarter_values.values, expected_quarters)


def test_empty_dataset():
    """Test handling of empty dataset."""
    data = pd.DataFrame(columns=["load"]).astype(float)
    data.index = pd.DatetimeIndex([])
    dataset = TimeSeriesDataset(data, timedelta(hours=1))

    transform = DatetimeFeatures()
    transform.fit(dataset)
    result = transform.transform(dataset)

    assert len(result.data) == 0
    assert len(result.feature_names) == 6  # 1 original + 5 datetime columns

    # Check that all expected columns exist even with empty data
    expected_columns = ["load", "is_week_day", "is_weekend_day", "is_sunday", "month_of_year", "quarter_of_year"]
    assert set(result.data.columns) == set(expected_columns)


def test_leap_year_handling():
    """Test datetime features handle leap year correctly."""
    # Test February 29 in a leap year
    data = pd.DataFrame(
        {"load": [100.0, 110.0]},
        index=pd.DatetimeIndex(["2024-02-28", "2024-02-29"]),  # 2024 is a leap year
    )
    dataset = TimeSeriesDataset(data, timedelta(days=1))

    transform = DatetimeFeatures()
    transform.fit(dataset)
    result = transform.transform(dataset)

    # Both dates should be in February (month 2) and Q1
    month_values = result.data["month_of_year"]
    quarter_values = result.data["quarter_of_year"]

    assert (month_values == 2).all()
    assert (quarter_values == 1).all()
    assert len(result.data) == 2  # Should handle leap day without issues
