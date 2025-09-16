# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the HolidayFeaturesTransform."""

from datetime import date, timedelta

import pandas as pd
import pytest
from pydantic_extra_types.country import CountryAlpha2

from openstef_core.datasets import TimeSeriesDataset
from openstef_models.transforms.time_domain.holiday_features_transform import HolidayFeaturesTransform


@pytest.fixture
def sample_dataset() -> TimeSeriesDataset:
    """Create a sample TimeSeriesDataset for testing.

    Returns:
        TimeSeriesDataset: A dataset with daily frequency spanning Christmas period.
    """
    data = pd.DataFrame(
        {"load": [100.0, 110.0, 120.0, 130.0, 140.0]}, index=pd.date_range("2025-12-24", periods=5, freq="D")
    )
    return TimeSeriesDataset(data, timedelta(days=1))


def test_holiday_features_initialization():
    """Test HolidayFeaturesTransform can be initialized properly."""
    # Arrange & Act
    transform = HolidayFeaturesTransform(country_code=CountryAlpha2("NL"))

    # Assert
    assert transform.country_code == "NL"


@pytest.mark.parametrize(
    ("input_name", "expected_output"),
    [
        pytest.param("Christmas Day", "christmas_day", id="christmas_day"),
        pytest.param("New Year's Day", "new_year_s_day", id="new_years_day"),
        pytest.param("St. Patrick's Day", "st_patrick_s_day", id="st_patrick_s_day"),
        pytest.param("INDEPENDENCE DAY", "independence_day", id="independence_day"),
        pytest.param("Labor Day (May Day)", "labor_day_may_day", id="labor_day_may_day"),
    ],
)
def test_sanitize_holiday_name(input_name: str, expected_output: str):
    """Test holiday name sanitization for feature names."""
    # Act & Assert
    assert HolidayFeaturesTransform._sanitize_holiday_name(input_name) == expected_output


def test_get_holidays_dataframe_returns_cleaned_data(sample_dataset: TimeSeriesDataset):
    """Test getting holidays DataFrame with cleaned data using real holiday data."""
    # Arrange
    transform = HolidayFeaturesTransform(country_code=CountryAlpha2("NL"))

    # Act
    result = transform._get_holidays_dataframe(sample_dataset)

    # Assert
    # Check DataFrame structure
    assert set(result.columns) == {"date", "holiday_name", "sanitized_name"}
    assert len(result) >= 2  # Should have at least Christmas Day and Second Day of Christmas

    # Check that specific known holidays are present for the date range (2025-12-24 to 2025-12-28)
    holiday_dates = result["date"].tolist()
    holiday_names = result["sanitized_name"].tolist()

    # Christmas Day should be included
    assert date(2025, 12, 25) in holiday_dates
    assert "christmas_day" in holiday_names

    # Second Day of Christmas should be included
    assert date(2025, 12, 26) in holiday_dates
    assert "second_day_of_christmas" in holiday_names

    # Check that sanitized names are properly formatted (no special characters, lowercase)
    for name in holiday_names:
        assert isinstance(name, str)
        assert name.islower()
        assert all(c.isalnum() or c == "_" for c in name)
        assert not name.startswith("_")
        assert not name.endswith("_")


def test_create_general_holiday_feature(sample_dataset: TimeSeriesDataset):
    """Test creation of general is_holiday feature."""
    # Arrange
    holidays_df = pd.DataFrame({
        "date": [date(2025, 12, 25), date(2025, 12, 26)],
        "holiday_name": ["Christmas Day", "Second Day of Christmas"],
        "sanitized_name": ["christmas_day", "second_day_of_christmas"],
    })

    # Act
    result = HolidayFeaturesTransform._create_general_holiday_feature(sample_dataset, holidays_df)

    # Assert
    assert result.name == "is_holiday"
    assert len(result) == len(sample_dataset.index)
    assert result.dtype == int
    # Check specific dates
    assert result.loc["2025-12-24"] == 0  # Not a holiday
    assert result.loc["2025-12-25"] == 1  # Christmas Day
    assert result.loc["2025-12-26"] == 1  # Second Day of Christmas
    assert result.loc["2025-12-27"] == 0  # Not a holiday


def test_create_individual_features(sample_dataset: TimeSeriesDataset):
    """Test creation of individual holiday features."""
    # Arrange
    holidays_df = pd.DataFrame({
        "date": [date(2025, 12, 25), date(2025, 12, 26)],
        "holiday_name": ["Christmas Day", "Second Day of Christmas"],
        "sanitized_name": ["christmas_day", "second_day_of_christmas"],
    })

    # Act
    result = HolidayFeaturesTransform._create_individual_features(sample_dataset, holidays_df)

    # Assert
    expected_columns = {"is_christmas_day", "is_second_day_of_christmas"}
    assert set(result.columns) == expected_columns
    assert len(result) == len(sample_dataset.index)
    # Check specific holiday features
    assert result.loc["2025-12-25", "is_christmas_day"] == 1
    assert result.loc["2025-12-26", "is_christmas_day"] == 0
    assert result.loc["2025-12-26", "is_second_day_of_christmas"] == 1
    assert result.loc["2025-12-25", "is_second_day_of_christmas"] == 0


def test_transform_adds_holiday_features(sample_dataset: TimeSeriesDataset):
    """Test that transform adds holiday features to the dataset using real holiday data."""
    # Arrange
    transform = HolidayFeaturesTransform(country_code=CountryAlpha2("NL"))

    # Act
    result = transform.transform(sample_dataset)

    # Assert
    # Check structure
    assert isinstance(result, TimeSeriesDataset)
    assert result.sample_interval == sample_dataset.sample_interval
    # Check that original features are preserved
    for feature in sample_dataset.feature_names:
        assert feature in result.feature_names
        pd.testing.assert_series_equal(result.data[feature], sample_dataset.data[feature])
    # Check that holiday features are added
    assert "is_holiday" in result.data.columns
    assert "is_christmas_day" in result.data.columns
    assert "is_second_day_of_christmas" in result.data.columns
    # Check that holiday values are correct for known dates
    assert result.data.loc["2025-12-25", "is_holiday"] == 1  # Christmas Day
    assert result.data.loc["2025-12-26", "is_holiday"] == 1  # Second Day of Christmas
    assert result.data.loc["2025-12-24", "is_holiday"] == 0  # Not a holiday
    assert result.data.loc["2025-12-25", "is_christmas_day"] == 1
    assert result.data.loc["2025-12-26", "is_second_day_of_christmas"] == 1


def test_transform_with_real_holidays(sample_dataset: TimeSeriesDataset):
    """Test transform with real holiday data (integration test)."""
    # Arrange
    transform = HolidayFeaturesTransform(country_code=CountryAlpha2("NL"))

    # Act
    result = transform.transform(sample_dataset)

    # Assert
    assert isinstance(result, TimeSeriesDataset)
    assert "is_holiday" in result.data.columns
    # Check that Christmas Day is correctly identified (using real Dutch holiday data)
    assert result.data.loc["2025-12-25", "is_holiday"] == 1
    assert result.data.loc["2025-12-24", "is_holiday"] == 0


def test_transform_with_empty_holidays():
    """Test handling when no holidays are found in the date range."""
    # Arrange
    # Create a dataset for a very short period that realistically has no holidays
    # Use a random Tuesday in March (typically no holidays in Netherlands)
    data = pd.DataFrame(
        {"load": [100.0, 110.0]},
        index=pd.date_range("2025-03-11", periods=2, freq="D"),  # Tuesday-Wednesday in March
    )
    dataset = TimeSeriesDataset(data, timedelta(days=1))

    # Use Netherlands but with a date range that has no holidays
    transform = HolidayFeaturesTransform(country_code=CountryAlpha2("NL"))

    # Act
    result = transform.transform(dataset)

    # Assert
    assert isinstance(result, TimeSeriesDataset)
    assert "is_holiday" in result.data.columns
    # All values should be 0 (no holidays in this specific date range)
    assert (result.data["is_holiday"] == 0).all()


@pytest.mark.parametrize(
    ("holiday_name", "expected_feature"),
    [
        pytest.param("Christmas Day", "is_christmas_day", id="christmas_day"),
        pytest.param("New Year's Day", "is_new_year_s_day", id="new_years_day"),
        pytest.param("Independence Day", "is_independence_day", id="independence_day"),
        pytest.param("Labor Day", "is_labor_day", id="labor_day"),
    ],
)
def test_holiday_name_to_feature_mapping(holiday_name: str, expected_feature: str):
    """Test that holiday names are correctly mapped to feature names."""
    # Arrange
    sample_index = pd.date_range("2025-01-01", periods=1, freq="D")
    sample_data = TimeSeriesDataset(pd.DataFrame({"load": [100]}, index=sample_index), timedelta(days=1))
    holidays_df = pd.DataFrame({
        "date": [date(2025, 1, 1)],
        "holiday_name": [holiday_name],
        "sanitized_name": [HolidayFeaturesTransform._sanitize_holiday_name(holiday_name)],
    })

    # Act
    result = HolidayFeaturesTransform._create_individual_features(sample_data, holidays_df)

    # Assert
    assert expected_feature in result.columns
    assert result.loc["2025-01-01", expected_feature] == 1
