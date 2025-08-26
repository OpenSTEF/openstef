# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the HolidayFeatures transform."""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.feature_engineering.temporal_transforms.holiday_features import HolidayFeatures


@pytest.fixture
def sample_dataset() -> TimeSeriesDataset:
    """
    Create a sample TimeSeriesDataset for testing.

    Returns:
        TimeSeriesDataset: A dataset with daily frequency spanning Christmas period.
    """
    data = pd.DataFrame(
        {"load": [100.0, 110.0, 120.0, 130.0, 140.0]}, index=pd.date_range("2025-12-24", periods=5, freq="D")
    )
    return TimeSeriesDataset(data, timedelta(days=1))


@pytest.fixture
def mock_holidays() -> dict[date, str]:
    """Mock holidays data for testing.

    Returns:
        A dictionary mapping dates to holiday names.
    """
    return {
        date(2025, 12, 25): "Christmas Day",
        date(2025, 12, 26): "Second Day of Christmas",
        date(2025, 1, 1): "New Year's Day",
    }


def test_holiday_features_initialization():
    """Test HolidayFeatures can be initialized properly."""
    # Arrange
    country_code = "NL"

    # Act
    transform = HolidayFeatures(country_code=country_code)

    # Assert
    assert transform.country_code == "NL"
    assert transform.include_individual_holidays is True
    assert transform.holiday_features.empty


@pytest.mark.parametrize(
    ("country_code", "include_individual"),
    [
        ("NL", True),
        ("US", False),
        ("DE", True),
    ],
)
def test_holiday_features_initialization_parameters(country_code: str, include_individual: bool):
    """Test HolidayFeatures initialization with different parameters."""
    # Arrange
    # Parameters provided by pytest.mark.parametrize

    # Act
    transform = HolidayFeatures(country_code=country_code, include_individual_holiday_features=include_individual)

    # Assert
    assert transform.country_code == country_code
    assert transform.include_individual_holidays == include_individual


@pytest.mark.parametrize(
    ("input_name", "expected_output"),
    [
        ("Christmas Day", "christmas_day"),
        ("New Year's Day", "new_year_s_day"),
        ("St. Patrick's Day", "st_patrick_s_day"),
        ("INDEPENDENCE DAY", "independence_day"),
        ("Labor Day (May Day)", "labor_day_may_day"),
    ],
)
def test_sanitize_holiday_name(input_name: str, expected_output: str):
    """Test holiday name sanitization for feature names."""
    # Act & Assert
    assert HolidayFeatures._sanitize_holiday_name(input_name) == expected_output


def test_create_general_holiday_feature(sample_dataset: TimeSeriesDataset, mock_holidays: dict):
    """Test creation of general is_holiday feature."""
    # Arrange
    index = sample_dataset.index

    # Act
    result = HolidayFeatures._create_general_holiday_feature(index, mock_holidays)

    # Assert
    assert result.name == "is_holiday"
    assert len(result) == len(sample_dataset.index)
    assert result.dtype == int

    # Check specific dates
    assert result.loc["2025-12-24"] == 0  # Not a holiday
    assert result.loc["2025-12-25"] == 1  # Christmas Day
    assert result.loc["2025-12-26"] == 1  # Second Day of Christmas
    assert result.loc["2025-12-27"] == 0  # Not a holiday
    assert result.loc["2025-12-28"] == 0  # Not a holiday


@patch("openstef_core.feature_engineering.temporal_transforms.holiday_features.holidays.country_holidays")
def test_get_country_holidays(mock_country_holidays: MagicMock, sample_dataset: TimeSeriesDataset, mock_holidays: dict):
    """Test getting country holidays."""
    # Arrange
    mock_country_holidays.return_value = mock_holidays
    transform = HolidayFeatures(country_code="NL")

    # Act
    result = transform._get_country_holidays(sample_dataset.index)

    # Assert
    assert result == mock_holidays
    mock_country_holidays.assert_called_once_with(
        "NL", categories=["public"], years=range(2025, 2026), language="en_US"
    )


def test_create_individual_holiday_features_enabled(sample_dataset: TimeSeriesDataset, mock_holidays: dict):
    """Test creation of individual holiday features when enabled."""
    # Arrange
    transform = HolidayFeatures(country_code="NL", include_individual_holiday_features=True)
    index = sample_dataset.index

    # Act
    result = transform._create_individual_holiday_features(index, mock_holidays)

    # Assert
    expected_columns = ["is_christmas_day", "is_new_year_s_day", "is_second_day_of_christmas"]
    assert set(result.columns) == set(expected_columns)
    assert len(result) == len(sample_dataset.index)

    # Check specific holiday features
    assert result.loc["2025-12-25", "is_christmas_day"] == 1
    assert result.loc["2025-12-26", "is_christmas_day"] == 0
    assert result.loc["2025-12-26", "is_second_day_of_christmas"] == 1
    assert result.loc["2025-12-25", "is_second_day_of_christmas"] == 0


def test_create_individual_holiday_features_disabled(sample_dataset: TimeSeriesDataset, mock_holidays: dict):
    """Test creation of individual holiday features when disabled."""
    # Arrange
    transform = HolidayFeatures(country_code="NL", include_individual_holiday_features=False)
    index = sample_dataset.index

    # Act
    result = transform._create_individual_holiday_features(index, mock_holidays)

    # Assert
    assert result.empty
    assert len(result) == len(sample_dataset.index)


@patch("openstef_core.feature_engineering.temporal_transforms.holiday_features.holidays.country_holidays")
def test_fit_creates_holiday_features(
    mock_country_holidays: MagicMock, sample_dataset: TimeSeriesDataset, mock_holidays: dict
):
    """Test that fit creates expected holiday features."""
    # Arrange
    mock_country_holidays.return_value = mock_holidays
    transform = HolidayFeatures(country_code="NL")

    # Act
    transform.fit(sample_dataset)

    # Assert
    assert not transform.holiday_features.empty
    assert "is_holiday" in transform.holiday_features.columns
    assert "is_christmas_day" in transform.holiday_features.columns
    assert "is_second_day_of_christmas" in transform.holiday_features.columns
    assert len(transform.holiday_features) == len(sample_dataset.index)


@patch("openstef_core.feature_engineering.temporal_transforms.holiday_features.holidays.country_holidays")
def test_transform_adds_features(
    mock_country_holidays: MagicMock, sample_dataset: TimeSeriesDataset, mock_holidays: dict
):
    """Test that transform adds holiday features to the dataset."""
    # Arrange
    mock_country_holidays.return_value = mock_holidays
    transform = HolidayFeatures(country_code="NL")
    transform.fit(sample_dataset)

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


def test_fit_transform_integration(sample_dataset: TimeSeriesDataset):
    """Test the complete fit_transform workflow with real holiday data."""
    # Arrange
    transform = HolidayFeatures(country_code="NL")

    # Act
    result = transform.fit_transform(sample_dataset)

    # Assert
    assert isinstance(result, TimeSeriesDataset)
    assert "is_holiday" in result.data.columns

    # Check that Christmas Day is correctly identified
    assert result.data.loc["2025-12-25", "is_holiday"] == 1
    assert result.data.loc["2025-12-24", "is_holiday"] == 0


def test_empty_holidays_dict(sample_dataset: TimeSeriesDataset):
    """Test handling of empty holidays dictionary."""
    # Arrange
    transform = HolidayFeatures(country_code="NL")
    empty_holidays = {}
    index = sample_dataset.index

    # Act
    general_result = transform._create_general_holiday_feature(index, empty_holidays)
    individual_result = transform._create_individual_holiday_features(index, empty_holidays)

    # Assert
    # Test general feature with empty holidays
    assert (general_result == 0).all()

    # Test individual features with empty holidays
    assert individual_result.empty


@pytest.mark.parametrize(
    ("holiday_name", "expected_feature"),
    [
        ("Christmas Day", "is_christmas_day"),
        ("New Year's Day", "is_new_year_s_day"),
        ("Independence Day", "is_independence_day"),
        ("Labor Day", "is_labor_day"),
    ],
)
def test_holiday_name_to_feature_mapping(holiday_name: str, expected_feature: str):
    """Test that holiday names are correctly mapped to feature names."""
    # Arrange
    mock_holidays = {date(2025, 1, 1): holiday_name}
    sample_index = pd.date_range("2025-01-01", periods=1, freq="D")
    transform = HolidayFeatures(country_code="NL")

    # Act
    result = transform._create_individual_holiday_features(sample_index, mock_holidays)

    # Assert
    assert expected_feature in result.columns
    assert result.loc["2025-01-01", expected_feature] == 1
