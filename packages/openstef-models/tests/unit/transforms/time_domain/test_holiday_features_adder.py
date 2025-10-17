# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the HolidayFeatureAdder."""

from datetime import date, timedelta

import pandas as pd
import pytest
from pydantic_extra_types.country import CountryAlpha2

from openstef_core.datasets import TimeSeriesDataset
from openstef_models.transforms.time_domain.holiday_features_adder import (
    HolidayFeatureAdder,
    get_holiday_names,
    get_holidays,
    sanitize_holiday_name,
)

# Expected holidays for Netherlands in 2025 for testing
NL_2025_HOLIDAYS = [
    (date(2025, 1, 1), "New Year's Day", "new_year_s_day"),
    (date(2025, 4, 20), "Easter Sunday", "easter_sunday"),
    (date(2025, 4, 21), "Easter Monday", "easter_monday"),
    (date(2025, 4, 26), "King's Day", "king_s_day"),
    (date(2025, 5, 5), "Liberation Day", "liberation_day"),
    (date(2025, 5, 29), "Ascension Day", "ascension_day"),
    (date(2025, 6, 8), "Whit Sunday", "whit_sunday"),
    (date(2025, 6, 9), "Whit Monday", "whit_monday"),
    (date(2025, 12, 25), "Christmas Day", "christmas_day"),
    (date(2025, 12, 26), "Second Day of Christmas", "second_day_of_christmas"),
]


@pytest.mark.parametrize(
    ("input_name", "expected_output"),
    [
        pytest.param("Christmas Day", "christmas_day", id="christmas_day"),
        pytest.param("New Year's Day", "new_year_s_day", id="apostrophe"),
        pytest.param("St. Patrick's Day", "st_patrick_s_day", id="period_and_apostrophe"),
        pytest.param("INDEPENDENCE DAY", "independence_day", id="uppercase"),
        pytest.param("Labor Day (May Day)", "labor_day_may_day", id="parentheses"),
    ],
)
def test_sanitize_holiday_name(input_name: str, expected_output: str):
    """Test sanitization of holiday names to valid Python identifiers."""
    # Act
    result = sanitize_holiday_name(input_name)

    # Assert
    assert result == expected_output


@pytest.mark.parametrize(
    ("country_code", "expected_count", "expected_holidays"),
    [
        pytest.param(
            CountryAlpha2("NL"),
            10,
            ["christmas_day", "new_year_s_day", "king_s_day", "liberation_day"],
            id="netherlands",
        ),
        pytest.param(
            CountryAlpha2("US"),
            11,
            ["christmas_day", "new_year_s_day", "independence_day", "thanksgiving_day"],
            id="united_states",
        ),
    ],
)
def test_get_holiday_names(country_code: CountryAlpha2, expected_count: int, expected_holidays: list[str]):
    """Test that get_holiday_names returns correct count and expected holidays."""
    # Act
    result = get_holiday_names(country_code)

    # Assert
    # Check count of unique holidays
    assert len(result) == expected_count
    # Check that expected holidays are present
    for holiday in expected_holidays:
        assert holiday in result
    # Verify sorted order and uniqueness
    assert result == sorted(result)
    assert len(result) == len(set(result))


def test_get_holidays():
    """Test that get_holidays returns expected holidays with correct structure."""
    # Arrange
    index = pd.date_range("2025-01-01", periods=365, freq="D")
    expected_df = pd.DataFrame(NL_2025_HOLIDAYS, columns=["date", "holiday_name", "sanitized_name"])

    # Act
    result = get_holidays(index=index, country=CountryAlpha2("NL"))

    # Assert
    # Check DataFrame structure and all expected holidays are present
    pd.testing.assert_frame_equal(result.sort_values("date").reset_index(drop=True), expected_df)


@pytest.mark.parametrize(
    ("start_date", "periods", "expected_holiday_dates"),
    [
        pytest.param(
            "2025-12-24",
            5,
            {
                "2025-12-24": 0,
                "2025-12-25": 1,  # Christmas Day
                "2025-12-26": 1,  # Second Day of Christmas
                "2025-12-27": 0,
                "2025-12-28": 0,
            },
            id="christmas_period",
        ),
        pytest.param(
            "2025-03-11",
            2,
            {"2025-03-11": 0, "2025-03-12": 0},
            id="no_holidays",
        ),
    ],
)
def test_holiday_feature_adder_transform(start_date: str, periods: int, expected_holiday_dates: dict[str, int]):
    """Test that HolidayFeatureAdder correctly adds holiday features."""
    # Arrange
    data = pd.DataFrame(
        {"load": [100.0 + i * 10 for i in range(periods)]},
        index=pd.date_range(start_date, periods=periods, freq="D"),
    )
    dataset = TimeSeriesDataset(data, timedelta(days=1))
    transform = HolidayFeatureAdder(country_code=CountryAlpha2("NL"))

    # Act
    result = transform.transform(dataset)

    # Assert
    # Check result structure
    assert isinstance(result, TimeSeriesDataset)
    assert result.sample_interval == dataset.sample_interval
    # Check original data is preserved (columns and index)
    pd.testing.assert_frame_equal(result.data[["load"]], dataset.data)
    # Check is_holiday feature is added with correct values
    assert "is_holiday" in result.data.columns
    for date_str, expected_value in expected_holiday_dates.items():
        assert result.data["is_holiday"].loc[date_str] == expected_value
    # Check all individual holiday features are added
    all_holiday_names = get_holiday_names(CountryAlpha2("NL"))
    all_holiday_features = {f"is_{holiday_name}" for holiday_name in all_holiday_names}
    assert all_holiday_features.issubset(result.data.columns)
    # Verify specific holiday feature naming (from hardcoded list)
    expected_features = {"is_christmas_day", "is_new_year_s_day", "is_king_s_day"}
    assert expected_features.issubset(result.data.columns)
