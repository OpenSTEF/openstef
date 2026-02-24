# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <openstef@lfenergy.org> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import pandas as pd
import numpy as np
import pytest
from openstef.feature_engineering.holiday_features import (
    generate_holiday_feature_functions,
)
from openstef.feature_engineering import apply_features
from openstef.enums import BiddingZone
from test.unit.utils.base import BaseTestCase


class TestHolidayDetection(BaseTestCase):
    def test_full_year_holiday_counts(self):
        """Test a full year of data to count and verify all holidays."""
        # Create a dataframe with a full year of data at hourly intervals
        full_year = pd.date_range(
            start="2023-01-01 00:00:00", end="2023-12-31 23:00:00", freq="1H"
        ).tz_localize("UTC")

        input_data = pd.DataFrame(
            index=full_year,
            data={"load": np.ones(len(full_year))},
        )

        pj = {"electricity_bidding_zone": BiddingZone.NL}

        # Apply holiday features for the year 2023
        input_data_with_features = apply_features.apply_features(
            pj=pj, data=input_data, horizon=24, years=[2023]
        )

        # Count days for each holiday (divide by 24 since we have hourly data)
        holiday_counts = {
            "nieuwjaarsdag": input_data_with_features["is_nieuwjaarsdag"].sum() / 24,
            "goede_vrijdag": input_data_with_features["is_goede_vrijdag"].sum() / 24,
            "eerste_paasdag": input_data_with_features["is_eerste_paasdag"].sum() / 24,
            "tweede_paasdag": input_data_with_features["is_tweede_paasdag"].sum() / 24,
            "koningsdag": input_data_with_features["is_koningsdag"].sum() / 24,
            "hemelvaart": input_data_with_features["is_hemelvaart"].sum() / 24,
            "eerste_pinksterdag": input_data_with_features[
                "is_eerste_pinksterdag"
            ].sum()
            / 24,
            "tweede_pinksterdag": input_data_with_features[
                "is_tweede_pinksterdag"
            ].sum()
            / 24,
            "eerste_kerstdag": input_data_with_features["is_eerste_kerstdag"].sum()
            / 24,
            "tweede_kerstdag": input_data_with_features["is_tweede_kerstdag"].sum()
            / 24,
        }

        # Expected counts for 2023
        expected_counts = {
            "nieuwjaarsdag": 1,
            "goede_vrijdag": 1,
            "eerste_paasdag": 1,
            "tweede_paasdag": 1,
            "koningsdag": 1,
            "hemelvaart": 1,
            "eerste_pinksterdag": 1,
            "tweede_pinksterdag": 1,
            "eerste_kerstdag": 1,
            "tweede_kerstdag": 1,
        }

        # Assert that each holiday appears the expected number of times
        for holiday, expected_count in expected_counts.items():
            assert (
                holiday_counts[holiday] == expected_count
            ), f"Expected {expected_count} days for {holiday}, but found {holiday_counts[holiday]}"

        # Also check specific dates for some key holidays
        new_years_day = input_data_with_features.loc["2023-01-01"]
        assert new_years_day[
            "is_nieuwjaarsdag"
        ].all(), "January 1st, 2023 not marked as New Year's Day"

        christmas_day = input_data_with_features.loc["2023-12-25"]
        assert christmas_day[
            "is_eerste_kerstdag"
        ].all(), "December 25th, 2023 not marked as First Christmas Day"

    def test_direct_holiday_generation(self):
        """Test directly calling the holiday function generator."""
        # Create test data for New Year's Day across multiple years
        dates = [
            "2020-01-01 10:00:00",
            "2021-01-01 10:00:00",
            "2022-01-01 10:00:00",
            "2023-01-01 10:00:00",
            "2024-01-01 10:00:00",
            "2025-01-01 10:00:00",
            # Add King's Day dates
            "2020-04-27 10:00:00",
            "2021-04-27 10:00:00",
            "2022-04-27 10:00:00",
            "2023-04-27 10:00:00",
        ]

        test_data = pd.DataFrame(
            index=pd.to_datetime(dates).tz_localize("UTC"),
            data={"load": np.ones(len(dates))},
        )

        years = list(range(2020, 2026))

        # Generate holiday functions
        holiday_functions = generate_holiday_feature_functions(
            country_code="NL", years=years
        )

        # Test each relevant holiday function
        results = pd.DataFrame(index=test_data.index)
        results["is_national_holiday"] = test_data.iloc[:, [0]].apply(
            holiday_functions["is_national_holiday"]
        )

        # Check if the holiday functions for specific holidays are present
        assert "is_nieuwjaarsdag" in holiday_functions, "Nieuwjaarsdag function missing"
        assert "is_koningsdag" in holiday_functions, "Koningsdag function missing"

        # Apply the specific holiday functions
        results["is_nieuwjaarsdag"] = test_data.iloc[:, [0]].apply(
            holiday_functions["is_nieuwjaarsdag"]
        )
        results["is_koningsdag"] = test_data.iloc[:, [0]].apply(
            holiday_functions["is_koningsdag"]
        )

        # Test that all New Year's days are detected correctly
        new_years_days = results.iloc[:6]
        kings_days = results.iloc[6:]

        # Check New Year's days
        assert new_years_days[
            "is_nieuwjaarsdag"
        ].all(), "Not all January 1st dates are marked as New Year's Day"
        assert new_years_days[
            "is_national_holiday"
        ].all(), "Not all January 1st dates are marked as national holidays"

        # Check King's days
        assert kings_days[
            "is_koningsdag"
        ].all(), "Not all April 27th dates are marked as Koningsdag"
        assert kings_days[
            "is_national_holiday"
        ].all(), "Not all April 27th dates are marked as national holidays"


if __name__ == "__main__":
    pytest.main()
