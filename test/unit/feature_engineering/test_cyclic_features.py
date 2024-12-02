# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com>
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import datetime

import pytest
from openstef.feature_engineering.cyclic_features import (
    add_seasonal_cyclic_features,
    add_time_cyclic_features,
    add_daylight_terrestrial_feature
)


class TestCyclicFeatures:
    def test_add_seasonal_cyclic_features(self):
        # Generate a year's worth of daily data
        data = pd.DataFrame(index=pd.date_range("2023-01-01", "2023-12-31", freq="D"))

        # Apply the function
        output_data = add_seasonal_cyclic_features(data)

        # Verify the columns are created
        assert "season_sine" in output_data.columns
        assert "season_cosine" in output_data.columns
        assert "day0fweek_sine" in output_data.columns
        assert "day0fweek_cosine" in output_data.columns
        assert "month_sine" in output_data.columns
        assert "month_cosine" in output_data.columns

        # Validate the seasonality calculations
        days_in_year = 365.25
        season_sine_expected = np.sin(2 * np.pi * data.index.dayofyear / days_in_year)
        season_cosine_expected = np.cos(2 * np.pi * data.index.dayofyear / days_in_year)

        assert np.allclose(output_data["season_sine"], season_sine_expected)
        assert np.allclose(output_data["season_cosine"], season_cosine_expected)

        # Validate the day of week calculations
        dayofweek_sine_expected = np.sin(2 * np.pi * data.index.day_of_week / 7)
        dayofweek_cosine_expected = np.cos(2 * np.pi * data.index.day_of_week / 7)

        assert np.allclose(output_data["day0fweek_sine"], dayofweek_sine_expected)
        assert np.allclose(output_data["day0fweek_cosine"], dayofweek_cosine_expected)

        # Validate the month calculations
        month_sine_expected = np.sin(2 * np.pi * data.index.month / 12)
        month_cosine_expected = np.cos(2 * np.pi * data.index.month / 12)

        assert np.allclose(output_data["month_sine"], month_sine_expected)
        assert np.allclose(output_data["month_cosine"], month_cosine_expected)

    def test_add_time_cyclic_features(self):
        # Generate 2 days of data at 15-minute intervals
        num_points = int(24 * 60 / 15 * 2)
        data = pd.DataFrame(
            index=pd.date_range(
                start="2023-01-01 00:00:00", freq="15min", periods=num_points
            )
        )

        # Apply the function
        output_data = add_time_cyclic_features(data)

        # Verify the columns are created
        assert "time0fday_sine" in output_data.columns
        assert "time0fday_cosine" in output_data.columns

        # Validate the sine and cosine time features
        num_seconds_in_day = 24 * 60 * 60  # Total seconds in a day
        second_of_the_day = (
            data.index.second + data.index.minute * 60 + data.index.hour * 60 * 60
        )
        period_of_the_day = 2 * np.pi * second_of_the_day / num_seconds_in_day

        sin_time_expected = np.sin(period_of_the_day)
        cos_time_expected = np.cos(period_of_the_day)

        assert np.allclose(output_data["time0fday_sine"], sin_time_expected)
        assert np.allclose(output_data["time0fday_cosine"], cos_time_expected)

    def test_add_time_cyclic_features_non_datetime_index(self):
        # Test for non-datetime index
        data = pd.DataFrame(index=range(10))

        with pytest.raises(ValueError, match="Index should be a pandas DatetimeIndex"):
            add_time_cyclic_features(data)

    def test_add_seasonal_cyclic_features_non_datetime_index(self):
        # Test for non-datetime index
        data = pd.DataFrame(index=range(10))

        with pytest.raises(
            ValueError, match="The DataFrame index must be a DatetimeIndex."
        ):
            add_seasonal_cyclic_features(data)


    def test_add_daylight_terrestrial_feature_valid_data(self):
        # Create input data with UTC timezone
        index = pd.date_range("2023-01-01 00:00:00", "2023-01-02 12:45:00", freq="15min", tz="UTC")
        input_data = pd.DataFrame(index=index)

        # Verify the timezone of the index is UTC
        assert input_data.index.tzinfo is not None, "Index must have a timezone."
        assert input_data.index.tzinfo == datetime.timezone.utc, "Timezone of the index must be UTC."

        # Call the function
        output_data = add_daylight_terrestrial_feature(input_data)

        # Verify the output contains the "daylight_continuous" column
        assert "daylight_continuous" in output_data.columns