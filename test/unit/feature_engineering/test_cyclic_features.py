# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com>
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import pytest

from openstef.feature_engineering.cyclic_features import (
    add_seasonal_cyclic_features,
    add_time_cyclic_features,
)


class TestCyclicFeatures:
    def test_add_seasonal_cyclic_features(self):
        # Generate a year's worth of daily data
        data = pd.DataFrame(index=pd.date_range("2023-01-01", "2023-12-31", freq="D"))

        # Apply the function
        output_data = add_seasonal_cyclic_features(data)

        # Verify the columns are created
        assert "season_sin" in output_data.columns
        assert "season_cos" in output_data.columns
        assert "dayofweek_sin" in output_data.columns
        assert "dayofweek_cos" in output_data.columns
        assert "month_sin" in output_data.columns
        assert "month_cos" in output_data.columns

        # Validate the seasonality calculations
        days_in_year = 365.25
        season_sin_expected = np.sin(2 * np.pi * data.index.dayofyear / days_in_year)
        season_cos_expected = np.cos(2 * np.pi * data.index.dayofyear / days_in_year)

        assert np.allclose(output_data["season_sin"], season_sin_expected)
        assert np.allclose(output_data["season_cos"], season_cos_expected)

        # Validate the day of week calculations
        dayofweek_sin_expected = np.sin(2 * np.pi * data.index.day_of_week / 7)
        dayofweek_cos_expected = np.cos(2 * np.pi * data.index.day_of_week / 7)

        assert np.allclose(output_data["dayofweek_sin"], dayofweek_sin_expected)
        assert np.allclose(output_data["dayofweek_cos"], dayofweek_cos_expected)

        # Validate the month calculations
        month_sin_expected = np.sin(2 * np.pi * data.index.month / 12)
        month_cos_expected = np.cos(2 * np.pi * data.index.month / 12)

        assert np.allclose(output_data["month_sin"], month_sin_expected)
        assert np.allclose(output_data["month_cos"], month_cos_expected)

    def test_add_time_cyclic_features(self):
        # Generate 2 days of data at 15-minute intervals
        num_points = int(24 * 60 / 15 * 2)
        data = pd.DataFrame(
            index=pd.date_range(
                start="2023-01-01 00:00:00", freq="15min", periods=num_points
            )
        )

        # Apply the function with default period (24 hours)
        output_data = add_time_cyclic_features(data, frequency="15min")

        # Verify the columns are created
        assert "sin_time" in output_data.columns
        assert "cos_time" in output_data.columns

        # Validate the sine and cosine time features
        base_interval_minutes = 15  # Interval in minutes
        time_in_minutes = data.index.hour * 60 + data.index.minute
        time_interval = time_in_minutes / base_interval_minutes

        intervals_per_cycle = (24 * 60) / base_interval_minutes

        sin_time_expected = np.sin(2 * np.pi * time_interval / intervals_per_cycle)
        cos_time_expected = np.cos(2 * np.pi * time_interval / intervals_per_cycle)

        assert np.allclose(output_data["sin_time"], sin_time_expected)
        assert np.allclose(output_data["cos_time"], cos_time_expected)

    def test_add_time_cyclic_features_invalid_frequency(self):
        # Test for invalid frequency input
        data = pd.DataFrame(index=pd.date_range("2023-01-01", periods=10, freq="h"))

        with pytest.raises(ValueError, match="Invalid frequency string"):
            add_time_cyclic_features(data, frequency="invalid_freq")

    def test_add_seasonal_cyclic_features_non_datetime_index(self):
        # Test for non-datetime index
        data = pd.DataFrame(index=range(10))

        with pytest.raises(
            ValueError, match="The DataFrame index must be a DatetimeIndex"
        ):
            add_seasonal_cyclic_features(data)

    def test_add_time_cyclic_features_non_datetime_index(self):
        # Test for non-datetime index
        data = pd.DataFrame(index=range(10))

        with pytest.raises(
            ValueError, match="The DataFrame index must be a DatetimeIndex"
        ):
            add_time_cyclic_features(data)

    def test_add_time_cyclic_features_invalid_period(self):
        # Test for invalid period input
        data = pd.DataFrame(index=pd.date_range("2023-01-01", periods=10, freq="h"))

        with pytest.raises(
            ValueError, match="The 'period' parameter must be greater than 0"
        ):
            add_time_cyclic_features(data, period=-5)
