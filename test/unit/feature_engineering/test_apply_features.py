# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from openstef.enums import AggregateFunction, BiddingZone
from openstef.feature_engineering import apply_features, weather_features
from openstef.feature_engineering.feature_applicator import TrainFeatureApplicator
from openstef.feature_engineering.lag_features import (
    generate_lag_feature_functions,
    generate_non_trivial_lag_times,
)


class TestApplyFeaturesModule(BaseTestCase):
    def test_generate_lag_functions(self):
        """Test generate lag functions.

            Test the `generate_lag_functions` function and compare the keys of the
            returned dictionary (the lag function names) with a previously saved set
            of lag functions names

        Raises:
            AssertionError: When the lag function names are different then the expected
                function names
        """
        lag_functions = generate_lag_feature_functions(horizon=24.0)

        lag_functions_keys = sorted(lag_functions.keys())

        expected_lag_functions_keys = sorted(TestData.LAG_FUNCTIONS_KEYS)

        self.assertEqual(lag_functions_keys, expected_lag_functions_keys)

    def test_generate_lag_features_with_features(self):
        """Does the function work properly if specific features are given"""
        specific_features = ["T-30min", "T-7d"]
        lag_functions = generate_lag_feature_functions(
            horizon=0.25, feature_names=specific_features
        )
        self.assertEqual(len(lag_functions), 2)

    def test_generate_lag_features_with_unavailable_horizon(self):
        """Does the function work properly if specific features are given"""
        specific_features = ["T-30min", "T-7d"]  # feature = T-30min, horizon=24
        lag_functions = generate_lag_feature_functions(
            horizon=24, feature_names=specific_features
        )
        self.assertEqual(
            list(lag_functions.keys()), ["T-7d"]
        )  # Only T-7d should be returned

    def test_additional_minute_space(self):
        additional_minute_lags_list = generate_non_trivial_lag_times(
            data=TestData.load("input_data_train.csv"), height_threshold=0.1
        )
        expected_additional_minute_lags_list = [1410, 2880]
        self.assertEqual(
            additional_minute_lags_list, expected_additional_minute_lags_list
        )

    def test_additional_minute_space_empty_data(self):
        additional_minute_lags_list = generate_non_trivial_lag_times(pd.DataFrame())
        self.assertEqual(len(additional_minute_lags_list), 0)

    def test_additional_minute_space_no_peaks_in_correlation(self):
        input_data = pd.DataFrame(
            {"random_column_name": np.linspace(0, 1000, 1000, endpoint=False)}
        )

        additional_minute_lags_list = generate_non_trivial_lag_times(input_data)
        self.assertEqual(len(additional_minute_lags_list), 0)

    def test_apply_features_with_windspeed_100m(self):
        """Test the 'apply_features' function.

            Test if the returned data frame with the generated and added features is
            equal to a previously saved data frame.

        Raises:
            AssertionError: When the returned data frame is not equal to the expected one
        """
        input_data_with_features = apply_features.apply_features(
            data=TestData.load("input_data.csv"),
            horizon=24,
            pj={"model": "xgb", "lat": 52.132633, "lon": 5.291266},
        )

        expected_output = TestData.load("input_data_with_features.csv")
        expected_output.drop(
            columns=["windspeed_100mExtrapolated", "windPowerFit_extrapolated"],
            inplace=True,
        )
        self.assertDataframeEqual(
            input_data_with_features,
            expected_output,
            check_like=True,  # ignore the order of index & columns
            check_dtype=False,
        )

    def test_apply_features_without_windspeed_100m(self):
        """Test the 'apply_features' function.

            Test if the returned data frame with the generated and added features is
            equal to a previously saved data frame.

        Raises:
            AssertionError: When the returned data frame is not equal to the expected one
        """

        full_input_data = TestData.load("input_data.csv")
        input_data_with_features = apply_features.apply_features(
            data=full_input_data.drop(columns=["windspeed_100m"]),
            horizon=24,
            pj={"model": "xgb", "lat": 52.132633, "lon": 5.291266},
        )

        expected_output = TestData.load("input_data_with_features.csv")
        expected_output.drop(
            columns=["windspeed_100m", "windpowerFit_harm_arome"], inplace=True
        )
        self.assertDataframeEqual(
            input_data_with_features,
            expected_output,
            check_like=True,  # ignore the order of index & columns
            check_dtype=False,
        )

    def test_apply_features_no_pj(self):
        """pj = None should work fine as well"""
        # Arrange
        input_data_without_features = TestData.load("input_data.csv")
        pj = None

        # Act
        input_data_with_features = apply_features.apply_features(
            input_data_without_features,
            horizon=24,
            pj=pj,
        )

        # Assert:
        assert len(input_data_with_features) == len(input_data_without_features)
        assert "gti" in list(input_data_with_features.columns)
        assert "dni" in list(input_data_with_features.columns)

    def test_train_feature_applicator_with_windspeed_100m(self):
        input_data_with_features = TrainFeatureApplicator(horizons=[0.25]).add_features(
            TestData.load("input_data.csv"),
            pj={"model": "proleaf", "lat": 52.132633, "lon": 5.291266},
        )

        expected_output = TestData.load("input_data_multi_horizon_features.csv")
        self.assertDataframeEqual(
            input_data_with_features,
            expected_output,
            check_like=True,  # ignore the order of index & columns
            check_dtype=False,
        )

    def test_train_feature_applicator_with_latency(self):
        input_data = pd.DataFrame(
            index=pd.to_datetime(
                [
                    "2020-02-01 10:00:00",
                    "2020-02-01 10:30:00",
                    "2020-02-01 11:00:00",
                    "2020-02-01 11:30:00",
                ]
            ).tz_localize("UTC"),
            data={
                "load": [10, 15, 20, 15],
                "day_ahead_electricity_price": [1, 2, 3, 4],
            },
        )
        horizons = [0.25, 47]

        input_data_with_features = TrainFeatureApplicator(
            horizons=horizons
        ).add_features(input_data)

        horizon = input_data_with_features.horizon

        # Skip first row, since T-30min not available for first row
        self.assertTrue(
            input_data_with_features.loc[
                horizon == 47, ["day_ahead_electricity_price", "T-30min"]
            ]
            .iloc[1:,]
            .isna()
            .all()
            .all()
        )
        self.assertFalse(
            input_data_with_features.loc[
                horizon == 0.25, ["day_ahead_electricity_price", "T-30min"]
            ]
            .iloc[1:,]
            .isna()
            .any()
            .any()
        )

    def test_apply_holiday_features(self):
        input_data = pd.DataFrame(
            index=pd.to_datetime(
                [
                    "2020-02-01 10:00:00",
                    "2020-02-01 10:10:00",
                    "2022-12-26 10:00:00",
                    "2020-04-27 11:00:00",
                ]
            ).tz_localize("UTC"),
            data={
                "load": [10, 15, 20, 15],
                "temp": [9, 9, 9, 9],
                "humidity": [1, 2, 3.0, 4.0],
                "pressure": [3, 4, 5, 6],
            },
        )
        pj = {
            "model": "proleaf",
            "lat": 52.132633,
            "lon": 5.291266,
            "electricity_bidding_zone": BiddingZone.NL,
        }

        input_data_with_features = apply_features.apply_features(
            pj=pj, data=input_data, horizon=24, years=[2020, 2021, 2022, 2023]
        )

        expected = TestData.load("../data/input_data_with_holiday_features.csv")

        self.assertDataframeEqual(
            input_data_with_features,
            expected,
            check_like=True,  # ignore the order of index & columns
            check_dtype=False,
        )

    def test_calculate_windspeed_at_hubheight_realistic_input(self):
        windspeed = 20
        from_height = 10
        hub_height = 100
        expected_wind_speed_at_hub_height = 27.799052624267063
        wind_speed_at_hub_height = weather_features.calculate_windspeed_at_hubheight(
            windspeed, from_height, hub_height
        )
        self.assertAlmostEqual(
            wind_speed_at_hub_height, expected_wind_speed_at_hub_height
        )

    def test_calculate_windspeed_at_hubheight_wrong_wind_speed_datatype(self):
        with self.assertRaises(TypeError):
            weather_features.calculate_windspeed_at_hubheight("20.25", 10, 100)

    def test_calculate_windspeed_at_hubheight_no_wind(self):
        wind_speed_at_hub_height = weather_features.calculate_windspeed_at_hubheight(
            0, 10, 100
        )
        self.assertEqual(wind_speed_at_hub_height, 0)

    def test_calculate_windspeed_at_hubheight_nan_input(self):
        wind_speed_nan = weather_features.calculate_windspeed_at_hubheight(float("nan"))
        self.assertIsNAN(wind_speed_nan)

    def test_calculate_windspeed_at_hubheight_negative_input(self):
        negative_windspeed = -5
        with self.assertRaises(ValueError):
            weather_features.calculate_windspeed_at_hubheight(negative_windspeed)

        negative_windspeeds = pd.Series([-1, 2, 3, 4])
        with self.assertRaises(ValueError):
            weather_features.calculate_windspeed_at_hubheight(negative_windspeeds)

    def test_calculate_windspeed_at_hubheight_list_input(self):
        windspeeds_list = [1, 2, 3, 4]
        expected_extrapolated_windspeeds = [1.3899526, 2.7799052, 4.16985, 5.559810]

        windspeeds = pd.Series(windspeeds_list)
        extrapolated_windspeeds = weather_features.calculate_windspeed_at_hubheight(
            windspeeds
        )
        self.assertSeriesEqual(
            extrapolated_windspeeds, pd.Series(expected_extrapolated_windspeeds)
        )

        windspeeds = np.array(windspeeds_list)
        extrapolated_windspeeds = weather_features.calculate_windspeed_at_hubheight(
            windspeeds
        )
        self.assertArrayEqual(
            extrapolated_windspeeds.round(decimals=3),
            np.array(expected_extrapolated_windspeeds).round(decimals=3),
        )

    def test_calculate_windturbine_power_output_no_wind(self):
        power = weather_features.calculate_windturbine_power_output(0)
        self.assertAlmostEqual(power, 0, places=2)

    def test_apply_power_curve_nan_wind(self):
        power = weather_features.calculate_windturbine_power_output(float("nan"))
        self.assertIsNAN(power)

    def test_calculate_windturbine_power_output_realistic_values(self):
        power_v5 = weather_features.calculate_windturbine_power_output(5)
        self.assertAlmostEqual(power_v5, 0.11522159872442202)

        windspeed_v8 = weather_features.calculate_windturbine_power_output(8)
        self.assertAlmostEqual(windspeed_v8, 0.48838209152618717)

        # The generated  power should level off to the rated power for high wind speeds
        power_v100 = weather_features.calculate_windturbine_power_output(100)
        self.assertAlmostEqual(power_v100, 1)

    def test_calculate_windturbine_power_output(self):
        windspeed = 20
        n_turbines = 1
        turbine_data = {"slope_center": 1, "rated_power": 1, "steepness": 0.1}
        power_output = weather_features.calculate_windturbine_power_output(
            windspeed, n_turbines, turbine_data
        )
        expected_power_output = 0.8698915256370021
        self.assertAlmostEqual(power_output, expected_power_output)

    def test_add_rolling_aggregate_features(self):
        pj = {
            "rolling_aggregate_features": [
                AggregateFunction.MEAN,
                AggregateFunction.MAX,
                AggregateFunction.MIN,
            ],
            "electricity_bidding_zone": BiddingZone.NL,
        }

        input_data = pd.DataFrame(
            index=pd.date_range(
                start="2023-01-01 00:00:00", freq="15min", periods=100, tz="UTC"
            )
        )
        input_data["load"] = list(range(100))

        input_data_with_features = apply_features.apply_features(
            data=input_data,
            pj=pj,
        )

        self.assertIn("rolling_mean_load_P1D", input_data_with_features.columns)
        self.assertIn("rolling_max_load_P1D", input_data_with_features.columns)
        self.assertIn("rolling_min_load_P1D", input_data_with_features.columns)

    def test_add_rolling_aggregate_features_when_none(self):
        pj = {
            "rolling_aggregate_features": None,
            "electricity_bidding_zone": BiddingZone.NL,
        }

        input_data = pd.DataFrame(
            index=pd.date_range(
                start="2023-01-01 00:00:00", freq="15min", periods=100, tz="UTC"
            )
        )
        input_data["load"] = list(range(100))

        input_data_with_features = apply_features.apply_features(
            data=input_data,
            pj=pj,
        )
        self.assertTrue(
            all(["rolling" not in x for x in input_data_with_features.columns])
        )


if __name__ == "__main__":
    unittest.main()
