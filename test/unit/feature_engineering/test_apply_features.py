# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import unittest
from test.utils import BaseTestCase, TestData

import numpy as np
import pandas as pd

from stf.feature_engineering import apply_features


class TestApplyFeaturesModule(BaseTestCase):

    def test_generate_lag_functions(self):
        """Test generate lag functions.

            Test the `generate_lag_functions` function and compare the keys of the
            returned dictionairy (the lag function names) with a previously saved set
            of lag functions names

        Raises:
            AssertionError: When the lag function names are different then the expected
                function names
        """
        lag_functions = apply_features.generate_lag_functions(
            TestData.load("input_data_train.pickle")
        )

        lag_functions_keys = sorted(lag_functions.keys())
        expected_lag_functions_keys = sorted(TestData.LAG_FUNCTIONS_KEYS)

        self.assertEqual(lag_functions_keys, expected_lag_functions_keys)

    def test_additional_minute_space(self):
        additional_minute_lags_list = apply_features.additional_minute_space(
            data=TestData.load("input_data_train.pickle"), height_treshold=0.1
        )
        expected_additional_minute_lags_list = [1410, 2880]
        self.assertEqual(
            additional_minute_lags_list, expected_additional_minute_lags_list
        )

    def test_additional_minute_space_empty_data(self):
        additional_minute_lags_list = apply_features.additional_minute_space(
            pd.DataFrame()
        )
        self.assertEqual(len(additional_minute_lags_list), 0)

    def test_additional_minute_space_no_peaks_in_correlation(self):
        input_data = pd.DataFrame(
            {"random_column_name": np.linspace(0, 1000, 1000, endpoint=False)}
        )

        additional_minute_lags_list = apply_features.additional_minute_space(input_data)
        self.assertEqual(len(additional_minute_lags_list), 0)

    def test_apply_features(self):
        """Test the 'apply_features' function.

            Test if the returned data frame with the generated and added features is
            equal to a previously saved data frame.

        Raises:
            AssertionError: When the returned data frame is not equal to the expected one
        """
        input_data_with_features = apply_features.apply_features(
            # data=self.test_data.INPUT_DATA, h_ahead=24
            data=TestData.load("input_data.pickle"), h_ahead=24
        )
        self.assertDataframeEqual(
            input_data_with_features,
            TestData.load("input_data_with_features.pickle"),
            check_like=True,  # ignore the order of index & columns
        )

    def test_apply_multiple_horizon_features(self):
        input_data_with_features = apply_features.apply_multiple_horizon_features(
            data=TestData.load("input_data.pickle"), h_aheads=[0.25]
        )
        self.assertDataframeEqual(
            input_data_with_features,
            TestData.load("input_data_multi_horizon_features.pickle"),
            check_like=True,  # ignore the order of index & columns
        )

    def test_apply_multiple_horizon_features_with_latency(self):
        input_data = pd.DataFrame(
            index=pd.to_datetime(
                [
                    "2020-02-01 10:00:00",
                    "2020-02-01 10:10:00",
                    "2022-12-26 10:00:00",
                    "2020-04-27 11:00:00",
                ]
            ),
            data={"load": [10, 15, 20, 15], "APX": [1, 2, 3, 4]},
        )
        h_aheads = [0.25, 47]
        input_data_with_features = apply_features.apply_multiple_horizon_features(
            input_data, h_aheads=h_aheads, latency={"APX": 24}
        )
        horizon = input_data_with_features.Horizon

        self.assertTrue(
            input_data_with_features.loc[horizon == 47, "APX"].isnull().all()
        )
        self.assertFalse(
            input_data_with_features.loc[horizon == 0.25, "APX"].isnull().all()
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
            ),
            data={
                "load": [10, 15, 20, 15],
                "temp": [9, 9, 9, 9],
                "humidity": [1, 2, 3, 4],
                "pressure": [3, 4, 5, 6],
            },
        )
        input_data_with_features = apply_features.apply_features(
            data=input_data, minute_list=[], h_ahead=24
        )
        self.assertDataframeEqual(
            input_data_with_features,
            TestData.load("input_data_holiday_features.pickle"),
            check_like=True,  # ignore the order of index & columns
        )

    def test_calculate_windspeed_at_hubheight_realistic_input(self):
        windspeed = 20
        from_height = 10
        hub_height = 100
        expected_wind_speed_at_hub_height = 27.799052624267063
        wind_speed_at_hub_height = apply_features.calculate_windspeed_at_hubheight(
            windspeed, from_height, hub_height
        )
        self.assertAlmostEqual(
            wind_speed_at_hub_height, expected_wind_speed_at_hub_height
        )
        windspeed = 0

    def test_calculate_windspeed_at_hubheight_wrong_wind_speed_datatype(self):
        with self.assertRaises(TypeError):
            apply_features.calculate_windspeed_at_hubheight("20.25", 10, 100)

    def test_calculate_windspeed_at_hubheight_no_wind(self):
        wind_speed_at_hub_height = apply_features.calculate_windspeed_at_hubheight(
            0, 10, 100
        )
        self.assertEqual(wind_speed_at_hub_height, 0)

    def test_calculate_windspeed_at_hubheight_nan_input(self):
        wind_speed_nan = apply_features.calculate_windspeed_at_hubheight(float("nan"))
        self.assertIsNAN(wind_speed_nan)

    def test_calculate_windspeed_at_hubheight_negative_input(self):
        negative_windspeed = -5
        with self.assertRaises(ValueError):
            apply_features.calculate_windspeed_at_hubheight(negative_windspeed)

        negative_windspeeds = pd.Series([-1, 2, 3, 4])
        with self.assertRaises(ValueError):
            apply_features.calculate_windspeed_at_hubheight(negative_windspeeds)

    def test_calculate_windspeed_at_hubheight_list_input(self):
        windspeeds_list = [1, 2, 3, 4]
        expected_extrapolated_windspeeds = [1.3899526, 2.7799052, 4.16985, 5.559810]

        windspeeds = pd.Series(windspeeds_list)
        extrapolated_windspeeds = apply_features.calculate_windspeed_at_hubheight(
            windspeeds
        )
        self.assertSeriesEqual(
            extrapolated_windspeeds, pd.Series(expected_extrapolated_windspeeds)
        )

        windspeeds = np.array(windspeeds_list)
        extrapolated_windspeeds = apply_features.calculate_windspeed_at_hubheight(
            windspeeds
        )
        self.assertArrayEqual(
            extrapolated_windspeeds.round(decimals=3),
            np.array(expected_extrapolated_windspeeds).round(decimals=3),
        )

    def test_calculate_windturbine_power_output_no_wind(self):
        power = apply_features.calculate_windturbine_power_output(0)
        self.assertAlmostEqual(power, 0, places=2)

    def test_apply_power_curve_nan_wind(self):
        power = apply_features.calculate_windturbine_power_output(float("nan"))
        self.assertIsNAN(power)

    def test_calculate_windturbine_power_output_realistic_values(self):

        power_v5 = apply_features.calculate_windturbine_power_output(5)
        self.assertAlmostEqual(power_v5, 0.11522159872442202)

        windspeed_v8 = apply_features.calculate_windturbine_power_output(8)
        self.assertAlmostEqual(windspeed_v8, 0.48838209152618717)

        # The generated  power should level off to the rated power for high wind speeds
        power_v100 = apply_features.calculate_windturbine_power_output(100)
        self.assertAlmostEqual(power_v100, 1)

    def test_calculate_windturbine_power_output(self):
        windspeed = 20
        n_turbines = 1
        turbine_data = {"slope_center": 1, "rated_power": 1, "steepness": 0.1}
        power_output = apply_features.calculate_windturbine_power_output(
            windspeed, n_turbines, turbine_data
        )
        expected_power_output = 0.8698915256370021
        self.assertAlmostEqual(power_output, expected_power_output)


if __name__ == "__main__":
    unittest.main()
