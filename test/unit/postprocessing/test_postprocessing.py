# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest
from test.unit.utils.base import BaseTestCase

import pandas as pd

from openstef.enums import ForecastType
from openstef.postprocessing import postprocessing


class TestPostProcess(BaseTestCase):
    def test_post_process_wind_solar(self):
        forecast = pd.DataFrame({"forecast": [-10, -15, -33, 1, 1.3806e-23]})

        forecast_positive_removed = pd.DataFrame()

        forecast_positive_removed["forecast"] = postprocessing.post_process_wind_solar(
            forecast["forecast"], ForecastType.SOLAR
        )

        self.assertTrue((forecast_positive_removed["forecast"] <= 0.0).all())

    def test_normalize_and_convert_weather_data_for_splitting(self):
        # Create testing input
        weather_data_test = pd.DataFrame(
            {"windspeed_100m": [0, 25, 8.07, 0, 0], "radiation": [10, 16, 33, -1, -2]}
        )

        # Define test reference output
        weather_data_norm_ref = pd.DataFrame(
            {
                "radiation": [
                    -0.309406,
                    -0.495050,
                    -1.021040,
                    0.030941,
                    0.061881,
                ],
                "windpower": [
                    0,
                    -1,
                    -0.5,
                    0,
                    0,
                ],
            }
        )

        # Carry out test
        weather_data_norm_test = (
            postprocessing.normalize_and_convert_weather_data_for_splitting(
                weather_data_test
            )
        )
        print(weather_data_norm_test.columns)
        # Check column names are correctly set
        self.assertTrue(
            all(
                elem in ["windpower", "radiation"]
                for elem in list(weather_data_norm_test.columns)
            )
        )

        # Check dataframe content are equal
        self.assertDataframeEqual(
            weather_data_norm_test, weather_data_norm_ref, check_exact=False, atol=1e-2
        )

    def test_calculate_wind_power(self):
        # Arrange
        expected_measures = [
            0.8749573212034569,
            0.010507738070416014,
            0.00681214123510128,
        ]
        windspeed_100m = pd.DataFrame(
            index=pd.to_datetime(
                [
                    "2021-07-12 14:00:00+0200",
                    "2021-07-13 15:00:00+0200",
                    "2021-07-14 16:00:00+0200",
                ]
            ),
            data={"windspeed_100m": [11, 1.225, 0.5666666666666667]},
        )

        # Act
        model = postprocessing.calculate_wind_power(windspeed_100m)

        # Assert
        self.assertEqual(expected_measures, list(model.windenergy))

    def test_split_forecast_in_components(self):
        # Define test input
        weather_data_test = pd.DataFrame(
            {"windspeed_100m": [10, 15, 33, 1, 2], "radiation": [10, 16, 33, -1, -2]}
        )

        split_coefs_test = {"pv_ref": 0.5, "wind_ref": 0.25}

        forecast = pd.DataFrame({"forecast": [10, 15, 33, -1, -2]})
        forecast["pid"] = 123
        forecast["customer"] = "test_customer"
        forecast["description"] = "test_desription"
        forecast["type"] = "component"
        forecast["stdev"] = 0

        forecasts = postprocessing.split_forecast_in_components(
            forecast, weather_data_test, split_coefs_test
        )

        # Check column names are correctly set
        self.assertTrue(
            all(
                elem in list(forecasts.columns)
                for elem in [
                    "forecast_wind_on_shore",
                    "forecast_solar",
                    "forecast_other",
                ]
            )
        )

        # Check sign of the components is correct
        self.assertLessEqual(forecasts["forecast_solar"].sum(), 0)
        self.assertLessEqual(forecasts["forecast_wind_on_shore"].sum(), 0)

        # Check we have enough columns
        self.assertEqual(len(forecasts), 5)


class TestSortValuesByRow(unittest.TestCase):
    def test_sort_values_by_row_simple(self):
        # Test DataFrame with columns starting with 'P_'
        df = pd.DataFrame(
            {
                "quantile_P05": [1, 3, 5],
                "quantile_P95": [2, 0, 6],
            }
        )

        sorted_df = postprocessing.sort_quantiles(df)
        expected_result = pd.DataFrame(
            {
                "quantile_P05": [1, 0, 5],
                "quantile_P95": [2, 3, 6],
            }
        )
        pd.testing.assert_frame_equal(sorted_df, expected_result)

    def test_sort_values_by_row_complex(self):
        df = pd.DataFrame(
            {
                "quantile_P05": [1, 3, 5],
                "quantile_P95": [2, 0, 6],
                "quantile_P50": [1.5, 5, 1],
                "Other_column": [1, 0, 3],
                "forecast": [1.5, 5, 1],
            }
        )

        sorted_df = postprocessing.sort_quantiles(df)
        expected_result = pd.DataFrame(
            {
                "quantile_P05": [1, 0, 1],
                "quantile_P95": [2, 5, 6],
                "quantile_P50": [1.5, 3, 5],
                "Other_column": [1, 0, 3],
                "forecast": [1.5, 3, 5],
            }
        )

        sorted_df = postprocessing.sort_quantiles(df)
        pd.testing.assert_frame_equal(sorted_df, expected_result, check_dtype=False)

    def test_sort_values_no_rel_cols(self):
        df = pd.DataFrame(
            {
                "A": [1, 3, 5],
                "B": [2, 0, 6],
            }
        )

        sorted_df = postprocessing.sort_quantiles(df)

        pd.testing.assert_frame_equal(sorted_df, df)


if __name__ == "__main__":
    unittest.main()
