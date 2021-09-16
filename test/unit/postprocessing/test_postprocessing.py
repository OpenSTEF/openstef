# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest
from test.utils.base import BaseTestCase

import pandas as pd

from openstf.enums import ForecastType
from openstf.postprocessing import postprocessing


class TestPostProcess(BaseTestCase):
    def test_post_process_wind_solar(self):
        forecast_positive_sum = pd.DataFrame({"forecast": [10, 15, 33, -1, -2]})
        forecast_negative_sum = pd.DataFrame({"forecast": [-10, -15, -33, 1, 2]})

        forecast_negative_removed = pd.DataFrame()
        forecast_positive_removed = pd.DataFrame()

        forecast_negative_removed["forecast"] = postprocessing.post_process_wind_solar(
            forecast_positive_sum["forecast"], ForecastType.WIND
        )

        forecast_positive_removed["forecast"] = postprocessing.post_process_wind_solar(
            forecast_negative_sum["forecast"], ForecastType.SOLAR
        )

        self.assertTrue((forecast_negative_removed["forecast"] >= 0).all())
        self.assertTrue((forecast_positive_removed["forecast"] <= 0).all())

    def test_normalize_and_convert_weather_data_for_splitting(self):
        # Create testing input
        weather_data_test = pd.DataFrame(
            {"windspeed_100m": [10, 15, 33, 1, 2], "radiation": [10, 16, 33, -1, -2]}
        )

        # Define test reference output
        weather_data_norm_ref = pd.DataFrame(
            {
                "radiation": [-0.309406, -0.495050, -1.021040, 0.030941, 0.061881,],
                "windpower": [-0.303030, -0.454545, -1.000000, -0.030303, -0.060606,],
            }
        )

        # Carry out test
        weather_data_norm_test = postprocessing.normalize_and_convert_weather_data_for_splitting(
            weather_data_test
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
            weather_data_norm_test, weather_data_norm_ref, check_exact=False, rtol=1e-3
        )

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


if __name__ == "__main__":
    unittest.main()
