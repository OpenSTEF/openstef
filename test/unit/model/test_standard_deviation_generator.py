# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest

import pandas as pd

from openstf.model.standard_deviation_generator import StandardDeviationGenerator


class MockModel:
    def predict(self, mock_input_data):

        # Do something with the input
        print(mock_input_data.head())

        # Prepare mock_forecast
        mock_forecast = pd.DataFrame(
            {"forecast": [1, 2, 3, 4], "horizon": [47.0, 47.0, 47.0, 47.0]}
        )
        mock_forecast = mock_forecast.set_index(
            pd.date_range("2018-01-01", periods=4, freq="H")
        )
        mock_forecast_far = pd.DataFrame(
            {"forecast": [1, 3, 3, 7], "horizon": [24.0, 24.0, 24.0, 24.0]}
        )
        mock_forecast_far = mock_forecast_far.set_index(
            pd.date_range("2018-01-01", periods=4, freq="H")
        )
        mock_forecast = mock_forecast.append(mock_forecast_far)

        # Return mock forecast
        return mock_forecast["forecast"]


class TestStandardDeviationGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.model = MockModel()

        # Prepare mock validation data
        mock_validation_data = pd.DataFrame(
            {
                "load": [4, 2, 5, 2],
                "feature_1": [4, 2, 5, 2],
                "feature_2": [4, 2, 5, 2],
                "horizon": [47.0, 47.0, 47.0, 47.0],
            }
        )
        mock_validation_data = mock_validation_data.set_index(
            pd.date_range("2018-01-01", periods=4, freq="H")
        )
        mock_validation_data_far = pd.DataFrame(
            {
                "load": [4, 2, 5, 2],
                "feature_1": [5, 4, 8, 2],
                "feature_2": [5, 4, 8, 2],
                "horizon": [24.0, 24.0, 24.0, 24.0],
            }
        )
        mock_validation_data_far = mock_validation_data_far.set_index(
            pd.date_range("2018-01-01", periods=4, freq="H")
        )
        mock_validation_data = mock_validation_data.append(mock_validation_data_far)
        self.mock_validation_data = mock_validation_data

    def test_generate_horizons_happy_flow(self):
        # Test happy flow

        # Generate reference dataframe of expected output
        ref_df = pd.DataFrame(
            {
                "stdev": [0, 0.5, 0, 1.5],
                "hour": [0.0, 1.0, 2.0, 3.0],
                "horizon": [47.0, 47.0, 47.0, 47.0],
            }
        )

        # Carry out stdev generation
        model = StandardDeviationGenerator(
            self.mock_validation_data
        ).generate_standard_deviation_data(self.model)

        # Assert if result is as expected
        pd.testing.assert_frame_equal(model.standard_deviation.head(4), ref_df)

        # Check if all horizons are pressent
        self.assertEqual([47.0, 24.0], list(model.standard_deviation.horizon.unique()))
