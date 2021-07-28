# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest

import pandas as pd

from openstf.model.standard_deviation_generator import StandardDeviationGenerator


class MockModel:
    def predict(self, *args):

        # Prepare mock_forecast
        mock_forecast = pd.DataFrame(
            {
                "forecast": [1, 2, 3, 4, 1, 3, 3, 7],
                "horizon": [47.0, 47.0, 47.0, 47.0, 24.0, 24.0, 24.0, 24.0],
            }
        )
        mock_forecast = mock_forecast.set_index(
            pd.to_datetime(
                [
                    "2018-01-01 00:00:00",
                    "2018-01-01 01:00:00",
                    "2018-01-01 02:00:00",
                    "2018-01-01 03:00:00",
                    "2018-01-01 00:00:00",
                    "2018-01-01 01:00:00",
                    "2018-01-01 02:00:00",
                    "2018-01-01 03:00:00",
                ]
            )
        )

        # Return mock forecast
        return mock_forecast["forecast"]


class TestStandardDeviationGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.model = MockModel()

        # Prepare mock validation data
        mock_validation_data = pd.DataFrame(
            {
                "load": [4, 2, 5, 2, 4, 2, 5, 2],
                "feature_1": [4, 2, 5, 2, 5, 4, 8, 2],
                "feature_2": [4, 2, 5, 2, 5, 4, 8, 2],
                "horizon": [47.0, 47.0, 47.0, 47.0, 24.0, 24.0, 24.0, 24.0],
            }
        )

        mock_validation_data = mock_validation_data.set_index(
            pd.to_datetime(
                [
                    "2018-01-01 00:00:00",
                    "2018-01-01 01:00:00",
                    "2018-01-01 02:00:00",
                    "2018-01-01 03:00:00",
                    "2018-01-01 00:00:00",
                    "2018-01-01 01:00:00",
                    "2018-01-01 02:00:00",
                    "2018-01-01 03:00:00",
                ]
            )
        )

        self.mock_validation_data = mock_validation_data

    def test_generate_standard_deviation_data_happy_flow(self):
        # Test happy flow

        # Generate reference dataframe of expected output, this data has been checked.
        ref_df = pd.DataFrame(
            {
                "stdev": [0, 0.0, 0.0, 0.0],
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
