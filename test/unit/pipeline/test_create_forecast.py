# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest
from datetime import datetime, timedelta, timezone
from test.utils import BaseTestCase, TestData
from unittest import mock
from unittest.mock import MagicMock, patch

from openstf.pipeline import create_forecast

NOW = datetime.now(timezone.utc)
PJ = TestData.get_prediction_job(pid=60)


@mock.patch("openstf.pipeline.create_forecast.DataBase", MagicMock())
class TestCreateForecast(BaseTestCase):
    def test_generate_inputdata_datetime_range(self):
        t_behind_days = 14
        t_ahead_days = 3
        # get current date UTC
        date_today_utc = datetime.now(timezone.utc).date()
        # Time range for input data
        datetime_start_expected = date_today_utc - timedelta(days=t_behind_days)
        datetime_end_expected = date_today_utc + timedelta(days=t_ahead_days)

        (
            datetime_start,
            datetime_end,
        ) = create_forecast.generate_inputdata_datetime_range(
            t_behind_days=t_behind_days, t_ahead_days=t_ahead_days
        )
        self.assertEqual(datetime_start, datetime_start_expected)
        self.assertEqual(datetime_end, datetime_end_expected)

    @patch("openstf.pipeline.create_forecast.datetime")
    def test_forecast_datetime_range(self, datetime_mock):
        datetime_mock.now.return_value = NOW
        # get current date and time UTC
        datetime_utc = NOW
        # Time range for time interval to be predicted
        forecast_start_expected = datetime_utc - timedelta(
            minutes=PJ["resolution_minutes"]
        )
        forecast_end_expected = datetime_utc + timedelta(minutes=PJ["horizon_minutes"])

        forecast_start, forecast_end = create_forecast.generate_forecast_datetime_range(
            resolution_minutes=PJ["resolution_minutes"],
            horizon_minutes=PJ["horizon_minutes"],
        )
        self.assertEqual(forecast_start, forecast_start_expected)
        self.assertEqual(forecast_end, forecast_end_expected)

    def test_get_model_input_demand(
        self,
    ):
        create_forecast._clear_input_data_cache()
        input_data = create_forecast.get_model_input(
            pj=PJ, datetime_start=NOW, datetime_end=NOW
        )
        self.assertTrue(isinstance(input_data, MagicMock))

    @patch("openstf.pipeline.create_forecast.validation.find_nonzero_flatliner")
    @patch("openstf.pipeline.create_forecast.preprocessing.replace_invalid_data")
    def test_pre_process_input_data(
        self, replace_invalid_data_mock, nonzero_flatliner_mock
    ):
        suspicious_moments = True

        null_row = MagicMock()
        null_row.isnull.return_value = [True]
        processed_input_data_rows = [(0, null_row), (1, null_row)]
        processed_input_data = MagicMock()
        processed_input_data.iterrows.return_value = processed_input_data_rows

        nonzero_flatliner_mock.return_value = suspicious_moments
        replace_invalid_data_mock.return_value = processed_input_data

        create_forecast.pre_process_input_data(
            input_data=None, flatliner_threshold=None
        )

        # simply check if all mocks are called
        for mock_func in [nonzero_flatliner_mock, replace_invalid_data_mock]:
            self.assertEqual(mock_func.call_count, 1)

    @patch("openstf.pipeline.create_forecast.feature_engineering")
    def test_perform_feature_engineering(self, fe_mock):
        create_forecast.perform_feature_engineering(input_data=None, feature_names=None)

        for mock_func in [
            fe_mock.general.extract_minute_features,
            fe_mock.apply_features.apply_features,
            fe_mock.general.add_missing_feature_columns,
        ]:
            self.assertEqual(mock_func.call_count, 1)


if __name__ == "__main__":
    unittest.main()
