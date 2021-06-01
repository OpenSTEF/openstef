# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from numpy import mod
from scipy.stats.stats import mode
from openstf.model.serializer import PersistentStorageSerializer
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from test.utils import BaseTestCase, TestData
from unittest.mock import MagicMock, patch

from openstf.pipeline import create_forecast_sklearn

NOW = datetime.now(timezone.utc)
PJ = TestData.get_prediction_job(pid=60)

forecast_input = TestData.load("reference_sets/307-test-data.csv")


class TestCreateForecastPipeline(BaseTestCase):
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
        ) = create_forecast_sklearn.generate_inputdata_datetime_range(
            t_behind_days=t_behind_days, t_ahead_days=t_ahead_days
        )
        self.assertEqual(datetime_start, datetime_start_expected)
        self.assertEqual(datetime_end, datetime_end_expected)

    @patch("openstf.pipeline.create_forecast_sklearn.datetime")
    def test_forecast_datetime_range(self, datetime_mock):
        datetime_mock.now.return_value = NOW
        # get current date and time UTC
        datetime_utc = NOW
        # Time range for time interval to be predicted
        forecast_start_expected = datetime_utc - timedelta(
            minutes=PJ["resolution_minutes"]
        )
        forecast_end_expected = datetime_utc + timedelta(minutes=PJ["horizon_minutes"])

        forecast_start, forecast_end = create_forecast_sklearn.generate_forecast_datetime_range(
            resolution_minutes=PJ["resolution_minutes"],
            horizon_minutes=PJ["horizon_minutes"],
        )
        self.assertEqual(forecast_start, forecast_start_expected)
        self.assertEqual(forecast_end, forecast_end_expected)

        # def test_get_model_input_demand(
        #         self,
        # ):
        #     create_forecast_sklearn._clear_input_data_cache()
        # input_data = create_forecast.get_model_input(
        #     pj=PJ, datetime_start=NOW, datetime_end=NOW
        # )
        # self.assertTrue(isinstance(input_data, MagicMock))

    @patch("openstf.validation.validation.find_nonzero_flatliner")
    @patch("openstf.preprocessing.preprocessing.replace_invalid_data")
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

        create_forecast_sklearn.pre_process_input_data(
            input_data=None, flatliner_threshold=None
        )

        # simply check if all mocks are called
        for mock_func in [nonzero_flatliner_mock, replace_invalid_data_mock]:
            self.assertEqual(mock_func.call_count, 1)

    @patch("openstf.validation.validation.is_data_sufficient")
    def test_create_forecast_pipeline_incomplete_inputdata(self, is_data_sufficient_mock):
        """Test if a fallback forecast is used when input is incomplete"""
        input_data = forecast_input
        is_data_sufficient_mock.return_value = False
        pj = TestData.get_prediction_job(pid=307)
        model = PersistentStorageSerializer(
            trained_models_folder=Path("./test/trained_models")
        ).load_model(pid=307)
        forecast = create_forecast_sklearn.create_forecast_pipeline_core(
            pj=pj, input_data=input_data, model=model
        )
        assert "substituted" in forecast.quality.values

    @patch(
        "openstf.pipeline.create_forecast_sklearn.MODEL_LOCATION", Path("./test/trained_models")
    )
    def test_create_forecast_pipeline_happy_flow(self):
        """Test the happy flow of the predict pipeline, using a previously trained model"""
        self.pj = TestData.get_prediction_job(pid=307)
        self.forecast_input = forecast_input
        self.pj["feature_names"] = self.forecast_input.columns[1:]
        model = PersistentStorageSerializer(
            trained_models_folder=Path("./test/trained_models")
        ).load_model(pid=307)
        forecast = create_forecast_sklearn.create_forecast_pipeline_core(
            pj=self.pj, input_data=self.forecast_input, model=model
        )
        self.assertEqual(len(forecast), 2878)
        self.assertEqual(len(forecast.columns), 15)
        self.assertGreater(forecast.forecast.min(), -5)
        self.assertLess(forecast.forecast.max(), 85)


if __name__ == "__main__":
    unittest.main()
