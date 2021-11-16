# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest
from datetime import datetime as dt
from pathlib import Path
from unittest.mock import patch

from openstf.model.serializer import PersistentStorageSerializer
from openstf.pipeline import create_forecast
from openstf.pipeline import utils
from test.utils import BaseTestCase, TestData


class TestCreateForecastPipeline(BaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.pj = TestData.get_prediction_job(pid=307)

    def test_generate_forecast_datetime_range_single_null_values_target_column(self):
        """Test if correct forecast window is made with single range of nulls."""
        time_format = "%Y-%m-%d %H:%M:%S%z"
        forecast_start_expected = dt.strptime("2020-11-26 00:00:00+0000", time_format)
        forecast_end_expected = dt.strptime("2020-11-30 00:00:00+0000", time_format)
        forecast_data = TestData.load("reference_sets/307-test-data.csv")
        forecast_data.loc["2020-11-26":"2020-12-01", forecast_data.columns[0]] = None

        forecast_start, forecast_end = utils.generate_forecast_datetime_range(
            forecast_data=forecast_data
        )

        self.assertEqual(forecast_start, forecast_start_expected)
        self.assertEqual(forecast_end, forecast_end_expected)

    def test_generate_forecast_datetime_range_multiple_null_values_target_column(self):
        """Test if correct forecast window is made with multiple ranges of nulls."""
        time_format = "%Y-%m-%d %H:%M:%S%z"
        forecast_start_expected = dt.strptime("2020-11-26 00:00:00+0000", time_format)
        forecast_end_expected = dt.strptime("2020-11-30 00:00:00+0000", time_format)
        forecast_data = TestData.load("reference_sets/307-test-data.csv")
        forecast_data.loc["2020-11-26":"2020-12-01", forecast_data.columns[0]] = None
        forecast_data.loc["2020-11-23":"2020-11-24", forecast_data.columns[0]] = None

        forecast_start, forecast_end = utils.generate_forecast_datetime_range(
            forecast_data=forecast_data
        )

        self.assertEqual(forecast_start, forecast_start_expected)
        self.assertEqual(forecast_end, forecast_end_expected)

    def test_generate_forecast_datetime_range_not_null_values_target_column(self):
        """Test if error is raised when data has no nulls."""
        forecast_data = TestData.load("reference_sets/307-test-data.csv")
        forecast_data.loc["2020-11-26":"2020-12-01", forecast_data.columns[0]] = 1
        self.assertRaises(
            ValueError, utils.generate_forecast_datetime_range, forecast_data
        )

    def test_generate_forecast_datetime_range_only_null_values_target_column(self):
        """Test if correct forecast window is made when data only has nulls."""
        time_format = "%Y-%m-%d %H:%M:%S%z"
        forecast_start_expected = dt.strptime("2020-10-31 00:45:00+0000", time_format)
        forecast_end_expected = dt.strptime("2020-11-30 00:00:00+0000", time_format)
        forecast_data = TestData.load("reference_sets/307-test-data.csv")
        forecast_data.loc[:, forecast_data.columns[0]] = None

        forecast_start, forecast_end = utils.generate_forecast_datetime_range(
            forecast_data=forecast_data
        )

        self.assertEqual(forecast_start, forecast_start_expected)
        self.assertEqual(forecast_end, forecast_end_expected)

    @patch("openstf.validation.validation.is_data_sufficient")
    def test_create_forecast_pipeline_incomplete_inputdata(
        self, is_data_sufficient_mock
    ):
        """Test if a fallback forecast is used when input is incomplete."""
        # Load mock value, forecast data, prediction job and model
        is_data_sufficient_mock.return_value = False

        forecast_data = TestData.load("reference_sets/307-test-data.csv")
        col_name = forecast_data.columns[0]
        forecast_data.loc["2020-11-28 00:00:00":"2020-12-01", col_name] = None

        model, modelspecs = PersistentStorageSerializer(
            trained_models_folder="./test/trained_models"
        ).load_model(self.pj["id"])
        if not hasattr(model, "standard_deviation"):  # Renamed the attribute
            model.standard_deviation = model.confidence_interval

        # Forecast
        forecast = create_forecast.create_forecast_pipeline_core(
            pj=self.pj, input_data=forecast_data, model=model
        )

        # Verify backtest was performed
        assert "substituted" in forecast.quality.values

    def test_create_forecast_pipeline_happy_flow_2_days(self):
        """Test the happy flow of the forecast pipeline with a trained model."""
        # Load prediction job and forecast data
        forecast_data = TestData.load("reference_sets/307-test-data.csv")
        col_name = forecast_data.columns[0]
        forecast_data.loc["2020-11-28 00:00:00":"2020-12-01", col_name] = None

        # Load model
        model, modelspecs = PersistentStorageSerializer(
            trained_models_folder="./test/trained_models"
        ).load_model(self.pj["id"])
        modelspecs.feature_names = forecast_data.columns[1:]

        if not hasattr(model, "standard_deviation"):  # Renamed the attribute
            model.standard_deviation = model.confidence_interval

        # Forecast
        forecast = create_forecast.create_forecast_pipeline_core(
            self.pj, forecast_data, model
        )

        # Verify forecast works correctly
        self.assertEqual(len(forecast), 193)
        self.assertEqual(len(forecast.columns), 15)
        self.assertGreater(forecast.forecast.min(), -5)
        self.assertLess(forecast.forecast.max(), 85)

    def test_create_forecast_pipeline_happy_flow_4_days(self):
        """Test the happy flow of the forecast pipeline with a trained model."""
        # Load prediction job and forecast data
        forecast_data = TestData.load("reference_sets/307-test-data.csv")
        col_name = forecast_data.columns[0]
        forecast_data.loc["2020-11-26 00:00:00":"2020-12-01", col_name] = None

        # Load model
        model, modelspecs = PersistentStorageSerializer(
            trained_models_folder="./test/trained_models"
        ).load_model(self.pj["id"])
        modelspecs.feature_names = forecast_data.columns[1:]

        if not hasattr(model, "standard_deviation"):  # Renamed the attribute
            model.standard_deviation = model.confidence_interval

        # Forecast
        forecast = create_forecast.create_forecast_pipeline_core(
            pj=self.pj, input_data=forecast_data, model=model
        )

        # Verify forecast works correctly
        self.assertEqual(len(forecast), 385)
        self.assertEqual(len(forecast.columns), 15)
        self.assertGreater(forecast.forecast.min(), -5)
        self.assertLess(forecast.forecast.max(), 85)

    def test_create_forecast_pipeline_happy_flow_5_days(self):
        """Test the happy flow of the forecast pipeline with a trained model."""
        # Load prediction job and forecast data
        forecast_data = TestData.load("reference_sets/307-test-data.csv")
        col_name = forecast_data.columns[0]
        forecast_data.loc["2020-11-25 00:00:00":"2020-12-01", col_name] = None

        # Load model
        model, modelspecs = PersistentStorageSerializer(
            trained_models_folder="./test/trained_models"
        ).load_model(self.pj["id"])
        modelspecs.feature_names = forecast_data.columns[1:]

        if not hasattr(model, "standard_deviation"):  # Renamed the attribute
            model.standard_deviation = model.confidence_interval

        # Forecast
        forecast = create_forecast.create_forecast_pipeline_core(
            pj=self.pj, input_data=forecast_data, model=model
        )

        # Verify forecast works correctly
        self.assertEqual(len(forecast), 481)
        self.assertEqual(len(forecast.columns), 15)
        self.assertGreater(forecast.forecast.min(), -5)
        self.assertLess(forecast.forecast.max(), 85)


if __name__ == "__main__":
    unittest.main()
