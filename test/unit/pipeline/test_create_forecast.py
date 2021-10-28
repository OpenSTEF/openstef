# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from openstf.model.serializer import PersistentStorageSerializer
from openstf.pipeline import create_forecast
from openstf.pipeline import utils
from test.utils import BaseTestCase, TestData


class TestCreateForecastPipeline(BaseTestCase):
    def test_generate_forecast_datetime_range(self):
        """Test if correct forecast window is made based on forecast data."""
        time_format = "%Y-%m-%d %H:%M:%S"
        forecast_start_expected = datetime.strptime("2020-11-26 00:00:00", time_format)
        forecast_end_expected = datetime.strptime("2020-11-30 00:00:00", time_format)

        forecast_data = TestData.load("forecast_data_missing_4_days.csv")
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
        pj = TestData.get_prediction_job(pid=307)
        pj["model_type_group"] = "default"
        model = PersistentStorageSerializer(
            trained_models_folder=Path("./test/trained_models")
        ).load_model(pid=307)
        if not hasattr(model, "standard_deviation"):  # Renamed the attribute
            model.standard_deviation = model.confidence_interval

        # Forecast
        forecast = create_forecast.create_forecast_pipeline_core(
            pj=pj, input_data=forecast_data, model=model
        )

        # Verify backtest was performed
        assert "substituted" in forecast.quality.values

    def test_create_forecast_pipeline_happy_flow_2_days(self):
        """Test the happy flow of the forecast pipeline with a trained model."""
        # Load prediction job and forecast data
        pj = TestData.get_prediction_job(pid=307)
        pj["model_type_group"] = "default"
        forecast_data = TestData.load("reference_sets/307-test-data.csv")
        col_name = forecast_data.columns[0]
        forecast_data.loc["2020-11-28 00:00:00":"2020-12-01", col_name] = None
        pj["feature_names"] = forecast_data.columns[1:]

        # Load model
        model = PersistentStorageSerializer(
            trained_models_folder=Path("./test/trained_models")
        ).load_model(pid=307)

        if not hasattr(model, "standard_deviation"):  # Renamed the attribute
            model.standard_deviation = model.confidence_interval

        # Forecast
        forecast = create_forecast.create_forecast_pipeline_core(
            pj=pj, input_data=forecast_data, model=model
        )

        # Verify forecast works correctly
        self.assertEqual(len(forecast), 193)
        self.assertEqual(len(forecast.columns), 15)
        self.assertGreater(forecast.forecast.min(), -5)
        self.assertLess(forecast.forecast.max(), 85)

    def test_create_forecast_pipeline_happy_flow_4_days(self):
        """Test the happy flow of the forecast pipeline with a trained model."""
        # Load prediction job and forecast data
        pj = TestData.get_prediction_job(pid=307)
        pj["model_type_group"] = "default"
        forecast_data = TestData.load("reference_sets/307-test-data.csv")
        col_name = forecast_data.columns[0]
        forecast_data.loc["2020-11-26 00:00:00":"2020-12-01", col_name] = None
        pj["feature_names"] = forecast_data.columns[1:]

        # Load model
        model = PersistentStorageSerializer(
            trained_models_folder=Path("./test/trained_models")
        ).load_model(pid=307)

        if not hasattr(model, "standard_deviation"):  # Renamed the attribute
            model.standard_deviation = model.confidence_interval

        # Forecast
        forecast = create_forecast.create_forecast_pipeline_core(
            pj=pj, input_data=forecast_data, model=model
        )

        # Verify forecast works correctly
        self.assertEqual(len(forecast), 385)
        self.assertEqual(len(forecast.columns), 15)
        self.assertGreater(forecast.forecast.min(), -5)
        self.assertLess(forecast.forecast.max(), 85)

    def test_create_forecast_pipeline_happy_flow_5_days(self):
        """Test the happy flow of the forecast pipeline with a trained model."""
        # Load prediction job and forecast data
        pj = TestData.get_prediction_job(pid=307)
        pj["model_type_group"] = "default"
        forecast_data = TestData.load("reference_sets/307-test-data.csv")
        col_name = forecast_data.columns[0]
        forecast_data.loc["2020-11-25 00:00:00":"2020-12-01", col_name] = None
        pj["feature_names"] = forecast_data.columns[1:]

        # Load model
        model = PersistentStorageSerializer(
            trained_models_folder=Path("./test/trained_models")
        ).load_model(pid=307)

        if not hasattr(model, "standard_deviation"):  # Renamed the attribute
            model.standard_deviation = model.confidence_interval

        # Forecast
        forecast = create_forecast.create_forecast_pipeline_core(
            pj=pj, input_data=forecast_data, model=model
        )

        # Verify forecast works correctly
        self.assertEqual(len(forecast), 481)
        self.assertEqual(len(forecast.columns), 15)
        self.assertGreater(forecast.forecast.min(), -5)
        self.assertLess(forecast.forecast.max(), 85)


if __name__ == "__main__":
    unittest.main()
