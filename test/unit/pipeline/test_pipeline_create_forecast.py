# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import pickle
import unittest
from datetime import datetime as dt
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData
from unittest.mock import patch

from openstef.model.serializer import MLflowSerializer
from openstef.pipeline import create_forecast, utils


class TestCreateForecastPipeline(BaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.pj = TestData.get_prediction_job(pid=307)
        self.serializer = MLflowSerializer(
            trained_models_folder="./test/unit/trained_models"
        )
        self.data = TestData.load("reference_sets/307-test-data.csv")
        self.train_input = TestData.load("reference_sets/307-train-data.csv")
        # Use MLflowSerializer to load a model
        self.model, _ = self.serializer.load_model(pid=307)

    def test_generate_forecast_datetime_range_single_null_values_target_column(self):
        """Test if correct forecast window is made with single range of nulls."""
        time_format = "%Y-%m-%d %H:%M:%S%z"
        forecast_start_expected = dt.strptime("2020-11-26 00:00:00+0000", time_format)
        forecast_end_expected = dt.strptime("2020-11-30 00:00:00+0000", time_format)
        forecast_data = self.data
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
        forecast_data = self.data
        forecast_data.loc["2020-11-26":"2020-12-01", forecast_data.columns[0]] = None
        forecast_data.loc["2020-11-23":"2020-11-24", forecast_data.columns[0]] = None

        forecast_start, forecast_end = utils.generate_forecast_datetime_range(
            forecast_data=forecast_data
        )

        self.assertEqual(forecast_start, forecast_start_expected)
        self.assertEqual(forecast_end, forecast_end_expected)

    def test_generate_forecast_datetime_range_not_null_values_target_column(self):
        """Test if error is raised when data has no nulls."""
        forecast_data = self.data
        forecast_data.loc["2020-11-26":"2020-12-01", forecast_data.columns[0]] = 1
        self.assertRaises(
            ValueError, utils.generate_forecast_datetime_range, forecast_data
        )

    def test_generate_forecast_datetime_range_only_null_values_target_column(self):
        """Test if correct forecast window is made when data only has nulls."""
        time_format = "%Y-%m-%d %H:%M:%S%z"
        forecast_start_expected = dt.strptime("2020-10-31 00:45:00+0000", time_format)
        forecast_end_expected = dt.strptime("2020-11-30 00:00:00+0000", time_format)
        forecast_data = self.data
        forecast_data.loc[:, forecast_data.columns[0]] = None

        forecast_start, forecast_end = utils.generate_forecast_datetime_range(
            forecast_data=forecast_data
        )

        self.assertEqual(forecast_start, forecast_start_expected)
        self.assertEqual(forecast_end, forecast_end_expected)

    @patch("mlflow.sklearn.load_model")
    @patch("openstef.validation.validation.is_data_sufficient")
    def test_create_forecast_pipeline_incomplete_inputdata(
        self, is_data_sufficient_mock, load_mock
    ):
        """Test if a fallback forecast is used when input is incomplete."""
        load_mock.return_value = self.model
        # Load mock value, forecast data, prediction job and model
        is_data_sufficient_mock.return_value = False

        forecast_data = self.data
        col_name = forecast_data.columns[0]
        forecast_data.loc["2020-11-28 00:00:00":"2020-12-01", col_name] = None

        model, modelspecs = self.serializer.load_model(self.pj["id"])
        if not hasattr(model, "standard_deviation"):  # Renamed the attribute
            model.standard_deviation = model.confidence_interval

        # Forecast
        forecast = create_forecast.create_forecast_pipeline_core(
            pj=self.pj, input_data=forecast_data, model=model
        )

        # Verify backtest was performed
        assert "substituted" in forecast.quality.values

    @patch("mlflow.sklearn.load_model")
    def test_create_forecast_pipeline_happy_flow_2_days(self, load_mock):
        """Test the happy flow of the forecast pipeline with a trained model."""
        load_mock.return_value = self.model
        # Load prediction job and forecast data
        forecast_data = self.data
        col_name = forecast_data.columns[0]
        forecast_data.loc["2020-11-28 00:00:00":"2020-12-01", col_name] = None

        # Load model
        model, modelspecs = self.serializer.load_model(self.pj["id"])
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

    @patch("mlflow.sklearn.load_model")
    def test_create_forecast_pipeline_happy_flow_4_days(self, load_mock):
        """Test the happy flow of the forecast pipeline with a trained model."""
        load_mock.return_value = self.model
        # Load prediction job and forecast data
        forecast_data = self.data
        col_name = forecast_data.columns[0]
        forecast_data.loc["2020-11-26 00:00:00":"2020-12-01", col_name] = None

        # Load model
        model, modelspecs = self.serializer.load_model(self.pj["id"])
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

    @patch("mlflow.sklearn.load_model")
    def test_create_forecast_pipeline_happy_flow_5_days(self, load_mock):
        """Test the happy flow of the forecast pipeline with a trained model."""
        load_mock.return_value = self.model
        # Load prediction job and forecast data
        forecast_data = self.data
        col_name = forecast_data.columns[0]
        forecast_data.loc["2020-11-25 00:00:00":"2020-12-01", col_name] = None

        # Load model
        model, modelspecs = self.serializer.load_model(self.pj["id"])
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
