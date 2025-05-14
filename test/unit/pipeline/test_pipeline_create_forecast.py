# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from datetime import datetime as dt
from pathlib import Path

import pandas as pd
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData
from unittest.mock import MagicMock, patch

from mlflow.exceptions import MlflowException

from openstef.model.serializer import MLflowSerializer
from openstef.pipeline import create_forecast, utils


class TestCreateForecastPipeline(BaseTestCase):
    @patch("openstef.model.serializer.MLflowSerializer._get_model_uri")
    def setUp(self, _get_model_uri_mock) -> None:
        super().setUp()
        self.pj = TestData.get_prediction_job(pid=307)
        self.serializer = MLflowSerializer(
            mlflow_tracking_uri="./test/unit/trained_models/mlruns"
        )
        self.data = TestData.load("reference_sets/307-test-data.csv")
        self.train_input = TestData.load("reference_sets/307-train-data.csv")

        # mock model location
        # Determine absolute location where already stored model is, based on relative path.
        # This is needed so the model stored in the repo can be found when running remote
        rel_path = "test/unit/trained_models/mlruns/893156335105023143/2ca1d126e8724852b303b256e64a6c4f/artifacts/model"
        _get_model_uri_mock.return_value = Path(rel_path).absolute().as_uri()

        # Use MLflowSerializer to load a model
        # Note that this model was trained using xgboost v1.6.1
        # in time, this should be replaced by a model trained by a newer version, so temporary fixes
        # in loading of the model (serializer.py) can be removed.
        self.model, self.model_specs = self.serializer.load_model(experiment_name="307")

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

        model, model_specs = self.serializer.load_model(str(self.pj["id"]))
        if not hasattr(model, "standard_deviation"):  # Renamed the attribute
            model.standard_deviation = model.confidence_interval

        # Forecast
        forecast = create_forecast.create_forecast_pipeline_core(
            pj=self.pj, input_data=forecast_data, model=model, model_specs=model_specs
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
        model, model_specs = self.serializer.load_model(str(self.pj["id"]))
        model_specs.feature_names = forecast_data.columns[1:]

        if not hasattr(model, "standard_deviation"):  # Renamed the attribute
            model.standard_deviation = model.confidence_interval

        # Forecast
        forecast = create_forecast.create_forecast_pipeline_core(
            self.pj, forecast_data, model, model_specs
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
        model, model_specs = self.serializer.load_model(str(self.pj["id"]))
        model_specs.feature_names = forecast_data.columns[1:]

        if not hasattr(model, "standard_deviation"):  # Renamed the attribute
            model.standard_deviation = model.confidence_interval

        # Forecast
        forecast = create_forecast.create_forecast_pipeline_core(
            pj=self.pj, input_data=forecast_data, model=model, model_specs=model_specs
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
        model, model_specs = self.serializer.load_model(str(self.pj["id"]))
        model_specs.feature_names = forecast_data.columns[1:]

        if not hasattr(model, "standard_deviation"):  # Renamed the attribute
            model.standard_deviation = model.confidence_interval

        # Forecast
        forecast = create_forecast.create_forecast_pipeline_core(
            pj=self.pj, input_data=forecast_data, model=model, model_specs=model_specs
        )

        # Verify forecast works correctly
        self.assertEqual(len(forecast), 481)
        self.assertEqual(len(forecast.columns), 15)
        self.assertGreater(forecast.forecast.min(), -5)
        self.assertLess(forecast.forecast.max(), 85)

    @patch("openstef.model.serializer.MLflowSerializer.load_model")
    @patch("openstef.pipeline.create_forecast.create_forecast_pipeline_core")
    def test_create_forecast_pipeline_wrong_forecast_pid(
        self, create_forecast_pipeline_core_mock, load_mock
    ):
        """Test the forecast_pid parameter of th pj with a wrong pid."""

        def side_effects(experiment_name):
            if experiment_name == "307":
                return self.model, self.model_specs
            raise MlflowException("Wrong pid")

        load_mock.side_effect = side_effects
        create_forecast_pipeline_core_mock.return_value = MagicMock()

        self.pj.alternative_forecast_model_pid = "703"
        # Load prediction job and forecast data
        forecast_data = self.data
        col_name = forecast_data.columns[0]
        forecast_data.loc["2020-11-28 00:00:00":"2020-12-01", col_name] = None
        with self.assertRaises(MlflowException):
            create_forecast.create_forecast_pipeline(
                self.pj, forecast_data, "./test/unit/trained_models/mlruns"
            )

    @patch("openstef.model.serializer.MLflowSerializer.load_model")
    @patch("openstef.pipeline.create_forecast.create_forecast_pipeline_core")
    def test_create_forecast_pipeline_valid_forecast_pid(
        self, create_forecast_pipeline_core_mock, load_mock
    ):
        """Test the forecast_pid parameter of th pj with a valid pid."""

        def side_effects(experiment_name):
            if experiment_name == "3070":
                return self.model, self.model_specs
            raise MlflowException("Wrong pid")

        load_mock.side_effect = side_effects
        create_forecast_pipeline_core_mock.return_value = MagicMock()

        self.pj.alternative_forecast_model_pid = "3070"
        # Load prediction job and forecast data
        forecast_data = self.data
        col_name = forecast_data.columns[0]
        forecast_data.loc["2020-11-28 00:00:00":"2020-12-01", col_name] = None
        create_forecast.create_forecast_pipeline(
            self.pj, forecast_data, "./test/unit/trained_models/mlruns"
        )
        self.assertTrue(create_forecast_pipeline_core_mock.called)


class TestCreateForecastPipelineQuantileModel(BaseTestCase):
    @patch("openstef.model.serializer.MLflowSerializer._get_model_uri")
    def setUp(self, _get_model_uri_mock) -> None:
        super().setUp()
        self.pj = TestData.get_prediction_job(pid=308)
        self.serializer = MLflowSerializer(
            mlflow_tracking_uri="./test/unit/trained_models/mlruns"
        )
        self.data = TestData.load("reference_sets/307-test-data.csv")
        self.train_input = TestData.load("reference_sets/307-train-data.csv")

        # mock model location
        # Determine absolute location where already stored model is, based on relative path.
        # This is needed so the model stored in the repo can be found when running remote
        rel_path = "test/unit/trained_models/mlruns/245086199943716459/de3f1dae27204c4f94462a19f0ea0a99/artifacts/model"

        _get_model_uri_mock.return_value = Path(rel_path).absolute().as_uri()

        # Use MLflowSerializer to load a model
        # Note that this model was trained using ...
        self.model, self.model_specs = self.serializer.load_model(experiment_name="308")

    @patch("mlflow.sklearn.load_model")
    def test_create_forecast_pipeline_quantile_model(self, load_mock):
        """Test the happy flow of the forecast pipeline with a quantile model."""

        # TODO: Add quantile model to trained models folder
        load_mock.return_value = self.model

        # Load prediction job and forecast data
        forecast_data = self.data
        col_name = forecast_data.columns[0]
        forecast_data.loc["2020-11-28 00:00:00":"2020-12-01", col_name] = None

        # Load model
        model, model_specs = self.serializer.load_model(str(self.pj["id"]))
        model_specs.feature_names = forecast_data.columns[1:]

        if not hasattr(model, "standard_deviation"):  # Renamed the attribute
            model.standard_deviation = model.confidence_interval

        # Forecast
        forecast = create_forecast.create_forecast_pipeline_core(
            self.pj, forecast_data, model, model_specs
        )

        self.assertEqual(len(forecast), 193)
        self.assertEqual(len(forecast.columns), 15)
        self.assertGreater(forecast.forecast.min(), -5)
        self.assertLess(forecast.forecast.max(), 12)

        # Verify forecast column is equal to quantile_P50 column
        pd.testing.assert_series_equal(
            forecast.forecast, forecast.quantile_P50, check_names=False
        )
