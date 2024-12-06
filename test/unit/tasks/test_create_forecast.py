# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import os
from pathlib import Path

from openstef.app_settings import AppSettings
from openstef.settings import Settings
from test.unit.utils.data import TestData
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pytest

import openstef.tasks.create_forecast as task
from openstef.enums import PipelineType
from openstef.exceptions import InputDataOngoingZeroFlatlinerError
from openstef.model.serializer import MLflowSerializer
from openstef.tasks.create_forecast import create_forecast_task

FORECAST_MOCK = "forecast_mock"


class TestCreateForecastTask(TestCase):
    @patch("openstef.model.serializer.MLflowSerializer._get_model_uri")
    def setUp(self, _get_model_uri_mock) -> None:
        self.pj, self.modelspecs = TestData.get_prediction_job_and_modelspecs(pid=307)
        self.serializer = MLflowSerializer(
            mlflow_tracking_uri="./test/unit/trained_models/mlruns"
        )

        # mock model location
        # Determine absolute location where already stored model is, based on relative path.
        # This is needed so the model stored in the repo can be found when running remote
        rel_path = "test/unit/trained_models/mlruns/893156335105023143/2ca1d126e8724852b303b256e64a6c4f/artifacts/model"
        _get_model_uri_mock.return_value = Path(rel_path).absolute().as_uri()
        # Use MLflowSerializer to load a model
        self.model, _ = self.serializer.load_model(experiment_name="307")

    def test_mocked_model_path(self):
        """This test explicitely tests if the model path is mocked correctly"""
        assert (
            "test/unit/trained_models/mlruns/893156335105023143/2ca1d126e8724852b303b256e64a6c4f/artifacts/model"
            in self.model.path
        )

    @patch(
        "openstef.tasks.create_forecast.create_forecast_pipeline",
        MagicMock(return_value=FORECAST_MOCK),
    )
    def test_create_forecast_task_happy_flow_1(self):
        """Test happy flow of create forecast task."""
        # Arrange
        context = MagicMock()
        context.config.externally_posted_forecasts_pids = None

        # Act
        create_forecast_task(self.pj, context)

        # Assert
        self.assertEqual(context.mock_calls[1][0], "database.write_forecast")
        self.assertEqual(context.mock_calls[1].args[0], FORECAST_MOCK)

    @patch(
        "openstef.tasks.create_forecast.create_forecast_pipeline",
        MagicMock(return_value=FORECAST_MOCK),
    )
    def test_create_forecast_task_happy_flow(self):
        """Test happy flow of create forecast task."""
        # Arrange
        context = MagicMock()
        context.config.externally_posted_forecasts_pids = [999]

        # Act
        create_forecast_task(self.pj, context)

        # Assert
        self.assertNotEqual(
            self.pj.id, context.config.externally_posted_forecasts_pids[0]
        )
        self.assertEqual(context.mock_calls[1][0], "database.write_forecast")
        self.assertEqual(context.mock_calls[1].args[0], FORECAST_MOCK)

    def test_create_forecast_task_skip_external(self):
        """Test that making a forecast is skipped for externally posted pids."""
        # Arrange
        context = MagicMock()
        context.config.externally_posted_forecasts_pids = [307]

        # Act
        create_forecast_task(self.pj, context)

        # Assert
        self.assertEqual(self.pj.id, context.config.externally_posted_forecasts_pids[0])
        self.assertEqual(
            context.mock_calls[0].args[0],
            "Skip this PredictionJob because its forecasts are posted by an external process.",
        )

    @patch(
        "openstef.tasks.create_forecast.create_forecast_pipeline",
        MagicMock(side_effect=InputDataOngoingZeroFlatlinerError()),
    )
    def test_create_forecast_known_zero_flatliner(self):
        """Test that making a forecast is skipped for known zero flatliners."""
        # Arrange
        context = MagicMock()
        context.config.externally_posted_forecasts_pids = None
        context.config.known_zero_flatliners = [307]

        # Act
        create_forecast_task(self.pj, context)

        # Assert
        self.assertEqual(self.pj.id, context.config.known_zero_flatliners[0])
        self.assertEqual(
            context.mock_calls[1].args[0],
            "No forecasts were made for this known zero flatliner prediction job. No forecasts need to be made either, since the fallback forecasts are sufficient.",
        )
        assert (
            not context.database.write_forecast.called
        ), "The `write_forecast` method should not have been called."

    @patch(
        "openstef.tasks.create_forecast.create_forecast_pipeline",
        MagicMock(side_effect=LookupError()),
    )
    def test_create_forecast_known_zero_flatliner_no_model(self):
        """Test that making a forecast is skipped for known zero flatliners for which no model has been trained yet."""
        # Arrange
        context = MagicMock()
        context.config.externally_posted_forecasts_pids = None
        context.config.known_zero_flatliners = [307]

        # Act
        create_forecast_task(self.pj, context)

        # Assert
        self.assertEqual(self.pj.id, context.config.known_zero_flatliners[0])
        self.assertEqual(
            context.mock_calls[1].args[0],
            "No forecasts were made for this known zero flatliner prediction job. No forecasts need to be made either, since the fallback forecasts are sufficient.",
        )
        assert (
            not context.database.write_forecast.called
        ), "The `write_forecast` method should not have been called."

    @patch(
        "openstef.tasks.create_forecast.create_forecast_pipeline",
        MagicMock(side_effect=InputDataOngoingZeroFlatlinerError()),
    )
    def test_create_forecast_unexpected_zero_flatliner(self):
        """Test that there is an informative error message for unexpected zero flatliners."""
        # Arrange
        context = MagicMock()
        context.config.externally_posted_forecasts_pids = None
        context.config.known_zero_flatliners = None

        # Act & Assert
        with pytest.raises(InputDataOngoingZeroFlatlinerError) as e:
            create_forecast_task(self.pj, context)

        assert (
            e.value.args[0]
            == 'All recent load measurements are zero. Check the load profile of this pid as well as related/neighbouring prediction jobs. Afterwards, consider adding this pid to the "known_zero_flatliners" app_setting and possibly removing other pids from the same app_setting.'
        )

    @patch(
        "openstef.tasks.create_forecast.create_forecast_pipeline",
        MagicMock(side_effect=LookupError("Model not found. First train a model!")),
    )
    @patch(
        "openstef.tasks.create_forecast.detect_ongoing_zero_flatliner",
        MagicMock(return_value=True),
    )
    def test_create_forecast_unexpected_zero_flatliner_lookuperror(self):
        """Test that the lookuperror is properly raised when the prediction job is an unexpected zero flatliner."""
        # Arrange
        context = MagicMock()
        context.config.externally_posted_forecasts_pids = None
        context.config.known_zero_flatliners = None

        # Act & Assert
        with pytest.raises(LookupError) as e:
            create_forecast_task(self.pj, context)

        assert (
            e.value.args[0]
            == 'Model not found. Consider checking for a zero flatliner and adding this pid to the "known_zero_flatliners" app_setting. For zero flatliners, no model can be trained.'
        )

    @patch(
        "openstef.tasks.create_forecast.create_forecast_pipeline",
        MagicMock(side_effect=LookupError("Model not found. First train a model!")),
    )
    @patch(
        "openstef.tasks.create_forecast.detect_ongoing_zero_flatliner",
        MagicMock(return_value=False),
    )
    def test_create_forecast_lookuperror(self):
        """Test that the lookuperror is properly raised when the prediction job is not a zero flatliner."""
        # Arrange
        context = MagicMock()
        context.config.externally_posted_forecasts_pids = None
        context.config.known_zero_flatliners = None

        # Act & Assert
        with pytest.raises(LookupError) as e:
            create_forecast_task(self.pj, context)

        assert e.value.args[0] == "Model not found. First train a model!"

    @patch("openstef.tasks.create_forecast.create_forecast_pipeline")
    def test_create_forecast_task_train_only(self, create_forecast_pipeline_mock):
        """Test happy flow of create forecast task for train only pj."""
        context = MagicMock()
        pj = self.pj
        pj.pipelines_to_run = [PipelineType.TRAIN]
        create_forecast_task(pj, context)
        self.assertEqual(create_forecast_pipeline_mock.call_count, 0)

    @patch("openstef.tasks.create_forecast.create_forecast_pipeline")
    def test_create_forecast_task_forecast_only(self, create_forecast_pipeline_mock):
        """Test happy flow of create forecast task for forecast only pj."""
        # Arrange
        context = MagicMock()
        create_forecast_pipeline_mock.return_value = FORECAST_MOCK
        pj = self.pj
        pj.pipelines_to_run = [PipelineType.FORECAST]

        # Act
        create_forecast_task(pj, context)

        # Assert
        self.assertEqual(create_forecast_pipeline_mock.call_count, 1)
        self.assertEqual(context.mock_calls[3].args[0], FORECAST_MOCK)

    @patch("mlflow.sklearn.load_model")
    @patch("openstef.model.serializer.MLflowSerializer")
    @patch("openstef.tasks.utils.taskcontext.post_teams")
    def test_create_forecast_task_with_context(
        self, post_teams_mock, serializer_mock, load_mock
    ):
        """Test create forecast task with context."""
        configmock_taskcontext = MagicMock()
        dbmock = MagicMock()

        load_mock.return_value = self.model
        dbmock.get_prediction_jobs.return_value = [
            self.pj,
            self.pj,
        ]

        dbmock.get_modelspecs.return_value = self.modelspecs

        forecast_data = TestData.load("reference_sets/307-test-data.csv")
        col_name = forecast_data.columns[0]
        forecast_data.loc["2020-11-28 00:00:00":"2020-12-01", col_name] = None
        dbmock.get_model_input.return_value = forecast_data

        configmock_taskcontext.return_value.paths_mlflow_tracking_uri = (
            "./test/unit/trained_models/mlruns"
        )
        configmock_taskcontext.return_value.paths_artifact_folder = (
            "./test/unit/trained_models"
        )

        task.main(config=configmock_taskcontext(), database=dbmock)

        # assert if results forecast has been made
        written_forecast = dbmock.write_forecast.call_args.args[0]
        self.assertEqual(len(written_forecast), 193)
        self.assertListEqual(
            list(written_forecast.columns),
            [
                "forecast",
                "tAhead",
                "stdev",
                "quantile_P05",
                "quantile_P10",
                "quantile_P30",
                "quantile_P50",
                "quantile_P70",
                "quantile_P90",
                "quantile_P95",
                "pid",
                "customer",
                "description",
                "type",
                "algtype",
            ],
        )

    @patch(
        "openstef.tasks.create_forecast.create_forecast_pipeline",
        MagicMock(return_value=FORECAST_MOCK),
    )
    def test_create_forecast_task_with_custom_weather_sources(self):
        """Test happy flow of create forecast task."""
        # Arrange
        weather_sources = ["weather_source_1", "weather_source_2"]

        context = MagicMock()
        context.config.externally_posted_forecasts_pids = None
        Settings.weather_sources = weather_sources

        # Act
        create_forecast_task(self.pj, context)

        # Assert
        self.assertEqual(context.mock_calls[1][0], "database.write_forecast")
        self.assertEqual(context.mock_calls[1].args[0], FORECAST_MOCK)
        self.assertEqual(
            context.database.get_model_input.call_args.kwargs["weather_sources"],
            weather_sources,
        )
