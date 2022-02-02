# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import pickle
from test.unit.utils.data import TestData
from unittest import TestCase
from unittest.mock import MagicMock, patch
from pathlib import Path

import openstef.tasks.create_forecast as task
from openstef.tasks.create_forecast import create_forecast_task
from openstef.model.serializer import MLflowSerializer

FORECAST_MOCK = "forecast_mock"


class TestCreateForeCastTask(TestCase):
    @patch("openstef.model.serializer._get_model_uri")
    def setUp(self, _get_model_uri_mock) -> None:
        self.pj, self.modelspecs = TestData.get_prediction_job_and_modelspecs(pid=307)
        self.serializer = MLflowSerializer(
            trained_models_folder="./test/unit/trained_models"
        )

        # mock model location
        # Determine absolute location where already stored model is, based on relative path.
        # This is needed so the model stored in the repo can be found when running remote
        rel_path = "test/unit/trained_models/mlruns/0/d7719d5d316d4416a947e4f7ea7e73a8/artifacts/model"
        _get_model_uri_mock.return_value = Path(rel_path).absolute().as_uri()
        # Use MLflowSerializer to load a model
        self.model, _ = self.serializer.load_model(pid=307)

    def test_mocked_model_path(self):
        """This test explicitely tests if the model path is mocked correctly"""
        assert (
            "/test/unit/trained_models/mlruns/0/d7719d5d316d4416a947e4f7ea7e73a8/artifacts/model"
            in self.model.path
        )

    @patch(
        "openstef.tasks.create_forecast.create_forecast_pipeline",
        MagicMock(return_value=FORECAST_MOCK),
    )
    def test_create_forecast_task_happy_flow(self):
        """Test happy flow of create forecast task."""
        context = MagicMock()
        create_forecast_task(self.pj, context)
        self.assertEqual(context.mock_calls[1].args[0], FORECAST_MOCK)

    @patch("mlflow.sklearn.load_model")
    @patch("openstef.model.serializer.MLflowSerializer")
    @patch("openstef.tasks.utils.taskcontext.DataBase")
    @patch("openstef.tasks.utils.taskcontext.ConfigManager")
    def test_create_forecast_task_with_context(
        self, configmock_taskcontext, dbmock, serializer_mock, load_mock
    ):
        """Test create forecast task with context."""
        load_mock.return_value = self.model
        dbmock().get_prediction_jobs.return_value = [
            self.pj,
            self.pj,
        ]

        dbmock().get_modelspecs.return_value = self.modelspecs

        forecast_data = TestData.load("reference_sets/307-test-data.csv")
        col_name = forecast_data.columns[0]
        forecast_data.loc["2020-11-28 00:00:00":"2020-12-01", col_name] = None
        dbmock().get_model_input.return_value = forecast_data

        configmock_taskcontext.get_instance.return_value.paths.trained_models_folder = (
            "./test/unit/trained_models/"
        )
        configmock_taskcontext.get_instance.return_value.paths.webroot = "test_webroot"

        task.main()

        # assert if results forecast has been made
        written_forecast = dbmock().write_forecast.call_args.args[0]
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
