# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import copy
from datetime import UTC, datetime
from test.unit.utils.data import TestData
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from openstef.enums import PipelineType
from openstef.exceptions import InputDataOngoingFlatlinerError
from openstef.tasks.train_model import TRAINING_PERIOD_DAYS
from openstef.tasks.train_model import main as task_main
from openstef.tasks.train_model import train_model_task


class TestTrainModelTask(TestCase):
    def setUp(self) -> None:
        self.pj, self.modelspecs = TestData.get_prediction_job_and_modelspecs(pid=307)
        self.train_data = TestData.load("reference_sets/307-train-data.csv")

        self.dbmock = MagicMock()
        self.dbmock.get_model_input.return_value = self.train_data

        self.context = MagicMock()
        self.context.database = self.dbmock
        self.context.config.paths_mlflow_tracking_uri = (
            "./test/unit/trained_models/mlruns"
        )
        self.context.config.paths_artifact_folder = "./test/unit/trained_models"
        self.context.paths.webroot = "test_webroot"

    @patch("openstef.tasks.train_model.train_model_pipeline")
    def test_create_train_model_task_happy_flow(self, train_model_pipeline_mock):
        # Test happy flow of create forecast task

        # Arrange
        context = MagicMock()
        test_data = TestData.load("reference_sets/307-train-data.csv")
        context.database.get_model_input.return_value = test_data

        # Act
        train_model_task(self.pj, context)

        # Assert
        self.assertEqual(train_model_pipeline_mock.call_count, 1)
        self.assertEqual(
            train_model_pipeline_mock.call_args_list[0][0][0]["id"], self.pj["id"]
        )
        self.assertEqual(
            len(train_model_pipeline_mock.call_args_list[0][0][1]),
            len(test_data),
        )

    @patch("openstef.tasks.train_model.train_model_pipeline")
    def test_create_train_model_task_data_balancing(self, train_model_pipeline_mock):
        # Test happy flow of create forecast task

        # Arrange
        context = MagicMock()
        test_data = TestData.load("reference_sets/307-train-data.csv")

        def get_model_input(*args, datetime_start=None, datetime_end=None, **kwargs):
            # Shift test data to datetime_start and datetime_end
            result = test_data.copy()
            result.index = pd.date_range(
                datetime_start, freq="15T", periods=len(result)
            )
            return result[datetime_start:datetime_end]

        context.database.get_model_input = get_model_input
        pj_with_balancing = self.pj.copy(update={"data_balancing_ratio": 0.5})

        datetime_end = datetime.now(tz=UTC)
        # +2 because pandas slicing is inclusive
        expected_data_points = TRAINING_PERIOD_DAYS * 24 * 4 + 2

        # Act
        train_model_task(pj_with_balancing, context, datetime_end=datetime_end)

        # Assert
        self.assertEqual(train_model_pipeline_mock.call_count, 1)
        self.assertEqual(
            train_model_pipeline_mock.call_args_list[0][0][0]["id"], self.pj["id"]
        )
        input_data = train_model_pipeline_mock.call_args_list[0][0][1]
        self.assertIn(datetime_end, input_data.index)
        self.assertIn(datetime_end - pd.Timedelta(days=365), input_data.index)
        self.assertEqual(len(input_data), expected_data_points)

    @patch(
        "openstef.tasks.train_model.train_model_pipeline",
        MagicMock(side_effect=InputDataOngoingFlatlinerError()),
    )
    def test_train_model_known_zero_flatliner(self):
        """Test that training a model is skipped for known zero flatliners."""
        # Arrange
        context = MagicMock()
        context.config.externally_posted_forecasts_pids = None
        context.config.known_zero_flatliners = [307]

        # Act
        train_model_task(self.pj, context)

        # Assert
        self.assertEqual(self.pj.id, context.config.known_zero_flatliners[0])
        self.assertEqual(
            context.mock_calls[-1].args[0],
            "No model was trained for this known flatliner. No model needs to be trained either, since the fallback forecasts are sufficient.",
        )

    @patch(
        "openstef.tasks.train_model.train_model_pipeline",
        MagicMock(side_effect=InputDataOngoingFlatlinerError()),
    )
    def test_train_model_unexpected_zero_flatliner(self):
        """Test that there is an informative error message for unexpected zero flatliners."""
        # Arrange
        context = MagicMock()
        context.config.externally_posted_forecasts_pids = None
        context.config.known_zero_flatliners = None

        # Act & Assert
        with pytest.raises(InputDataOngoingFlatlinerError) as e:
            train_model_task(self.pj, context)
        self.assertEqual(
            e.value.args[0],
            'All recent load measurements are constant. Check the load profile of this pid as well as related/neighbouring prediction jobs. Afterwards, consider adding this pid to the "known_zero_flatliners" app_setting and possibly removing other pids from the same app_setting.',
        )

    @patch("openstef.model.serializer.MLflowSerializer.save_model")
    @patch("openstef.pipeline.train_model.MLflowSerializer")
    @patch("openstef.tasks.utils.taskcontext.post_teams")
    def test_pipeline_train_model_with_context(
        self, post_teams_mock, serializer_mock, save_mock
    ):
        """Test create forecast task with context."""
        serializer_mock.return_value.load_model.side_effect = FileNotFoundError
        train_model_task(pj=self.pj, context=self.context)

    @patch("openstef.model.serializer.MLflowSerializer.save_model")
    @patch("openstef.pipeline.train_model.MLflowSerializer")
    @patch("openstef.tasks.utils.taskcontext.post_teams")
    def test_pipeline_train_model_with_save_train_forecasts(
        self, post_teams_mock, serializer_mock, save_mock
    ):
        """Test create forecast task with context."""

        pj = copy.deepcopy(self.pj)
        pj.save_train_forecasts = True
        serializer_mock.return_value.load_model.side_effect = FileNotFoundError
        train_model_task(pj=pj, context=self.context)

        datasets = self.dbmock.write_train_forecasts.call_args.args[1]
        self.assertIsNotNone(datasets)
        self.assertEqual(len(datasets), 3)
        for ds in datasets:
            self.assertIsInstance(ds, pd.DataFrame)
            self.assertIn("forecast", ds.columns)

    @patch("openstef.tasks.train_model.train_model_pipeline")
    def test_pipeline_train_model_with_save_train_forecasts_and_errors(
        self, pipeline_mock
    ):
        """Test create forecast task with context."""

        pipeline_mock.return_value = None
        pj = copy.deepcopy(self.pj)
        pj.save_train_forecasts = True

        # An error is raised when nothing is returned by the pipeline while save_train_forecasts is activated
        with self.assertRaises(RuntimeError):
            train_model_task(pj=pj, context=self.context)

        # An error is raised if the database does not have the write_train_forecasts method
        # and save_train_forecasts is activated
        context = copy.deepcopy(self.context)
        del context.database.write_train_forecasts
        pipeline_mock.return_value = MagicMock()  # Not None
        with self.assertRaises(RuntimeError):
            train_model_task(pj=pj, context=context)

    @patch("openstef.tasks.train_model.PredictionJobLoop")
    def test_main_task(self, pjloop_mock):
        """Test create forecast task with context."""

        not_none_object = 1

        with self.assertRaises(RuntimeError):
            task_main(None, None, not_none_object)

        with self.assertRaises(RuntimeError):
            task_main(None, not_none_object, None)

        task_main(None, not_none_object, not_none_object)

    @patch("openstef.tasks.train_model.train_model_pipeline")
    def test_create_train_model_task_train_only(self, train_model_pipeline_mock):
        # Test happy flow of create forecast task for train only pj
        context = MagicMock()
        pj = self.pj
        pj.pipelines_to_run = [PipelineType.TRAIN]

        train_model_task(pj, context)

        self.assertEqual(train_model_pipeline_mock.call_count, 1)
        self.assertEqual(
            train_model_pipeline_mock.call_args_list[0][0][0]["id"], pj["id"]
        )

    @patch("openstef.tasks.train_model.train_model_pipeline")
    def test_create_train_model_task_forecast_only(self, train_model_pipeline_mock):
        # Test happy flow of create forecast task for forecast only pj
        context = MagicMock()
        pj = self.pj
        pj.pipelines_to_run = [PipelineType.FORECAST]

        train_model_task(pj, context)

        self.assertEqual(train_model_pipeline_mock.call_count, 0)
