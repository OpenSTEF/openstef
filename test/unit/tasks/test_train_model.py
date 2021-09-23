# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from unittest import TestCase
from unittest.mock import MagicMock, patch

from openstf.tasks.train_model import train_model_task
from openstf.pipeline.train_model import MAXIMUM_MODEL_AGE
from openstf_dbc.services.prediction_job import PredictionJobDataClass

from test.utils import TestData

FORECAST_MOCK = "forecast_mock"


class TestTrainModelTask(TestCase):
    def setUp(self) -> None:
        self.pj = TestData.get_prediction_job(pid=307)

    @patch("openstf.tasks.train_model.train_model_pipeline")
    def test_create_train_model_task_happy_flow(self, train_model_pipeline_mock):
        # Test happy flow of create forecast task
        context = MagicMock()
        train_model_task(self.pj, context)

        self.assertEqual(train_model_pipeline_mock.call_count, 1)
        self.assertEqual(
            train_model_pipeline_mock.call_args_list[0][0][0]["id"], self.pj["id"]
        )

    @patch(
        "openstf.tasks.train_model.get_model_age", return_value=MAXIMUM_MODEL_AGE - 1
    )
    @patch("openstf.tasks.train_model.train_model_pipeline")
    def test_train_model_task_recent_old_model(
        self, train_model_pipeline_mock, get_model_age_mock
    ):
        """If an old model exists which is recent, abort the training immediately."""
        # Test happy flow of create forecast task
        context = MagicMock()
        train_model_task(self.pj, context)
        # Assert that the context is not called. This would be the case if input data was retrieved
        context.database.get_model_input.assert_not_called()

    @patch(
        "openstf.tasks.train_model.get_model_age", return_value=MAXIMUM_MODEL_AGE + 1
    )
    @patch("openstf.tasks.train_model.train_model_pipeline")
    def test_train_model_task_no_recent_old_model(
        self, train_model_pipeline_mock, get_model_age_mock
    ):
        """If an old model exists but is not recent, training should continue."""
        # Test happy flow of create forecast task
        context = MagicMock()
        train_model_task(self.pj, context)
        # Assert that the context is called. This would be the case if input data was retrieved
        context.database.get_model_input.assert_called()
