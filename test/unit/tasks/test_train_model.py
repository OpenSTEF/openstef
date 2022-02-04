# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from test.unit.utils.data import TestData
from unittest import TestCase
from unittest.mock import MagicMock, patch

from openstef.tasks.train_model import train_model_task


class TestTrainModelTask(TestCase):
    def setUp(self) -> None:
        self.pj, self.modelspecs = TestData.get_prediction_job_and_modelspecs(pid=307)

    @patch("openstef.tasks.train_model.train_model_pipeline")
    def test_create_train_model_task_happy_flow(self, train_model_pipeline_mock):
        # Test happy flow of create forecast task
        context = MagicMock()
        train_model_task(self.pj, context)

        self.assertEqual(train_model_pipeline_mock.call_count, 1)
        self.assertEqual(
            train_model_pipeline_mock.call_args_list[0][0][0]["id"], self.pj["id"]
        )
