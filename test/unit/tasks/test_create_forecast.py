# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from unittest import TestCase, skip
from unittest.mock import MagicMock, patch

from openstf.tasks.create_forecast import create_forecast_task
import openstf.tasks.create_forecast as task

from test.utils import TestData

FORECAST_MOCK = "forecast_mock"


class TestCreateForeCastTask(TestCase):
    def setUp(self) -> None:
        self.pj = TestData.get_prediction_job(pid=307)

    @patch(
        "openstf.tasks.create_forecast.create_forecast_pipeline",
        MagicMock(return_value=FORECAST_MOCK),
    )
    def test_create_forecast_task_happy_flow(self):
        # Test happy flow of create forecast task
        context = MagicMock()
        create_forecast_task(self.pj, context)
        self.assertEqual(context.mock_calls[1].args[0], FORECAST_MOCK)


    @skip('Didnt manage to get the mocking working...')
    @patch(
        "openstf.tasks.utils.taskcontext.DataBase",
    )
    @patch("openstf.tasks.utils.taskcontext.ConfigManager")
    @patch("openstf.tasks.utils.utils.ConfigManager")
    @patch("openstf.tasks.utils.taskcontext.post_teams")
    def test_create_forecast_task_with_context(self, post_teamsmock, configmock_utils, configmock_taskcontext, dbmock):
        dbmock().get_prediction_jobs.return_value = [TestData.get_prediction_job(pid=307),
                                                     TestData.get_prediction_job(pid=307)]

        configmock = MagicMock()
        configmock.paths.trained_models_folder = 'test/trained_models/'

        configmock_taskcontext = MagicMock()
        configmock_taskcontext.get_instance.return_value = configmock

        configmock_utils = MagicMock()
        configmock_utils().get_instance.return_value.paths.webroot = 'test_webroot'
        configmock_utils().get_instance.return_value.paths.trained_models_folder = 'test/trained_models/'

        task.main()