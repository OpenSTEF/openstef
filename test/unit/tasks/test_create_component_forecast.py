# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from test.unit.utils.data import TestData
from unittest import TestCase
from unittest.mock import MagicMock, patch

from openstf.tasks.create_components_forecast import create_components_forecast_task

FORECAST_MOCK = "forecast_mock"


class TestCreateComponentForecastTask(TestCase):
    def setUp(self) -> None:
        self.pj = TestData.get_prediction_job(pid=307)

    @patch(
        "openstf.tasks.create_components_forecast.create_components_forecast_pipeline",
        MagicMock(return_value=FORECAST_MOCK),
    )
    def test_create_basecase_forecast_task_happy_flow(self):
        # Test happy flow of create forecast task
        context = MagicMock()
        context.database.get_predicted_load.return_value = [1, 0]
        context.database.get_energy_split_coefs.return_value = [1, 0]

        create_components_forecast_task(self.pj, context)
        self.assertEqual(context.mock_calls[3].args[0], FORECAST_MOCK)

    @patch(
        "openstf.tasks.create_components_forecast.create_components_forecast_pipeline"
    )
    def test_create_basecase_forecast_task_no_input(self, pipeline_mock):
        # Test pipeline is not called when no input data is available
        context = MagicMock()
        context.database.get_predicted_load.return_value = []
        context.database.get_energy_split_coefs.return_value = [1, 0]
        pipeline_mock.return_value = FORECAST_MOCK
        create_components_forecast_task(self.pj, context)
        # When no data is available the pipeline should not be called
        self.assertFalse(pipeline_mock.called)

    @patch(
        "openstf.tasks.create_components_forecast.create_components_forecast_pipeline"
    )
    def test_create_basecase_forecast_task_no_coefs(self, pipeline_mock):
        # Test pipeline is not called when no coeficients are available
        context = MagicMock()
        context.database.get_predicted_load.return_value = [1, 0]
        context.database.get_energy_split_coefs.return_value = []
        pipeline_mock.return_value = FORECAST_MOCK
        create_components_forecast_task(self.pj, context)
        # When no coeficients are available the pipeline should not be called
        self.assertFalse(pipeline_mock.called)

    @patch(
        "openstf.tasks.create_components_forecast.create_components_forecast_pipeline"
    )
    def test_create_basecase_forecast_task_no_train_components(self, pipeline_mock):
        # Test pipeline is not called when the component foecasts are disabled in the prediciton job
        context = MagicMock()
        context.database.get_predicted_load.return_value = [1, 0]
        context.database.get_energy_split_coefs.return_value = [1.0]
        pipeline_mock.return_value = FORECAST_MOCK
        self.pj["train_components"] = 0
        create_components_forecast_task(self.pj, context)
        # When the component foecasts are disabled in the prediciton job the pipeline should not be called
        self.assertFalse(pipeline_mock.called)
