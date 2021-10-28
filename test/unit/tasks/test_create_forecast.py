# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from unittest import TestCase
from unittest.mock import MagicMock, patch

import openstf.tasks.create_forecast as task
from openstf.tasks.create_forecast import create_forecast_task
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

    @patch(
        "openstf.tasks.utils.taskcontext.DataBase",
    )
    @patch("openstf.tasks.utils.taskcontext.ConfigManager")
    @patch("openstf.tasks.utils.taskcontext.post_teams")
    @patch("openstf.tasks.create_forecast.datetime")
    @patch("openstf.pipeline.utils.datetime")
    def test_create_forecast_task_with_context(
        self,
        utils_datetime_mock,
        datetime_mock,
        post_teamsmock,
        configmock_taskcontext,
        dbmock,
    ):
        dbmock().get_prediction_jobs.return_value = [
            self.pj,
            self.pj,
        ]
        testdata = TestData.load("reference_sets/307-test-data.csv")
        dbmock().get_model_input.return_value = testdata

        configmock_taskcontext.get_instance.return_value.paths.trained_models_folder = (
            "test/trained_models/"
        )
        configmock_taskcontext.get_instance.return_value.paths.webroot = "test_webroot"

        # Set 'NOW' to -10% points of testdata with known load
        datetime_mock.utcnow.return_value = testdata.iloc[-100, :].name
        utils_datetime_mock.utcnow.return_value = testdata.iloc[-100, :].name

        task.main()

        # assert if results forecast has been made
        written_forecast = dbmock().write_forecast.call_args.args[0]
        self.assertEqual(len(written_forecast), 101)
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
        # Assert MEA
        mae = (written_forecast.forecast - testdata.iloc[-101:, 0]).abs().mean()
        self.assertLess(mae, 38)
