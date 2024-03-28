# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from datetime import date, timedelta
from test.unit.utils.data import TestData
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd
from pandas import Timestamp

from openstef.tasks.create_components_forecast import create_components_forecast_task
from openstef.enums import ForecastColumnName

FORECAST_MOCK = pd.DataFrame(
    data={
        "forecast_wind_on_shore": {
            Timestamp("2022-09-08 10:00:00+0000", tz="UTC"): 0.0,
            Timestamp("2022-09-08 10:15:00+0000", tz="UTC"): 0.0,
        },
        "forecast_solar": {
            Timestamp("2022-09-08 10:00:00+0000", tz="UTC"): -2.259999,
            Timestamp("2022-09-08 10:15:00+0000", tz="UTC"): -2.753998,
        },
        "forecast_other": {
            Timestamp("2022-09-08 10:00:00+0000", tz="UTC"): -11.632748,
            Timestamp("2022-09-08 10:15:00+0000", tz="UTC"): -10.278155,
        },
        ForecastColumnName.PID: {
            Timestamp("2022-09-08 10:00:00+0000", tz="UTC"): 123,
            Timestamp("2022-09-08 10:15:00+0000", tz="UTC"): 123,
        },
        ForecastColumnName.CUSTOMER: {
            Timestamp("2022-09-08 10:00:00+0000", tz="UTC"): "ABC",
            Timestamp("2022-09-08 10:15:00+0000", tz="UTC"): "ABC",
        },
        ForecastColumnName.DESCRIPTION: {
            Timestamp("2022-09-08 10:00:00+0000", tz="UTC"): "ABC",
            Timestamp("2022-09-08 10:15:00+0000", tz="UTC"): "ABC",
        },
        "type": {
            Timestamp("2022-09-08 10:00:00+0000", tz="UTC"): "demand",
            Timestamp("2022-09-08 10:15:00+0000", tz="UTC"): "demand",
        },
        ForecastColumnName.GENERAL_TYPE: {
            Timestamp("2022-09-08 10:00:00+0000", tz="UTC"): "component",
            Timestamp(
                f'{(date.today()+timedelta(days=3)).strftime(format="%Y-%m-%d")} 10:15:00+0000',
                tz="UTC",
            ): "component",
        },
    }
)


class TestCreateComponentForecastTask(TestCase):
    def setUp(self) -> None:
        self.pj = TestData.get_prediction_job(pid=307)

    @patch(
        "openstef.tasks.create_components_forecast.create_components_forecast_pipeline",
        MagicMock(return_value=FORECAST_MOCK),
    )
    def test_create_basecase_forecast_task_happy_flow(self):
        # Test happy flow of create forecast task
        context = MagicMock()
        context.database.get_predicted_load.return_value = [1, 0]
        context.database.get_energy_split_coefs.return_value = [1, 0]

        create_components_forecast_task(self.pj, context)
        pd.testing.assert_frame_equal(context.mock_calls[2].args[0], FORECAST_MOCK)

    @patch(
        "openstef.tasks.create_components_forecast.create_components_forecast_pipeline"
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
        "openstef.tasks.create_components_forecast.create_components_forecast_pipeline"
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
