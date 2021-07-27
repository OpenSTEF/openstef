# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from unittest import TestCase
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime, timedelta

from openstf.tasks.create_basecase_forecast import create_basecase_forecast_task

from test.utils import TestData

# Specify forecast mock.
# Make sure this has a datetime of at least NOW+48hours,
# since this is filtered in the task
FORECAST_MOCK = pd.DataFrame(
    index=pd.to_datetime([datetime.utcnow()], utc=True) + timedelta(days=3),
    data=dict(forecast=[10.0]),
)
FORECAST_NEAR_FUTURE_MOCK = pd.DataFrame(
    index=pd.to_datetime([datetime.utcnow()], utc=True) + timedelta(days=1),
    data=dict(forecast=[10.0]),
)


class TestCreateBasecaseForecastTask(TestCase):
    def setUp(self) -> None:
        self.pj = TestData.get_prediction_job(pid=307)

    @patch(
        "openstf.tasks.create_basecase_forecast.create_basecase_forecast_pipeline",
        MagicMock(return_value=FORECAST_MOCK),
    )
    def test_create_basecase_forecast_task_happy_flow(self):
        # Test happy flow of create forecast task
        context = MagicMock()
        create_basecase_forecast_task(self.pj, context)
        pd.testing.assert_frame_equal(context.mock_calls[1].args[0], FORECAST_MOCK)

    @patch(
        "openstf.tasks.create_basecase_forecast.create_basecase_forecast_pipeline",
        MagicMock(return_value=FORECAST_NEAR_FUTURE_MOCK),
    )
    def test_create_basecase_forecast_no_forecasts_first_48_hours(self):
        """If the basecase forecast pipeline returns forecasts within next 48 hours,
        those should not be written to database"""
        context = MagicMock()
        create_basecase_forecast_task(self.pj, context)

        # Mock call should be empty dataframe
        self.assertEqual(context.mock_calls[1].args[0].empty, True)
