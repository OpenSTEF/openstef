# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from datetime import datetime, timedelta, UTC
from test.unit.utils.data import TestData
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd

from openstef.enums import PipelineType
from openstef.tasks.create_basecase_forecast import create_basecase_forecast_task

# Specify forecast mock.
# Make sure this has a datetime of at least NOW+48hours,
# since this is filtered in the task
FORECAST_MOCK = pd.DataFrame(
    index=pd.to_datetime([datetime.now(tz=UTC)]) + timedelta(days=3),
    data=dict(forecast=[10.0]),
)
FORECAST_NEAR_FUTURE_MOCK = pd.DataFrame(
    index=pd.to_datetime([datetime.now(tz=UTC)]) + timedelta(days=1),
    data=dict(forecast=[10.0]),
)


class TestCreateBasecaseForecastTask(TestCase):
    def setUp(self) -> None:
        self.pj = TestData.get_prediction_job(pid=307)

    @patch(
        "openstef.tasks.create_basecase_forecast.create_basecase_forecast_pipeline",
        MagicMock(return_value=FORECAST_MOCK),
    )
    def test_create_basecase_forecast_task_happy_flow(self):
        # Test happy flow of create forecast task
        context = MagicMock()
        create_basecase_forecast_task(self.pj, context)
        pd.testing.assert_frame_equal(context.mock_calls[3].args[0], FORECAST_MOCK)

    def test_create_basecase_forecast_task_skip_external(self):
        """Test happy flow of create forecast task."""
        # Arrange
        context = MagicMock()
        context.config.externally_posted_forecasts_pids = [307]

        # Act
        create_basecase_forecast_task(self.pj, context)

        # Assert
        self.assertEqual(
            context.mock_calls[0].args[0],
            "Skip this PredictionJob because its forecasts are posted by an external process.",
        )

    @patch("openstef.tasks.create_basecase_forecast.create_basecase_forecast_pipeline")
    def test_create_forecast_task_train_only(
        self, create_basecase_forecast_pipeline_mock
    ):
        """Test happy flow of create forecast task for train only pj."""
        context = MagicMock()
        pj = self.pj
        pj.pipelines_to_run = [PipelineType.TRAIN]
        create_basecase_forecast_task(pj, context)
        self.assertEqual(create_basecase_forecast_pipeline_mock.call_count, 0)

    @patch("openstef.tasks.create_basecase_forecast.create_basecase_forecast_pipeline")
    def test_create_forecast_task_forecast_only(
        self, create_basecase_forecast_pipeline_mock
    ):
        """Test happy flow of create forecast task for forecast only pj."""
        # Arrange
        context = MagicMock()
        create_basecase_forecast_pipeline_mock.return_value = FORECAST_MOCK
        pj = self.pj
        pj.pipelines_to_run = [PipelineType.FORECAST]

        # Act
        create_basecase_forecast_task(pj, context)

        # Assert
        self.assertEqual(create_basecase_forecast_pipeline_mock.call_count, 1)
        pd.testing.assert_frame_equal(context.mock_calls[3].args[0], FORECAST_MOCK)

    @patch(
        "openstef.tasks.create_basecase_forecast.create_basecase_forecast_pipeline",
        MagicMock(return_value=FORECAST_NEAR_FUTURE_MOCK),
    )
    def test_create_basecase_forecast_no_forecasts_first_48_hours(self):
        """If the basecase forecast pipeline returns forecasts within next 48 hours,
        those should not be written to database"""
        context = MagicMock()
        create_basecase_forecast_task(self.pj, context)

        # Mock call should be empty dataframe
        self.assertEqual(context.mock_calls[3].args[0].empty, True)
