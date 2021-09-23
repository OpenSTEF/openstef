# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

# import builtins
import unittest
from test.utils import TestData
from unittest.mock import MagicMock, Mock, patch
from datetime import datetime, timedelta

from openstf.tasks.utils.predictionjobloop import (
    PredictionJobLoop,
    PredictionJobException,
)
from openstf.exceptions import NoPredictedLoadError, NoRealisedLoadError

# import project modules
from openstf.tasks.utils.taskcontext import TaskContext

from test.utils import BaseTestCase

# define constants
PREDICTION_JOBS = TestData.get_prediction_jobs()
NUM_PREDICTION_JOBS = len(PREDICTION_JOBS)


@patch("openstf.tasks.utils.taskcontext.ConfigManager", MagicMock())
@patch("openstf.tasks.utils.taskcontext.DataBase")
class TestTaskContext(BaseTestCase):
    def test_task_context_database(self, db_mock):
        db_mock.return_value = 1234

        with TaskContext("unit_test", post_teams_on_exception=False) as context:
            self.assertEqual(context.database, 1234)

    def test_task_context_callbacks(self, db_mock):
        on_exception = Mock()
        on_successful = Mock()
        on_end = Mock()

        # Test with successful run
        with TaskContext(
            "unit_test_supposed_to_succeed",
            True,
            False,
            on_exception,
            on_successful,
            on_end,
        ):
            pass

        on_exception.assert_not_called()
        on_successful.assert_called_once()
        on_end.assert_called_once_with(True)

        # Test with failing run
        on_exception.reset_mock()
        on_successful.reset_mock()
        on_end.reset_mock()

        with TaskContext(
            "unit_test_supposed_to_fail",
            True,
            False,
            on_exception,
            on_successful,
            on_end,
        ):
            raise Exception("Forced exception")

        on_exception.assert_called_once()
        on_successful.assert_not_called()
        on_end.assert_called_with(False)

    def test_task_context_with_prediction_loop(self, db_mock):
        func_fail = Mock()
        func_fail.side_effect = ValueError("Forced error")

        with TaskContext("unit_test_supposed_to_fail", True, False) as context:
            PredictionJobLoop(context, prediction_jobs=PREDICTION_JOBS).map(func_fail)

    @patch("openstf.tasks.utils.taskcontext.post_teams")
    def test_task_context_teams_message(self, postteamsmock, dbmock):
        """Test to check that:
        if multiple exceptions are raised,
        pids they are nicely grouped per exception type."""
        # Specify which types of exceptions are raised
        func_fail = Mock()
        # the int/pid is arbitrary, unused currently.
        start_time = datetime.utcnow()
        end_time = datetime.utcnow() - timedelta(days=1)
        func_fail.side_effect = [
            None,
            NoPredictedLoadError(60, start_time, end_time),
            NoRealisedLoadError(307, start_time, end_time),
            None,
        ]

        # Specify test prediction jobs.
        # Required are the 'id' key and a second 'random' key.
        # The presence of a second key ensures that no additional pj_data is collected
        test_prediction_jobs = [
            TestData.get_prediction_job(pid=307),
            TestData.get_prediction_job(pid=60),
            TestData.get_prediction_job(pid=307),
            TestData.get_prediction_job(pid=60),
        ]

        with self.assertRaises(PredictionJobException):
            with TaskContext("test_with_teams_message", False, True) as context:
                PredictionJobLoop(
                    context, prediction_jobs=test_prediction_jobs, random_order=False
                ).map(func_fail)

        # Assert that specification of exception: [pids] is 'posted' to the postteamsmock
        self.assertListEqual(
            postteamsmock.call_args_list[0].args[0]["sections"][2]["facts"],
            [
                (
                    "Exceptions: pid(s)",
                    "No predicted load found:[60]\n\nNo realised load found:[307]\n",
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
