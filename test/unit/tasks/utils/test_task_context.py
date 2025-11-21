# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <openstef@lfenergy.org> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

# import builtins
import unittest
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData
from unittest.mock import MagicMock, Mock, patch

from openstef.exceptions import NoPredictedLoadError, NoRealisedLoadError
from openstef.tasks.utils.predictionjobloop import (
    PredictionJobException,
    PredictionJobLoop,
)

# import project modules
from openstef.tasks.utils.taskcontext import TaskContext

# define constants
PREDICTION_JOBS = TestData.get_prediction_jobs()
NUM_PREDICTION_JOBS = len(PREDICTION_JOBS)


class TestTaskContext(BaseTestCase):
    def test_task_context_database(self):
        dbmock = MagicMock()
        config_mock = MagicMock()

        with TaskContext(
            "unit_test",
            config=config_mock,
            database=dbmock,
            post_teams_on_exception=False,
        ) as context:
            self.assertEqual(context.database, dbmock)

    def test_task_context_callbacks(self):
        dbmock = MagicMock()
        config_mock = MagicMock()
        on_exception = Mock()
        on_successful = Mock()
        on_end = Mock()

        # Test with successful run
        with TaskContext(
            "unit_test_supposed_to_succeed",
            dbmock,
            config_mock,
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
            dbmock,
            config_mock,
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

    def test_task_context_with_prediction_loop(self):
        dbmock = MagicMock()
        config_mock = MagicMock()
        func_fail = Mock()
        func_fail.side_effect = ValueError("Forced error")

        with TaskContext(
            "unit_test_supposed_to_fail", dbmock, config_mock, True, False
        ) as context:
            PredictionJobLoop(context, prediction_jobs=PREDICTION_JOBS).map(func_fail)

    @patch("openstef.tasks.utils.taskcontext.post_teams")
    def test_task_context_teams_message(self, postteamsmock):
        """Test to check that:
        if multiple exceptions are raised,
        pids they are nicely grouped per exception type."""

        dbmock = MagicMock()
        config_mock = MagicMock()

        # Specify which types of exceptions are raised
        func_fail = Mock()
        # the int/pid is arbitrary, unused currently.
        func_fail.side_effect = [
            None,
            NoPredictedLoadError(60),
            NoRealisedLoadError(307),
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
            with TaskContext(
                "test_with_teams_message", dbmock, config_mock, False, True
            ) as context:
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
