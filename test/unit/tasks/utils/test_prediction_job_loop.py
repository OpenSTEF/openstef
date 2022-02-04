# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

# import builtins
import unittest
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData
from unittest.mock import MagicMock, Mock

# import project modules
from openstef.tasks.utils.predictionjobloop import (
    PredictionJobException,
    PredictionJobLoop,
)

# define constants
PREDICTION_JOBS = TestData.get_prediction_jobs()
NUM_PREDICTION_JOBS = len(PREDICTION_JOBS)


class TestPredictionJob(BaseTestCase):
    def test_prediction_job_loop_success(self):
        # Build mocks
        context_mock = MagicMock()
        context_mock.database.get_prediction_jobs.return_value = PREDICTION_JOBS
        on_exception_callback = Mock()
        on_successful_callback = Mock()
        on_end_callback = Mock()
        function_mock = Mock()

        # Create loop that succeeds
        PredictionJobLoop(
            context_mock,
            False,
            True,
            on_exception_callback,
            on_successful_callback,
            on_end_callback,
        ).map(function_mock)

        self.assertEqual(function_mock.call_count, NUM_PREDICTION_JOBS)
        self.assertEqual(on_exception_callback.call_count, 0)
        self.assertEqual(on_successful_callback.call_count, NUM_PREDICTION_JOBS)
        self.assertEqual(on_end_callback.call_count, NUM_PREDICTION_JOBS)

    def test_prediction_job_loop_fail(self):
        # Build mocks
        context_mock = MagicMock()
        context_mock.database.get_prediction_jobs.return_value = PREDICTION_JOBS
        on_exception_callback = Mock()
        on_successful_callback = Mock()
        on_end_callback = Mock()
        function_mock = Mock(side_effect=Exception("Test"))

        # Create loop that fails
        with self.assertRaises(PredictionJobException):
            PredictionJobLoop(
                context_mock,
                False,
                True,
                on_exception_callback,
                on_successful_callback,
                on_end_callback,
            ).map(function_mock)

        self.assertEqual(function_mock.call_count, NUM_PREDICTION_JOBS)
        self.assertEqual(on_exception_callback.call_count, NUM_PREDICTION_JOBS)
        self.assertEqual(on_successful_callback.call_count, 0)
        self.assertEqual(on_end_callback.call_count, NUM_PREDICTION_JOBS)

    def test_prediction_job_loop_fail_once(self):
        # Build mocks
        context_mock = MagicMock()
        context_mock.database.get_prediction_jobs.return_value = PREDICTION_JOBS
        on_exception_callback = Mock()
        on_successful_callback = Mock()
        on_end_callback = Mock()
        function_mock = Mock(side_effect=Exception("Test"))

        # Create loop that fails and stops
        with self.assertRaises(PredictionJobException):
            PredictionJobLoop(
                context_mock,
                True,
                True,
                on_exception_callback,
                on_successful_callback,
                on_end_callback,
            ).map(function_mock)

        self.assertEqual(function_mock.call_count, 1)
        self.assertEqual(on_exception_callback.call_count, 1)
        self.assertEqual(on_successful_callback.call_count, 0)
        self.assertEqual(on_end_callback.call_count, 1)

    def test_prediction_job_loop_debug_pid(self):
        """Test if a list of prediction_jobs with len 1 is returned if debug_pid is given"""
        context_mock = MagicMock()
        pjl = PredictionJobLoop(context_mock, debug_pid=1)
        self.assertEqual(len(pjl.prediction_jobs), 1)


if __name__ == "__main__":
    unittest.main()
