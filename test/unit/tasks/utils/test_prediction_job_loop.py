# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

# import builtins
import unittest
from test.utils import TestData
from unittest.mock import MagicMock, Mock

# import project modules
from stf.tasks.utils.predictionjobloop import (
    PredictionJobException, PredictionJobLoop
)

from test.utils import BaseTestCase

# define constants
PREDICTION_JOBS = TestData.get_prediction_jobs()
NUM_PREDICTION_JOBS = len(PREDICTION_JOBS)


class TestPredictionJob(BaseTestCase):
    def test_prediction_job_loop_success(self):
        # Build mocks
        context_mock = MagicMock()
        context_mock.database.get_prediction_jobs.return_value = PREDICTION_JOBS
        on_exception = Mock()
        on_successful = Mock()
        on_end = Mock()
        function_mock = Mock()

        # Create loop that succeeds
        PredictionJobLoop(context_mock, False, True, on_exception,
                          on_successful, on_end).map(function_mock)

        self.assertEqual(function_mock.call_count, NUM_PREDICTION_JOBS)
        self.assertEqual(on_exception.call_count, 0)
        self.assertEqual(on_successful.call_count, NUM_PREDICTION_JOBS)
        self.assertEqual(on_end.call_count, NUM_PREDICTION_JOBS)

    def test_prediction_job_loop_fail(self):
        # Build mocks
        context_mock = MagicMock()
        context_mock.database.get_prediction_jobs.return_value = PREDICTION_JOBS
        on_exception = Mock()
        on_successful = Mock()
        on_end = Mock()
        function_mock = Mock(side_effect=Exception("Test"))

        # Create loop that fails
        with self.assertRaises(PredictionJobException):
            PredictionJobLoop(context_mock, False, True, on_exception,
                              on_successful, on_end).map(function_mock)

        self.assertEqual(function_mock.call_count, NUM_PREDICTION_JOBS)
        self.assertEqual(on_exception.call_count, NUM_PREDICTION_JOBS)
        self.assertEqual(on_successful.call_count, 0)
        self.assertEqual(on_end.call_count, NUM_PREDICTION_JOBS)

    def test_prediction_job_loop_fail_once(self):
        # Build mocks
        context_mock = MagicMock()
        context_mock.database.get_prediction_jobs.return_value = PREDICTION_JOBS
        on_exception = Mock()
        on_successful = Mock()
        on_end = Mock()
        function_mock = Mock(side_effect=Exception("Test"))

        # Create loop that fails and stops
        with self.assertRaises(PredictionJobException):
            PredictionJobLoop(context_mock, True, True, on_exception,
                              on_successful, on_end).map(function_mock)

        self.assertEqual(function_mock.call_count, 1)
        self.assertEqual(on_exception.call_count, 1)
        self.assertEqual(on_successful.call_count, 0)
        self.assertEqual(on_end.call_count, 1)


if __name__ == "__main__":
    unittest.main()
