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

from openstef.data_classes.prediction_job import PredictionJobDataClass

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

    def test_prediction_job_loop_with_dependencies(self):
        # We check that the prediction jobs are called in the
        # correct order when dependencies are present.
        def make_prediction_job(pj_id, depends_on=None):
            return PredictionJobDataClass(
                id=pj_id,
                depends_on=depends_on,
                model="",
                forecast_type="",
                train_components=False,
                name="",
                lat=0,
                lon=0,
                resolution_minutes=0,
                horizon_minutes=0
            )

        pjs = [
            make_prediction_job(1),
            make_prediction_job(2),
            make_prediction_job(3),
            make_prediction_job(4, depends_on=[1, 2]),
            make_prediction_job(5, depends_on=[1, 3]),
            make_prediction_job(6, depends_on=[4]),
            make_prediction_job(7),
        ]

        context_mock = MagicMock()
        context_mock.database.get_prediction_jobs.return_value = pjs

        class MockFunction:
            def __init__(self):
                self.pjs = []

            def __call__(self, pj, *args, **kwargs):
                self.pjs.append(pj)

        function_mock = MockFunction()

        PredictionJobLoop(
            context_mock,
        ).map(function_mock)

        found_pjs = function_mock.pjs
        group1 = set(pj.id for pj in found_pjs[:4])
        group2 = set(pj.id for pj in found_pjs[4:6])
        group3 = set(pj.id for pj in found_pjs[6:])

        assert group1 == {1, 2, 3, 7}
        assert group2 == {4, 5}
        assert group3 == {6}


if __name__ == "__main__":
    unittest.main()
