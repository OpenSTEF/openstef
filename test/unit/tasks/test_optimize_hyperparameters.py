# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from unittest import TestCase
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from test.utils import TestData
from openstf.tasks.optimize_hyperparameters import optimize_hyperparameters_task

HYPERPARAM_MOCK = dict(float_param=1.0, string_param="string")


class TestOptimizeHyperparametersTask(TestCase):
    def setUp(self) -> None:
        self.pj = TestData.get_prediction_job(pid=307)
        self.context = MagicMock()
        # Define default return values that make sense for happy flow
        LAST_OPTIMIZED_AGO_DAYS = 35
        self.context.database.get_hyper_params_last_optimized.return_value = (
            datetime.utcnow() - timedelta(days=LAST_OPTIMIZED_AGO_DAYS)
        )

    @patch(
        "openstf.tasks.optimize_hyperparameters.optimize_hyperparameters_pipeline",
        MagicMock(return_value=HYPERPARAM_MOCK),
    )
    def test_optimize_hyperparameters_happy_flow(self):
        # Test happy flow of create forecast task
        context = self.context

        optimize_hyperparameters_task(self.pj, context)
        # Assert call 5 (write hyperparams to db) matches hyperparam mock
        self.assertDictEqual(context.mock_calls[5].args[-1], HYPERPARAM_MOCK)

    def test_optimize_hyperparameters_new_params(self):
        """If new params are available, do nothing"""
        context = self.context
        # Set old param age to 2 days
        context.database.get_hyper_params_last_optimized.return_value = (
            datetime.utcnow() - timedelta(days=2)
        )
        optimize_hyperparameters_task(self.pj, context)
        self.assertEqual(
            context.mock_calls[1].args[0], "Skip hyperparameter optimization"
        )

    @patch(
        "openstf.tasks.optimize_hyperparameters.optimize_hyperparameters_pipeline",
        MagicMock(return_value=HYPERPARAM_MOCK),
    )
    def test_optimize_hyperparams_no_old_params(self):
        """If no old hyperparams exist, new hyperparams should be optimized"""
        context = self.context
        # Set old param age to None
        context.database.get_hyper_params_last_optimized.return_value = None

        optimize_hyperparameters_task(self.pj, context)
        # Same as happy flow, evaluate the pipeline has written hyperparams
        self.assertDictEqual(context.mock_calls[5].args[-1], HYPERPARAM_MOCK)
