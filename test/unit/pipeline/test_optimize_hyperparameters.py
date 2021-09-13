# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest
from unittest.mock import patch

from test.utils import BaseTestCase, TestData
from openstf.pipeline.optimize_hyperparameters import optimize_hyperparameters_pipeline


@patch("openstf.pipeline.optimize_hyperparameters.structlog")
@patch("openstf.pipeline.optimize_hyperparameters.validation")
@patch("openstf.pipeline.optimize_hyperparameters.TrainFeatureApplicator")
@patch("openstf.pipeline.optimize_hyperparameters.ObjectiveCreator")
@patch("openstf.pipeline.optimize_hyperparameters.optuna")
class TestOptimizeHyperParametersPipeline(BaseTestCase):
    def test_optimize_hyperparameters_pipeline(self, *args):
        pj = TestData.get_prediction_job(pid=307)
        input_data = TestData.load("input_data_train.pickle")
        hyperparameters = optimize_hyperparameters_pipeline(pj, input_data)
        self.assertIsInstance(hyperparameters, dict)

    def test_optimize_hyperparameters_pipeline_insufficient_data(self, *args):
        validation_mock = args[3]
        validation_mock.is_data_sufficient.return_value = False

        pj = TestData.get_prediction_job(pid=307)
        input_data = TestData.load("input_data_train.pickle")
        # if data is not sufficient a ValueError should be raised
        with self.assertRaises(ValueError):
            optimize_hyperparameters_pipeline(pj, input_data)


if __name__ == "__main__":
    unittest.main()
