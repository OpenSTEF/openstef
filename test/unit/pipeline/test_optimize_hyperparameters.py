# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest
from unittest.mock import patch

import pandas as pd

from openstf.exceptions import (
    InputDataInsufficientError,
    InputDataWrongColumnOrderError,
)
from openstf.pipeline.optimize_hyperparameters import (
    optimize_hyperparameters_pipeline,
)
from test.utils import BaseTestCase, TestData


class TestOptimizeHyperParametersPipeline(BaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.input_data = TestData.load("reference_sets/307-train-data.csv")
        self.pj = TestData.get_prediction_job(pid=307)

    def test_optimize_hyperparameters_pipeline(self):
        parameters = optimize_hyperparameters_pipeline(
            self.pj, self.input_data, "OTHER_TEST"
        )
        self.assertIsInstance(parameters, dict)

    @patch("openstf.validation.validation.is_data_sufficient", return_value=False)
    def test_optimize_hyperparameters_pipeline_insufficient_data(self, mock):

        # if data is not sufficient a InputDataInsufficientError should be raised
        with self.assertRaises(InputDataInsufficientError):
            optimize_hyperparameters_pipeline(self.pj, self.input_data, "OTHER_TEST")

    def test_optimize_hyperparameters_pipeline_no_data(self):
        input_data = pd.DataFrame()

        # if there is no data a InputDataInsufficientError should be raised
        with self.assertRaises(InputDataInsufficientError):
            optimize_hyperparameters_pipeline(self.pj, input_data, "OTHER_TEST")

    def test_optimize_hyperparameters_pipeline_no_load_data(self):

        input_data = self.input_data.drop("load", axis=1)
        # if there is no data a InputDataWrongColumnOrderError should be raised
        with self.assertRaises(InputDataWrongColumnOrderError):
            optimize_hyperparameters_pipeline(self.pj, input_data, "OTHER_TEST")


if __name__ == "__main__":
    unittest.main()
