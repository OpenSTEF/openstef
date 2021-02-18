# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest
from test.utils import TestData
from unittest.mock import MagicMock, patch

from stf.model.hyper_parameters import optimize_hyperparameters

from test.utils import BaseTestCase

# define constants
PJ = TestData.get_prediction_job(pid=307)
XGB_HYPER_PARAMS = {
    "subsample": 0.9,
    "min_child_weight": 4,
    "max_depth": 8,
    "gamma": 0.5,
    "colsample_bytree": 0.85,
    "eta": 0.1,
    "featureset_name": "G",
    "training_period_days": 90,
}


@patch("stf.model.hyper_parameters.post_teams")
@patch("stf.model.hyper_parameters.ModelTrainerCreator")
@patch("stf.model.hyper_parameters.is_data_sufficient")
@patch("stf.model.hyper_parameters.pre_process_data")
@patch("stf.model.hyper_parameters.DataBase")
@patch("stf.model.hyper_parameters.optuna")
class TestHyperParameters(BaseTestCase):
    def setUp(self):
        super().setUp()
        self.db_mock = MagicMock()
        self.db_mock.get_prediction_job.return_value = PJ

    def test_optimize_hyper_parameters_no_exception(
        self,
        optuna_mock,
        dataBase_mock,
        pre_process_data_mock,
        is_data_sufficient_mock,
        modeltrainercreator_mock,
        post_teams_mock
    ):
        self.add_mock_return_values(
            optuna_mock, dataBase_mock, pre_process_data_mock, is_data_sufficient_mock
        )
        # run function under test
        optimize_hyperparameters(PJ["id"])

        # anlyse testing results
        study_mock = optuna_mock.create_study.return_value

        # test if the following methods are called
        self.assertEqual(dataBase_mock.return_value.get_prediction_job.call_count, 1)
        self.assertEqual(dataBase_mock.return_value.get_model_input.call_count, 1)
        self.assertEqual(pre_process_data_mock.call_count, 1)
        self.assertEqual(optuna_mock.create_study.call_count, 2)
        self.assertEqual(study_mock.optimize.call_count, 2)
        self.assertEqual(post_teams_mock.call_count, 1)

        # test if the following return arguments are used as input argument
        # for the given functions/methods
        self.assertEqual(
            dataBase_mock.return_value.get_model_input.return_value,
            pre_process_data_mock.call_args[0][0],
        )
        self.assertEqual(modeltrainercreator_mock.call_args[0][0], PJ)
        self.assertEqual(
            pre_process_data_mock.return_value, is_data_sufficient_mock.call_args[0][0]
        )

    def test_optimize_hyper_parameters_exception(
        self,
        optuna_mock,
        dataBase_mock,
        pre_process_data_mock,
        is_data_sufficient_mock,
        modeltrainercreator_mock,
        post_teams_mock
    ):
        self.add_mock_return_values(
            optuna_mock, dataBase_mock, pre_process_data_mock, is_data_sufficient_mock
        )
        # Set condition which should case an Exception
        is_data_sufficient_mock.return_value = False

        # test if the right exception is raised
        with self.assertRaises(ValueError):
            optimize_hyperparameters(PJ["id"])

    @staticmethod
    def add_mock_return_values(
        optuna_mock, dataBase_mock, pre_process_data_mock, is_data_sufficient_mock
    ):
        # pre_process_data_mock.return_value = data_table
        dataBase_mock.return_value.get_prediction_job.return_value = PJ
        is_data_sufficient_mock.return_value = True
        optuna_mock.create_study.return_value.best_params = XGB_HYPER_PARAMS


if __name__ == "__main__":
    unittest.main()
