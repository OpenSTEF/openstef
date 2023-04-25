# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData
from unittest.mock import patch

import pandas as pd

from openstef.data_classes.model_specifications import ModelSpecificationDataClass
from openstef.data_classes.split_function import SplitFuncDataClass
from openstef.exceptions import (
    InputDataInsufficientError,
    InputDataWrongColumnOrderError,
)
from openstef.metrics.reporter import Report
from openstef.model_selection.model_selection import split_data_train_validation_test
from openstef.model.regressors.regressor import OpenstfRegressor
from openstef.model.objective_creator import ObjectiveCreator
from openstef.pipeline.optimize_hyperparameters import (
    optimize_hyperparameters_pipeline,
    optimize_hyperparameters_pipeline_core,
    optuna_optimization,
)


def dummy_split(data, validation_fraction, test_fraction):
    return data.iloc[:100], data.iloc[100:110], data.iloc[110:120]


class TestOptimizeHyperParametersPipeline(BaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.input_data = TestData.load("reference_sets/307-train-data.csv")
        self.pj, self.modelspecs = TestData.get_prediction_job_and_modelspecs(pid=307)
        self.pj.default_modelspecs = self.modelspecs
        self.input_data_with_features = TestData.load(
            "input_data_multi_horizon_features.csv"
        )

    @patch("openstef.model.serializer.MLflowSerializer.save_model")
    def test_optimize_hyperparameters_pipeline(self, save_model_mock):
        """Also check if non-default quantiles are processed correctly"""
        pj = self.pj
        predefined_quantiles = (0.001, 0.5)
        pj["quantiles"] = predefined_quantiles

        parameters = optimize_hyperparameters_pipeline(
            pj,
            self.input_data,
            mlflow_tracking_uri="./test/unit/trained_models/mlruns",
            artifact_folder="./test/unit/trained_models",
            n_trials=2,
        )
        self.assertIsInstance(parameters, dict)
        # Assert stored quantiles are the same as the predefined_quantiles
        stored_quantiles = save_model_mock.call_args[1]["model_specs"]["hyper_params"][
            "quantiles"
        ]
        self.assertTupleEqual(stored_quantiles, predefined_quantiles)

    def test_optimize_hyperparameters_pipeline_core(self):
        """Also check if non-default quantiles are processed correctly"""
        pj = self.pj
        predefined_quantiles = (0.001, 0.5)
        pj["quantiles"] = predefined_quantiles

        result = optimize_hyperparameters_pipeline_core(
            pj,
            self.input_data,
            n_trials=2,
        )
        self.assertIsInstance(result[0], OpenstfRegressor)
        self.assertIsInstance(result[1], ModelSpecificationDataClass)
        self.assertIsInstance(result[2], Report)
        self.assertIsInstance(result[3], dict)
        self.assertIsInstance(result[4], int)
        self.assertIsInstance(result[5], dict)

    def test_optuna_optimization_default_split(self):
        pj = self.pj

        objective = ObjectiveCreator.create_objective(model_type=pj["model"])

        _, objective = optuna_optimization(
            pj, objective, self.input_data_with_features, n_trials=2
        )

        self.assertEqual(objective.split_func, split_data_train_validation_test)

    def test_optuna_optimization_with_custom_split(self):
        pj = self.pj
        pj.train_split_func = SplitFuncDataClass(function=dummy_split, arguments={})

        objective = ObjectiveCreator.create_objective(model_type=pj["model"])

        _, objective = optuna_optimization(
            pj, objective, self.input_data_with_features, n_trials=2
        )

        self.assertEqual(objective.split_func, dummy_split)

    def test_optimize_hyperparameters_pipeline_with_custom_split(self):
        """Also check if non-default quantiles are processed correctly"""
        pj = self.pj
        pj.train_split_func = SplitFuncDataClass(function=dummy_split, arguments={})
        predefined_quantiles = (0.001, 0.5)
        pj["quantiles"] = predefined_quantiles

        result = optimize_hyperparameters_pipeline_core(
            pj,
            self.input_data,
            n_trials=2,
        )
        self.assertIsInstance(result[0], OpenstfRegressor)
        self.assertIsInstance(result[1], ModelSpecificationDataClass)
        self.assertIsInstance(result[2], Report)
        self.assertIsInstance(result[3], dict)
        self.assertIsInstance(result[4], int)
        self.assertIsInstance(result[5], dict)

    @patch("openstef.validation.validation.is_data_sufficient", return_value=False)
    def test_optimize_hyperparameters_pipeline_insufficient_data(self, mock):

        # if data is not sufficient a InputDataInsufficientError should be raised
        with self.assertRaises(InputDataInsufficientError):
            optimize_hyperparameters_pipeline_core(
                self.pj,
                self.input_data,
                n_trials=2,
            )

    def test_optimize_hyperparameters_pipeline_no_data(self):
        input_data = pd.DataFrame()

        # if there is no data a InputDataInsufficientError should be raised
        with self.assertRaises(InputDataInsufficientError):
            optimize_hyperparameters_pipeline_core(
                self.pj,
                input_data,
                n_trials=2,
            )

    def test_optimize_hyperparameters_pipeline_no_load_data(self):

        input_data = self.input_data.drop("load", axis=1)
        # if there is no data a InputDataWrongColumnOrderError should be raised
        with self.assertRaises(InputDataWrongColumnOrderError):
            optimize_hyperparameters_pipeline_core(
                self.pj,
                input_data,
                n_trials=2,
            )

    @patch("openstef.model.serializer.MLflowSerializer.save_model")
    def test_optimize_hyperparameters_pipeline_quantile_regressor(
        self, save_model_mock
    ):
        """If the regressor can predict quantiles explicitly,
        the model should be retrained for the desired quantiles"""
        pj = self.pj
        predefined_quantiles = (0.001, 0.5)
        pj["quantiles"] = predefined_quantiles
        pj["model"] = "xgb_quantile"

        parameters = optimize_hyperparameters_pipeline(
            pj,
            self.input_data,
            mlflow_tracking_uri="./test/unit/trained_models/mlruns",
            artifact_folder="./test/unit/trained_models",
            n_trials=1,
        )
        self.assertIsInstance(parameters, dict)
        # Assert stored quantiles are the same as the predefined_quantiles
        stored_quantiles = save_model_mock.call_args[1]["model"].quantiles
        self.assertTupleEqual(stored_quantiles, predefined_quantiles)
