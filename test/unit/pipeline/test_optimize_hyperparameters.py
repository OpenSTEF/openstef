# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData
from unittest.mock import patch

import pandas as pd

from openstef.exceptions import (
    InputDataInsufficientError,
    InputDataWrongColumnOrderError,
)
from openstef.pipeline.optimize_hyperparameters import optimize_hyperparameters_pipeline


class TestOptimizeHyperParametersPipeline(BaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.input_data = TestData.load("reference_sets/307-train-data.csv")
        self.pj, self.modelspecs = TestData.get_prediction_job_and_modelspecs(pid=307)

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
            trained_models_folder="./test/unit/trained_models/mlruns",
            n_trials=2,
        )
        self.assertIsInstance(parameters, dict)
        # Assert stored quantiles are the same as the predefined_quantiles
        stored_quantiles = save_model_mock.call_args[1]["model_specs"]["hyper_params"][
            "quantiles"
        ]
        self.assertTupleEqual(stored_quantiles, predefined_quantiles)

    @patch("openstef.validation.validation.is_data_sufficient", return_value=False)
    def test_optimize_hyperparameters_pipeline_insufficient_data(self, mock):

        # if data is not sufficient a InputDataInsufficientError should be raised
        with self.assertRaises(InputDataInsufficientError):
            optimize_hyperparameters_pipeline(
                self.pj,
                self.input_data,
                mlflow_tracking_uri="./test/unit/trained_models/mlruns",
                trained_models_folder="./test/unit/trained_models/mlruns",
                n_trials=2,
            )

    def test_optimize_hyperparameters_pipeline_no_data(self):
        input_data = pd.DataFrame()

        # if there is no data a InputDataInsufficientError should be raised
        with self.assertRaises(InputDataInsufficientError):
            optimize_hyperparameters_pipeline(
                self.pj,
                input_data,
                mlflow_tracking_uri="./test/unit/trained_models/mlruns",
                trained_models_folder="./test/unit/trained_models/mlruns",
                n_trials=2,
            )

    def test_optimize_hyperparameters_pipeline_no_load_data(self):

        input_data = self.input_data.drop("load", axis=1)
        # if there is no data a InputDataWrongColumnOrderError should be raised
        with self.assertRaises(InputDataWrongColumnOrderError):
            optimize_hyperparameters_pipeline(
                self.pj,
                input_data,
                mlflow_tracking_uri="./test/unit/trained_models/mlruns",
                trained_models_folder="./test/unit/trained_models/mlruns",
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
            trained_models_folder="./test/unit/trained_models/mlruns",
            n_trials=1,
        )
        self.assertIsInstance(parameters, dict)
        # Assert stored quantiles are the same as the predefined_quantiles
        stored_quantiles = save_model_mock.call_args[1]["model"].quantiles
        self.assertTupleEqual(stored_quantiles, predefined_quantiles)
