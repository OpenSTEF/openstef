# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest
from datetime import datetime, timedelta
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData
from unittest.mock import MagicMock, patch

import pandas as pd
import sklearn

from openstef.enums import MLModelType
from openstef.exceptions import (
    InputDataInsufficientError,
    InputDataWrongColumnOrderError,
)
from openstef.feature_engineering.feature_applicator import TrainFeatureApplicator
from openstef.metrics.reporter import Report
from openstef.model_selection.model_selection import split_data_train_validation_test
from openstef.pipeline.train_model import (
    train_model_pipeline,
    train_model_pipeline_core,
)
from openstef.validation import validation
from openstef.model.regressors.custom_regressor import CustomOpenstfRegressor
from openstef.model.objective import RegressorObjective

# define constants


class DummyObjective(RegressorObjective):
    ...


class DummyRegressor(CustomOpenstfRegressor):
    @staticmethod
    def valid_kwargs():
        return []

    @property
    def objective(self):
        return DummyObjective

    @property
    def feature_names(self):
        return self._feature_names

    def fit(self, X, y, **fit_params):
        self._feature_names = list(X.columns)
        return self

    def predict(self, X, **kwargs):
        import numpy as np

        return np.zeros(len(X))

    def set_feature_importance(self):
        return pd.DataFrame(
            {
                "weight": [0] * len(self.feature_names),
                "gain": [0] * len(self.feature_names),
            },
            index=self.feature_names,
        )


PJ = TestData.get_prediction_job(pid=307)

XGB_HYPER_PARAMS = {
    "subsample": 0.9,
    "min_child_weight": 4,
    "max_depth": 8,
    "gamma": 0.5,
    "colsample_bytree": 0.85,
    "eta": 0.1,
    "training_period_days": 90,
}


class TestTrainModelPipeline(BaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.pj, self.modelspecs = TestData.get_prediction_job_and_modelspecs(pid=307)
        # Set n_estimators to a small number to speed up training
        self.modelspecs.hyper_params["n_estimators"] = 3
        datetime_start = datetime.utcnow() - timedelta(days=90)
        datetime_end = datetime.utcnow()
        self.data_table = TestData.load("input_data_train.pickle").head(8641)
        self.data = pd.DataFrame(
            index=pd.date_range(datetime_start, datetime_end, freq="15T")
        )

        self.train_input = TestData.load("reference_sets/307-train-data.csv")

    @unittest.skip("Do this test if we want to train a new model")
    def test_train_model_pipeline_update_stored_model(self):
        """Test happy flow of the train model pipeline"""

        train_model_pipeline(
            pj=self.pj,
            input_data=self.train_input,
            check_old_model_age=False,
            trained_models_folder="./test/unit/trained_models",
        )

    def test_train_model_pipeline_core_happy_flow(self):
        """Test happy flow of the train model pipeline

        NOTE this does not explain WHY this is the case?
        The input data should not contain features (e.g. T-7d),
        but it can/should include predictors (e.g. weather data)

        """
        # Select 50 data points to speedup test
        train_input = self.train_input.iloc[::50, :]
        for model_type in list(MLModelType) + [__name__ + ".DummyRegressor"]:
            with self.subTest(model_type=model_type):
                pj = self.pj

                pj["model"] = (
                    model_type.value if hasattr(model_type, "value") else model_type
                )
                modelspecs = self.modelspecs
                train_input = self.train_input

                # Use default parameters
                self.modelspecs.hyper_params = {}
                self.modelspecs.hyper_params["max_epochs"] = 1

                # For Linear model we need to choose an imputation strategy to handle missing value
                if model_type == MLModelType.LINEAR:
                    self.modelspecs.hyper_params["imputation_strategy"] = "mean"

                model, report, modelspecs = train_model_pipeline_core(
                    pj=pj, modelspecs=self.modelspecs, input_data=train_input
                )

                # check if the model was fitted (raises NotFittedError when not fitted)
                self.assertIsNone(sklearn.utils.validation.check_is_fitted(model))

                # check if the model has a feature_names property
                self.assertIsNotNone(model.feature_names)

                # check if model is sklearn compatible
                self.assertTrue(isinstance(model, sklearn.base.BaseEstimator))

                # check if report is a Report
                self.assertTrue(isinstance(report, Report))

                # Validate and clean data
                validated_data = validation.clean(
                    validation.validate(pj["id"], train_input)
                )

                # Add features
                data_with_features = TrainFeatureApplicator(
                    horizons=[0.25, 47.0], feature_names=self.modelspecs.feature_names
                ).add_features(validated_data, pj=pj)

                # Split data
                (
                    _,
                    _,
                    train_data,
                    validation_data,
                    test_data,
                ) = split_data_train_validation_test(data_with_features)

                # not able to generate a feature importance for proloaf as this is a neural network
                if not pj["model"] == "proloaf":
                    importance = model.set_feature_importance()
                    self.assertIsInstance(importance, pd.DataFrame)

    def test_train_model_pipeline_with_featureAdders(self):
        pj = self.pj
        modelspecs = self.modelspecs
        modelspecs.hyper_params = {}
        modelspecs.feature_modules = [
            "test.unit.feature_engineering.test_feature_adder"
        ]
        dummy_feature = "dummy_0.5"
        modelspecs.feature_names.append(dummy_feature)

        train_input = self.train_input.iloc[::50, :]
        model, report, modelspecs = train_model_pipeline_core(
            pj=pj, modelspecs=modelspecs, input_data=train_input
        )

        # check if the model was fitted (raises NotFittedError when not fitted)
        self.assertIsNone(sklearn.utils.validation.check_is_fitted(model))

        # check if the model has a feature_names property
        self.assertIsNotNone(model.feature_names)
        self.assertTrue(dummy_feature in model.feature_names)

        # check if model is sklearn compatible
        self.assertTrue(isinstance(model, sklearn.base.BaseEstimator))

        # check if report is a Report
        self.assertTrue(isinstance(report, Report))

    @patch("openstef.model.serializer.MLflowSerializer.save_model")
    @patch("openstef.pipeline.train_model.train_model_pipeline_core")
    @patch("openstef.pipeline.train_model.MLflowSerializer")
    def test_train_model_pipeline_happy_flow(
        self, serializer_mock, pipeline_mock, save_model_mock
    ):
        """Test happy flow of the train model pipeline"""

        old_model_mock = MagicMock()
        old_model_mock.age = 8

        serializer_mock_instance = MagicMock()
        serializer_mock_instance.load_model.return_value = (
            old_model_mock,
            self.modelspecs,
        )
        serializer_mock.return_value = serializer_mock_instance

        report_mock = MagicMock()
        pipeline_mock.return_value = ("a", report_mock, self.modelspecs)

        trained_models_folder = "./test/unit/trained_models"
        train_model_pipeline(
            pj=self.pj,
            input_data=self.train_input,
            check_old_model_age=False,
            trained_models_folder=trained_models_folder,
        )

    @patch("openstef.model.serializer.MLflowSerializer.save_model")
    @patch("openstef.pipeline.train_model.train_model_pipeline_core")
    @patch("openstef.pipeline.train_model.MLflowSerializer")
    def test_train_model_pipeline_young_model(
        self, serializer_mock, pipeline_mock, save_model_mock
    ):
        """Test pipeline core is not called when model is young"""
        old_model_mock = MagicMock()
        old_model_mock.age = 3

        serializer_mock_instance = MagicMock()
        serializer_mock_instance.load_model.return_value = (
            old_model_mock,
            self.modelspecs,
        )
        serializer_mock.return_value = serializer_mock_instance

        report_mock = MagicMock()
        pipeline_mock.return_value = ("a", report_mock)

        train_model_pipeline(
            pj=self.pj,
            input_data=self.train_input,
            check_old_model_age=True,
            trained_models_folder="./test/unit/trained_models",
        )
        self.assertFalse(pipeline_mock.called)

    @patch("openstef.model.serializer.MLflowSerializer.save_model")
    @patch("openstef.validation.validation.is_data_sufficient", return_value=False)
    def test_train_model_InputDataInsufficientError(
        self, validation_is_data_sufficient_mock, save_model_mock
    ):
        # This error is caught and then raised again and logged
        with self.assertLogs(
            "openstef.pipeline.train_model", level="ERROR"
        ) as captured:
            with self.assertRaises(InputDataInsufficientError):
                train_model_pipeline(
                    pj=self.pj,
                    input_data=self.train_input,
                    check_old_model_age=False,
                    trained_models_folder="./test/unit/trained_models",
                )

        self.assertEqual(
            len(captured.records), 1
        )  # check that there is only one error log message
        # search for log
        self.assertRegex(
            captured.records[0].getMessage(),
            "Input data is insufficient after validation and cleaning",
        )

    @patch("openstef.model.serializer.MLflowSerializer.save_model")
    def test_train_model_InputDataWrongColumnOrderError(self, save_model_mock):
        # change the column order
        input_data = self.train_input.iloc[:, ::-1]

        # This error is caught and then raised again and logged
        with self.assertLogs(
            "openstef.pipeline.train_model", level="ERROR"
        ) as captured:
            with self.assertRaises(InputDataWrongColumnOrderError):
                train_model_pipeline(
                    pj=self.pj,
                    input_data=input_data,
                    check_old_model_age=False,
                    trained_models_folder="./test/unit/trained_models",
                )

        self.assertEqual(
            len(captured.records), 1
        )  # check that there is only one error log message
        # search for log
        self.assertRegex(captured.records[0].getMessage(), "Wrong column order")

    @patch("openstef.model.serializer.MLflowSerializer.save_model")
    @patch("openstef.pipeline.train_model.MLflowSerializer")
    def test_train_model_OldModelHigherScoreError(
        self, serializer_mock, save_model_mock
    ):
        # Mock an old model which is better than the new one.
        old_model_mock = MagicMock()
        old_model_mock.age = 8

        serializer_mock_instance = MagicMock()
        serializer_mock_instance.load_model.return_value = (
            old_model_mock,
            self.modelspecs,
        )
        serializer_mock.return_value = serializer_mock_instance
        old_model_mock.score.return_value = 5

        # This error is caught so we check if logging contains the error.
        with self.assertLogs(
            "openstef.pipeline.train_model", level="ERROR"
        ) as captured:
            train_model_pipeline(
                pj=self.pj,
                input_data=self.train_input,
                check_old_model_age=True,
                trained_models_folder="./test/unit/trained_models",
            )

        self.assertEqual(
            len(captured.records), 1
        )  # check that there is only one error log message
        # search for the old model is better log
        self.assertRegex(
            captured.records[0].getMessage(), "Old model is better than new model"
        )

    @patch("openstef.model.serializer.MLflowSerializer.save_model")
    @patch("openstef.pipeline.train_model.MLflowSerializer")
    def test_train_model_log_new_model_better(self, serializer_mock, save_model_mock):
        # Mock an old model which is better than the new one.
        old_model_mock = MagicMock()
        old_model_mock.age = 8

        serializer_mock_instance = MagicMock()
        serializer_mock_instance.load_model.return_value = (
            old_model_mock,
            self.modelspecs,
        )
        serializer_mock.return_value = serializer_mock_instance
        old_model_mock.score.return_value = 0.1

        with self.assertLogs("openstef.pipeline.train_model", level="INFO") as captured:
            train_model_pipeline(
                pj=self.pj,
                input_data=self.train_input,
                check_old_model_age=True,
                trained_models_folder="./test/unit/trained_models",
            )

        # search for the old model is better log
        self.assertRegex(
            captured.records[0].getMessage(), "New model is better than old model"
        )

    @patch("openstef.model.serializer.MLflowSerializer.save_model")
    @patch("openstef.pipeline.train_model.MLflowSerializer")
    def test_train_model_log_couldnt_compare(self, serializer_mock, save_model_mock):
        # Mock an old model which is better than the new one.
        old_model_mock = MagicMock()
        old_model_mock.age = 8

        serializer_mock_instance = MagicMock()
        serializer_mock_instance.load_model.return_value = (
            old_model_mock,
            self.modelspecs,
        )
        serializer_mock.return_value = serializer_mock_instance
        old_model_mock.score.side_effect = ValueError()

        with self.assertLogs("openstef.pipeline.train_model", level="INFO") as captured:
            train_model_pipeline(
                pj=self.pj,
                input_data=self.train_input,
                check_old_model_age=True,
                trained_models_folder="./test/unit/trained_models",
            )

        # search for the old model is better log
        self.assertRegex(
            captured.records[0].getMessage(), "Could not compare to old model"
        )

    @patch("openstef.model.serializer.MLflowSerializer.save_model")
    @patch("openstef.pipeline.train_model.MLflowSerializer")
    def test_train_model_No_old_model(self, serializer_mock, save_model_mock):
        # Mock an old model which is better than the new one.
        old_model_mock = MagicMock()
        old_model_mock.age = 8

        serializer_mock_instance = MagicMock()
        serializer_mock_instance.load_model.return_value = (
            old_model_mock,
            self.modelspecs,
        )
        serializer_mock_instance.load_model.side_effect = FileNotFoundError()
        serializer_mock.return_value = serializer_mock_instance

        with self.assertLogs(
            "openstef.pipeline.train_model", level="WARNING"
        ) as captured:
            train_model_pipeline(
                pj=self.pj,
                input_data=self.train_input,
                check_old_model_age=True,
                trained_models_folder="./test/unit/trained_models",
            )

        # search for the old model is better log
        self.assertRegex(captured.records[0].getMessage(), "No old model found")


if __name__ == "__main__":
    unittest.main()
