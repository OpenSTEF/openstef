# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import sklearn
from openstf_dbc.services.model_specifications import ModelSpecificationRetriever

from openstf.enums import MLModelType
from openstf.exceptions import (
    InputDataInsufficientError,
    InputDataWrongColumnOrderError,
)
from openstf.feature_engineering.feature_applicator import TrainFeatureApplicator
from openstf.metrics.reporter import Report
from openstf.model_selection.model_selection import split_data_train_validation_test
from openstf.pipeline.train_model import (
    train_model_pipeline,
    train_model_pipeline_core,
)
from openstf.validation import validation
from test.utils import BaseTestCase, TestData

# define constants

PJ, modelspecs = TestData.get_prediction_job(pid=307)
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
        self.pj, self.modelspecs = TestData.get_prediction_job(pid=307)
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
            modelspecs=self.modelspecs,
            input_data=self.train_input,
            check_old_model_age=False,
            trained_models_folder="./test/trained_models",
        )

    def test_train_model_pipeline_core_happy_flow(self):
        """Test happy flow of the train model pipeline

        NOTE this does not explain WHY this is the case?
        The input data should not contain features (e.g. T-7d),
        but it can/should include predictors (e.g. weather data)

        """
        # Select 50 data points to speedup test
        train_input = self.train_input.iloc[::50, :]
        for model_type in MLModelType:
            with self.subTest(model_type=model_type):
                pj = self.pj
                pj["model"] = model_type.value

                # Use default parameters
                self.modelspecs.hyper_params = {}
                model, report = train_model_pipeline_core(pj=pj, modelspecs=self.modelspecs, input_data=self.train_input)

                # check if the model was fitted (raises NotFittedError when not fitted)
                self.assertIsNone(sklearn.utils.validation.check_is_fitted(model))

                # check if model is sklearn compatible
                self.assertTrue(isinstance(model, sklearn.base.BaseEstimator))

                # check if report is a Report
                self.assertTrue(isinstance(report, Report))

                # Validate and clean data
                validated_data = validation.clean(
                    validation.validate(pj["id"], self.train_input)
                )

                # Add features
                data_with_features = TrainFeatureApplicator(
                    horizons=[0.25, 47.0], feature_names=self.modelspecs.feature_names
                ).add_features(validated_data)

                # Split data
                (
                    _,
                    _,
                    train_data,
                    validation_data,
                    test_data,
                ) = split_data_train_validation_test(data_with_features)

                # split x and y data
                train_x = train_data.iloc[:, 1:-1]
                importance = model.set_feature_importance(train_x.columns)
                self.assertIsInstance(importance, pd.DataFrame)

    @patch("openstf.model.serializer.PersistentStorageSerializer.save_model")
    @patch("openstf.pipeline.train_model.train_model_pipeline_core")
    @patch("openstf.pipeline.train_model.PersistentStorageSerializer")
    def test_train_model_pipeline_happy_flow(
        self, serializer_mock, pipeline_mock, save_model_mock
    ):
        """Test happy flow of the train model pipeline"""

        old_model_mock = MagicMock()
        old_model_mock.age = 8

        serializer_mock_instance = MagicMock()
        serializer_mock_instance.load_model.return_value = old_model_mock, self.modelspecs
        serializer_mock.return_value = serializer_mock_instance

        report_mock = MagicMock()
        pipeline_mock.return_value = ("a", report_mock)

        train_model_pipeline(
            pj=self.pj,
            modelspecs=self.modelspecs,
            input_data=self.train_input,
            check_old_model_age=False,
            trained_models_folder="./test/trained_models",
        )

    @patch("openstf.model.serializer.PersistentStorageSerializer.save_model")
    @patch("openstf.pipeline.train_model.train_model_pipeline_core")
    @patch("openstf.pipeline.train_model.PersistentStorageSerializer")
    def test_train_model_pipeline_young_model(
        self, serializer_mock, pipeline_mock, save_model_mock
    ):
        """Test pipeline core is not called when model is young"""
        old_model_mock = MagicMock()
        old_model_mock.age = 3

        serializer_mock_instance = MagicMock()
        serializer_mock_instance.load_model.return_value = old_model_mock, self.modelspecs
        serializer_mock.return_value = serializer_mock_instance

        report_mock = MagicMock()
        pipeline_mock.return_value = ("a", report_mock)

        train_model_pipeline(
            pj=self.pj,
            modelspecs=self.modelspecs,
            input_data=self.train_input,
            check_old_model_age=True,
            trained_models_folder="./test/trained_models",
        )
        self.assertFalse(pipeline_mock.called)

    @patch("openstf.model.serializer.PersistentStorageSerializer.save_model")
    @patch("openstf.validation.validation.is_data_sufficient", return_value=False)
    def test_train_model_InputDataInsufficientError(
        self, validation_is_data_sufficient_mock, save_model_mock
    ):
        # This error is caught and then raised again and logged
        with self.assertLogs("openstf.pipeline.train_model", level="ERROR") as captured:
            with self.assertRaises(InputDataInsufficientError):
                train_model_pipeline(
                    pj=self.pj,
                    modelspecs=self.modelspecs,
                    input_data=self.train_input,
                    check_old_model_age=False,
                    trained_models_folder="./test/trained_models",
            )

        self.assertEqual(
            len(captured.records), 1
        )  # check that there is only one error log message
        # search for log
        self.assertRegex(
            captured.records[0].getMessage(),
            "Input data is insufficient after validation and cleaning",
        )

    @patch("openstf.model.serializer.PersistentStorageSerializer.save_model")
    def test_train_model_InputDataWrongColumnOrderError(self, save_model_mock):
        # change the column order
        input_data = self.train_input.iloc[:, ::-1]

        # This error is caught and then raised again and logged
        with self.assertLogs("openstf.pipeline.train_model", level="ERROR") as captured:
            with self.assertRaises(InputDataWrongColumnOrderError):
                train_model_pipeline(
                    pj=self.pj,
                    modelspecs=self.modelspecs,
                    input_data=input_data,
                    check_old_model_age=False,
                    trained_models_folder="./test/trained_models",
                )

        self.assertEqual(
            len(captured.records), 1
        )  # check that there is only one error log message
        # search for log
        self.assertRegex(captured.records[0].getMessage(), "Wrong column order")

    @patch("openstf.model.serializer.PersistentStorageSerializer.save_model")
    @patch("openstf.pipeline.train_model.PersistentStorageSerializer")
    def test_train_model_OldModelHigherScoreError(
        self, serializer_mock, save_model_mock
    ):
        # Mock an old model which is better than the new one.
        old_model_mock = MagicMock()
        old_model_mock.age = 8

        serializer_mock_instance = MagicMock()
        serializer_mock_instance.load_model.return_value = old_model_mock, self.modelspecs
        serializer_mock.return_value = serializer_mock_instance
        old_model_mock.score.return_value = 5

        # This error is caught so we check if logging contains the error.
        with self.assertLogs("openstf.pipeline.train_model", level="ERROR") as captured:
            train_model_pipeline(
                pj=self.pj,
                modelspecs=self.modelspecs,
                input_data=self.train_input,
                check_old_model_age=True,
                trained_models_folder="./test/trained_models",
            )

        self.assertEqual(
            len(captured.records), 1
        )  # check that there is only one error log message
        # search for the old model is better log
        self.assertRegex(
            captured.records[0].getMessage(), "Old model is better than new model"
        )

    @patch("openstf.model.serializer.PersistentStorageSerializer.save_model")
    @patch("openstf.pipeline.train_model.PersistentStorageSerializer")
    def test_train_model_log_new_model_better(self, serializer_mock, save_model_mock):
        # Mock an old model which is better than the new one.
        old_model_mock = MagicMock()
        old_model_mock.age = 8

        serializer_mock_instance = MagicMock()
        serializer_mock_instance.load_model.return_value = old_model_mock, self.modelspecs
        serializer_mock.return_value = serializer_mock_instance
        old_model_mock.score.return_value = 0.1

        with self.assertLogs("openstf.pipeline.train_model", level="INFO") as captured:
            train_model_pipeline(
                pj=self.pj,
                modelspecs=self.modelspecs,
                input_data=self.train_input,
                check_old_model_age=True,
                trained_models_folder="./test/trained_models",
            )

        # search for the old model is better log
        self.assertRegex(
            captured.records[0].getMessage(), "New model is better than old model"
        )

    @patch("openstf.model.serializer.PersistentStorageSerializer.save_model")
    @patch("openstf.pipeline.train_model.PersistentStorageSerializer")
    def test_train_model_log_couldnt_compare(self, serializer_mock, save_model_mock):
        # Mock an old model which is better than the new one.
        old_model_mock = MagicMock()
        old_model_mock.age = 8

        serializer_mock_instance = MagicMock()
        serializer_mock_instance.load_model.return_value = old_model_mock, self.modelspecs
        serializer_mock.return_value = serializer_mock_instance
        old_model_mock.score.side_effect = ValueError()

        with self.assertLogs("openstf.pipeline.train_model", level="INFO") as captured:
            train_model_pipeline(
                pj=self.pj,
                modelspecs=self.modelspecs,
                input_data=self.train_input,
                check_old_model_age=True,
                trained_models_folder="./test/trained_models",
            )

        # search for the old model is better log
        self.assertRegex(
            captured.records[0].getMessage(), "Could not compare to old model"
        )

    @patch("openstf.model.serializer.PersistentStorageSerializer.save_model")
    @patch("openstf.pipeline.train_model.PersistentStorageSerializer")
    def test_train_model_No_old_model(self, serializer_mock, save_model_mock):
        # Mock an old model which is better than the new one.
        old_model_mock = MagicMock()
        old_model_mock.age = 8

        serializer_mock_instance = MagicMock()
        serializer_mock_instance.load_model.return_value = old_model_mock, self.modelspecs
        serializer_mock_instance.load_model.side_effect = FileNotFoundError()
        serializer_mock.return_value = serializer_mock_instance

        with self.assertLogs(
            "openstf.pipeline.train_model", level="WARNING"
        ) as captured:
            train_model_pipeline(
                pj=self.pj,
                modelspecs=self.modelspecs,
                input_data=self.train_input,
                check_old_model_age=True,
                trained_models_folder="./test/trained_models",
            )

        # search for the old model is better log
        self.assertRegex(captured.records[0].getMessage(), "No old model found")


if __name__ == "__main__":
    unittest.main()
