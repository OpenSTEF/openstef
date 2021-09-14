# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest
from datetime import datetime, timedelta
from test.utils import BaseTestCase, TestData
from unittest.mock import MagicMock, patch

import pandas as pd
import sklearn

from openstf.feature_engineering.feature_applicator import TrainFeatureApplicator
from openstf.validation import validation
from openstf.metrics.reporter import Report
from openstf.pipeline.train_model import (
    split_data_train_validation_test,
    train_model_pipeline,
    train_model_pipeline_core,
)

# define constants

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
        self.pj = TestData.get_prediction_job(pid=307)
        datetime_start = datetime.utcnow() - timedelta(days=90)
        datetime_end = datetime.utcnow()
        self.data_table = TestData.load("input_data_train.pickle").head(8641)
        self.data = pd.DataFrame(
            index=pd.date_range(datetime_start, datetime_end, freq="15T")
        )

        self.train_input = TestData.load("reference_sets/307-train-data.csv")

    @unittest.skip("If you want to store a newly trained model, use test below")
    def test_train_model_pipeline_update_stored_model(self):
        """Test happy flow of the train model pipeline"""

        train_model_pipeline(
            pj=self.pj,
            input_data=self.train_input,
            check_old_model_age=False,
            trained_models_folder="./test/trained_models",
            save_figures_folder="./test/trained_models",
        )

    def test_split_data_train_validation_test(self):
        train_data, validation_data, test_data = split_data_train_validation_test(
            self.data, period_sampling=False
        )
        self.assertEqual(len(train_data), 7345)
        self.assertEqual(len(validation_data), 1297)
        self.assertEqual(len(test_data), 1)

    # TODO move to model_selction unit test
    def test_split_data_train_validation_test_period(self):
        train_data, validation_data, test_data = split_data_train_validation_test(
            self.data, period_sampling=True
        )
        self.assertEqual(len(train_data), 7345)
        self.assertEqual(len(validation_data), 1296)
        self.assertEqual(len(test_data), 1)

    def test_train_model_pipeline_core_happy_flow(self):
        """Test happy flow of the train model pipeline

        NOTE this does not explain WHY this is the case?
        The input data should not contain features (e.g. T-7d),
        but it can/should include predictors (e.g. weather data)

        """
        model, report = train_model_pipeline_core(
            pj=self.pj, input_data=self.train_input
        )

        # check if the model was fitted (raises NotFittedError when not fitted)
        self.assertIsNone(sklearn.utils.validation.check_is_fitted(model))

        # check if model is sklearn compatible
        self.assertTrue(isinstance(model, sklearn.base.BaseEstimator))

        # check if report is a Report
        self.assertTrue(isinstance(report, Report))

    @patch("openstf.pipeline.train_model.train_model_pipeline_core")
    @patch("openstf.pipeline.train_model.PersistentStorageSerializer")
    def test_train_model_pipeline_happy_flow(self, serializer_mock, pipeline_mock):
        """Test happy flow of the train model pipeline"""

        old_model_mock = MagicMock()
        old_model_mock.age = 8

        serializer_mock_instance = MagicMock()
        serializer_mock_instance.load_model.return_value = old_model_mock
        serializer_mock.return_value = serializer_mock_instance

        report_mock = MagicMock()
        pipeline_mock.return_value = ("a", report_mock)

        train_model_pipeline(
            pj=self.pj,
            input_data=self.train_input,
            check_old_model_age=True,
            trained_models_folder="TEST",
            save_figures_folder="OTHER_TEST",
        )
        self.assertEqual(report_mock.method_calls[0].kwargs["save_path"], "OTHER_TEST")

    @patch("openstf.pipeline.train_model.train_model_pipeline_core")
    @patch("openstf.pipeline.train_model.PersistentStorageSerializer")
    def test_train_model_pipeline_young_model(self, serializer_mock, pipeline_mock):
        """Test pipeline core is not called when model is young"""
        old_model_mock = MagicMock()
        old_model_mock.age = 3

        serializer_mock_instance = MagicMock()
        serializer_mock_instance.load_model.return_value = old_model_mock
        serializer_mock.return_value = serializer_mock_instance

        report_mock = MagicMock()
        pipeline_mock.return_value = ("a", report_mock)

        train_model_pipeline(
            pj=self.pj,
            input_data=self.train_input,
            check_old_model_age=True,
            trained_models_folder="TEST",
            save_figures_folder="OTHER_TEST",
        )
        self.assertFalse(pipeline_mock.called)

    def test_feature_importance(self):
        """Test feature importance retrieval"""
        model, report = train_model_pipeline_core(
            pj=self.pj, input_data=self.train_input
        )

        # Validate and clean data
        validated_data = validation.clean(validation.validate(self.train_input))

        # Add features
        data_with_features = TrainFeatureApplicator(
            horizons=[0.25, 47.0], feature_names=self.pj["feature_names"]
        ).add_features(validated_data)

        # Split data
        train_data, validation_data, test_data = split_data_train_validation_test(
            data_with_features
        )

        # split x and y data
        train_x = train_data.iloc[:, 1:-1]

        self.assertIsInstance(
            model.get_feature_importance(train_x.columns), pd.DataFrame
        )


if __name__ == "__main__":
    unittest.main()
