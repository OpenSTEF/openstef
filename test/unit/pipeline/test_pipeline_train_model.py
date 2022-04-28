# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import copy
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
from openstef.model.objective import RegressorObjective
from openstef.model.regressors.custom_regressor import CustomOpenstfRegressor
from openstef.model_selection.model_selection import split_data_train_validation_test
from openstef.pipeline.train_model import (
    train_model_pipeline,
    train_model_pipeline_core,
    train_pipeline_common,
)
from openstef.validation import validation


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

    @property
    def can_predict_quantiles(self):
        return False

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


class TestTrainModelPipeline(BaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.pj, self.model_specs = TestData.get_prediction_job_and_modelspecs(pid=307)
        # Set n_estimators to a small number to speed up training
        self.model_specs.hyper_params["n_estimators"] = 3
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
            mlflow_tracking_uri="./test/unit/trained_models/mlruns",
            artifact_folder="./test/unit/trained_models",
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
                model_specs = self.model_specs
                train_input = self.train_input

                # Use default parameters
                model_specs.hyper_params = {}
                model_specs.hyper_params["max_epochs"] = 1

                # For Linear model we need to choose an imputation strategy to handle missing value
                if model_type == MLModelType.LINEAR:
                    model_specs.hyper_params["imputation_strategy"] = "mean"

                model, report, modelspecs, _ = train_model_pipeline_core(
                    pj=pj, model_specs=model_specs, input_data=train_input
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
                validated_data = validation.drop_target_na(
                    validation.validate(pj["id"], train_input, flatliner_threshold=24)
                )

                # Add features
                data_with_features = TrainFeatureApplicator(
                    horizons=[0.25, 47.0], feature_names=model_specs.feature_names
                ).add_features(validated_data, pj=pj)

                # Split data
                (
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
        model_specs = self.model_specs
        model_specs.hyper_params = {}
        model_specs.feature_modules = [
            "test.unit.feature_engineering.test_feature_adder"
        ]
        dummy_feature = "dummy_0.5"
        model_specs.feature_names.append(dummy_feature)
        pj.default_modelspecs = model_specs

        train_input = self.train_input.iloc[::50, :]
        model, report, modelspecs, _ = train_model_pipeline_core(
            pj=pj, model_specs=model_specs, input_data=train_input
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

    @patch("openstef.pipeline.train_model.MLflowSerializer")
    def test_train_model_pipeline_with_default_modelspecs(self, mock_serializer):
        """We check that the modelspecs object given as default in the prediction job
        is the one given to save_model when there is no previous model saved for the
        prediction job.
        """
        mock_serializer_instance = MagicMock()
        # Mimick the absence of older model.
        mock_serializer_instance.load_model.side_effect = FileNotFoundError()
        mock_serializer.return_value = mock_serializer_instance

        pj = copy.deepcopy(self.pj)

        # hyper params that are different from the defaults.
        xgb_hyper_params = {
            "subsample": 0.9,
            "min_child_weight": 4,
            "max_depth": 8,
            "gamma": 0.5,
            "colsample_bytree": 0.85,
            "eta": 0.1,
            "training_period_days": 90,
        }
        new_hyper_params = {
            key: (value + 0.01) if isinstance(value, float) else value + 1
            for key, value in xgb_hyper_params.items()
        }

        model_specs = copy.deepcopy(self.model_specs)
        model_specs.hyper_params = new_hyper_params

        # Custom features
        model_specs.feature_modules = [
            "test.unit.feature_engineering.test_feature_adder"
        ]
        model_specs.feature_names.append("dummy_0.5")

        pj.default_modelspecs = model_specs

        train_model_pipeline(
            pj=pj,
            input_data=self.train_input,
            check_old_model_age=True,
            mlflow_tracking_uri="./test/unit/trained_models/mlruns",
            artifact_folder="./test/unit/trained_models",
        )

        saved_model_specs = mock_serializer_instance.save_model.call_args.kwargs[
            "model_specs"
        ]
        self.assertEqual(saved_model_specs, model_specs)

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
            self.model_specs,
        )
        serializer_mock.return_value = serializer_mock_instance

        report_mock = MagicMock()
        pipeline_mock.return_value = (
            "a",
            report_mock,
            self.model_specs,
            (None, None, None),
        )

        train_model_pipeline(
            pj=self.pj,
            input_data=self.train_input,
            check_old_model_age=False,
            mlflow_tracking_uri="./test/unit/trained_models/mlruns",
            artifact_folder="./test/unit/trained_models",
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
            self.model_specs,
        )
        serializer_mock.return_value = serializer_mock_instance

        report_mock = MagicMock()
        pipeline_mock.return_value = ("a", report_mock)

        train_model_pipeline(
            pj=self.pj,
            input_data=self.train_input,
            check_old_model_age=True,
            mlflow_tracking_uri="./test/unit/trained_models/mlruns",
            artifact_folder="./test/unit/trained_models",
        )
        self.assertFalse(pipeline_mock.called)

    @patch("openstef.model.serializer.MLflowSerializer.save_model")
    @patch("openstef.validation.validation.is_data_sufficient", return_value=False)
    def test_train_model_InputDataInsufficientError(
        self, validation_is_data_sufficient_mock, save_model_mock
    ):
        # This error is caught and then raised again and logged

        with self.assertRaises(InputDataInsufficientError):
            train_model_pipeline(
                pj=self.pj,
                input_data=self.train_input,
                check_old_model_age=False,
                mlflow_tracking_uri="./test/unit/trained_models/mlruns",
                artifact_folder="./test/unit/trained_models",
            )

    @patch("openstef.model.serializer.MLflowSerializer.save_model")
    def test_train_model_InputDataWrongColumnOrderError(self, save_model_mock):
        # change the column order
        input_data = self.train_input.iloc[:, ::-1]

        with self.assertRaises(InputDataWrongColumnOrderError):
            train_model_pipeline(
                pj=self.pj,
                input_data=input_data,
                check_old_model_age=False,
                mlflow_tracking_uri="./test/unit/trained_models/mlruns",
                artifact_folder="./test/unit/trained_models",
            )

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
            self.model_specs,
        )
        serializer_mock.return_value = serializer_mock_instance
        old_model_mock.score.return_value = 5

        result = train_model_pipeline(
            pj=self.pj,
            input_data=self.train_input,
            check_old_model_age=True,
            mlflow_tracking_uri="./test/unit/trained_models/mlruns",
            artifact_folder="./test/unit/trained_models",
        )
        self.assertIsNone(result)
        self.assertEqual(len(serializer_mock_instance.method_calls), 1)

    @patch("openstef.model.serializer.MLflowSerializer.save_model")
    @patch("openstef.pipeline.train_model.MLflowSerializer")
    def test_train_model_log_new_model_better(self, serializer_mock, save_model_mock):
        # Mock an old model which is better than the new one.
        old_model_mock = MagicMock()
        old_model_mock.age = 8

        serializer_mock_instance = MagicMock()
        serializer_mock_instance.load_model.return_value = (
            old_model_mock,
            self.model_specs,
        )
        serializer_mock.return_value = serializer_mock_instance
        old_model_mock.score.return_value = 0.1

        result = train_model_pipeline(
            pj=self.pj,
            input_data=self.train_input,
            check_old_model_age=True,
            mlflow_tracking_uri="./test/unit/trained_models/mlruns",
            artifact_folder="./test/unit/trained_models",
        )
        self.assertIsNone(result)
        self.assertEqual(len(serializer_mock_instance.method_calls), 3)

    @patch("openstef.model.serializer.MLflowSerializer.save_model")
    @patch("openstef.pipeline.train_model.MLflowSerializer")
    def test_train_model_log_couldnt_compare(self, serializer_mock, save_model_mock):
        # Mock an old model which is better than the new one.
        old_model_mock = MagicMock()
        old_model_mock.age = 8

        serializer_mock_instance = MagicMock()
        serializer_mock_instance.load_model.return_value = (
            old_model_mock,
            self.model_specs,
        )
        serializer_mock.return_value = serializer_mock_instance
        old_model_mock.score.side_effect = ValueError()

        result = train_model_pipeline(
            pj=self.pj,
            input_data=self.train_input,
            check_old_model_age=True,
            mlflow_tracking_uri="./test/unit/trained_models/mlruns",
            artifact_folder="./test/unit/trained_models",
        )
        self.assertIsNone(result)
        self.assertEqual(len(serializer_mock_instance.method_calls), 3)

    @patch("openstef.model.serializer.MLflowSerializer.save_model")
    @patch("openstef.pipeline.train_model.MLflowSerializer")
    def test_train_model_No_old_model(self, serializer_mock, save_model_mock):
        # Mock an old model which is better than the new one.
        old_model_mock = MagicMock()
        old_model_mock.age = 8

        serializer_mock_instance = MagicMock()
        serializer_mock_instance.load_model.return_value = (
            old_model_mock,
            self.model_specs,
        )
        serializer_mock_instance.load_model.side_effect = FileNotFoundError()
        serializer_mock.return_value = serializer_mock_instance

        train_model_pipeline(
            pj=self.pj,
            input_data=self.train_input,
            check_old_model_age=True,
            mlflow_tracking_uri="./test/unit/trained_models/mlruns",
            artifact_folder="./test/unit/trained_models",
        )
        self.assertEqual(len(serializer_mock_instance.method_calls), 3)

    def test_train_pipeline_common_different_quantiles_with_quantile_regressor(self):
        """Incorporated after a bug.
        Test if PJ has different quantiles compared to old model, for quantile regressor
        those in the PJ should be used.

        NB: notice how we don't need to mock anything for testing this function!"""

        old_quantiles = (0.4, 0.5, 0.6)
        desired_quantiles = (0.01, 0.5, 0.74)

        # use a quantile prediction_job
        pj = self.pj
        pj["id"] = 460
        pj["model"] = "xgb_quantile"
        pj["quantiles"] = desired_quantiles

        # Change 'old' model_specs
        modified_model_specs = self.model_specs
        modified_model_specs["hyper_params"].update(dict(quantiles=old_quantiles))

        # train model
        model, report, train_data, validation_data, test_data = train_pipeline_common(
            pj, modified_model_specs, self.train_input, horizons=[0.25, 47.0]
        )

        # check quantiles
        self.assertListEqual(list(model.estimators_.keys()), list(desired_quantiles))

    def test_train_pipeline_common_with_missing_custom_horizon(self):
        with self.assertRaises(ValueError):
            (
                model,
                report,
                train_data,
                validation_data,
                test_data,
            ) = train_pipeline_common(
                self.pj, self.model_specs, self.train_input, horizons="custom_horizon"
            )

    @patch("openstef.pipeline.train_model.MLflowSerializer")
    def test_train_model_pipeline_with_save_train_forecasts(self, mock_serializer):
        """We check that the modelspecs object given as default in the prediction job
        is the one given to save_model when there is no previous model saved for the
        prediction job.
        """

        mock_serializer_instance = MagicMock()
        # Mimick the absence of older model.
        mock_serializer_instance.load_model.side_effect = FileNotFoundError()
        mock_serializer.return_value = mock_serializer_instance

        pj = copy.deepcopy(self.pj)
        # hyper params that are different from the defaults.

        datasets = train_model_pipeline(
            pj=pj,
            input_data=self.train_input,
            check_old_model_age=True,
            mlflow_tracking_uri="./test/unit/trained_models/mlruns",
            artifact_folder="./test/unit/trained_models",
        )

        self.assertIsNone(datasets)

        pj.save_train_forecasts = True

        datasets = train_model_pipeline(
            pj=pj,
            input_data=self.train_input,
            check_old_model_age=True,
            mlflow_tracking_uri="./test/unit/trained_models/mlruns",
            artifact_folder="./test/unit/trained_models",
        )
        self.assertIsNotNone(datasets)

        for dataset in datasets:
            self.assertIn("forecast", dataset.columns)


if __name__ == "__main__":
    unittest.main()
