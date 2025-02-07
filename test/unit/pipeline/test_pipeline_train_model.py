# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import copy
import glob
import os
import unittest
from datetime import datetime, timedelta, UTC
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData
from unittest.mock import MagicMock, patch

import pandas as pd
import sklearn

from openstef.data_classes.data_prep import DataPrepDataClass
from openstef.data_classes.split_function import SplitFuncDataClass
from openstef.enums import ModelType
from openstef.exceptions import (
    InputDataInsufficientError,
    InputDataWrongColumnOrderError,
    SkipSaveTrainingForecasts,
)
from openstef.feature_engineering.data_preparation import (
    ARDataPreparation,
    LegacyDataPreparation,
)
from openstef.feature_engineering.feature_applicator import TrainFeatureApplicator
from openstef.metrics.reporter import Report
from openstef.model.objective import RegressorObjective
from openstef.model.regressors.custom_regressor import CustomOpenstfRegressor
from openstef.model_selection.model_selection import split_data_train_validation_test
from openstef.pipeline.train_model import (
    DEFAULT_TRAIN_HORIZONS_HOURS,
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

    @staticmethod
    def objective():
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


def split_dummy_arima(data, test_fraction):
    return data.iloc[:-5], data.iloc[-10:-5], data.iloc[-5:], data.iloc[-5:]


class TestTrainModelPipeline(BaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.pj, self.model_specs = TestData.get_prediction_job_and_modelspecs(pid=307)
        # Set n_estimators to a small number to speed up training
        self.model_specs.hyper_params["n_estimators"] = 3
        datetime_start = datetime.now(tz=UTC) - timedelta(days=90)
        datetime_end = datetime.now(tz=UTC)
        self.data_table = TestData.load("input_data_train.csv").head(8641)
        self.data = pd.DataFrame(
            index=pd.date_range(datetime_start, datetime_end, freq="15T")
        )

        self.train_input = TestData.load("reference_sets/307-train-data.csv")

    @unittest.skip("Do this test if we want to train a new model")
    def test_train_model_pipeline_update_stored_model(self):
        """Test happy flow of the train model pipeline

        Other manual steps;
        - Identify the artifact forlder of the old model test/unit/trained_models/mlruns/0/d7719d5d316d4416a947e4f7ea7e73a8/artifacts/model
        - Rename the new artifact folder to that name
        - Update references to the artifact location in the new ../d7.../meta.yaml
        - Remove the old folder
        """

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
        # Remove modeltypes which are optional, and add a dummy regressor
        for model_type in list(ModelType) + [__name__ + ".DummyRegressor"]:
            with self.subTest(model_type=model_type):
                pj = self.pj

                pj["model"] = (
                    model_type.value if hasattr(model_type, "value") else model_type
                )
                model_specs = self.model_specs

                # Select 150 data points to speedup test
                train_input = self.train_input.iloc[:150, :]

                # Use default parameters
                model_specs.hyper_params = {}
                model_specs.hyper_params["max_epochs"] = 1

                # For Linear model we need to choose an imputation strategy to handle missing value
                if model_type == ModelType.LINEAR:
                    model_specs.hyper_params["imputation_strategy"] = "mean"

                if model_type == ModelType.ARIMA:
                    pj.data_prep_class = DataPrepDataClass(
                        klass=ARDataPreparation,
                        arguments={},
                    )
                    pj.train_split_func = SplitFuncDataClass(
                        function=split_dummy_arima,
                        arguments={},
                    )

                model, report, modelspecs, _ = train_model_pipeline_core(
                    pj=pj, model_specs=model_specs, input_data=train_input
                )

                # check if the model was fitted (raises NotFittedError when not fitted)
                self.assertIsNone(sklearn.utils.validation.check_is_fitted(model))

                # check if the model has a feature_names property
                self.assertIsNotNone(model.feature_names)

                # check if model is sklearn compatible
                self.assertIsInstance(model, sklearn.base.BaseEstimator)

                # check if report is a Report
                self.assertIsInstance(report, Report)

                # Validate and clean data
                validated_data = validation.drop_target_na(
                    validation.validate(
                        pj["id"],
                        train_input,
                        flatliner_threshold_minutes=360,
                        resolution_minutes=15,
                    )
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
                    operational_score_data,
                ) = split_data_train_validation_test(data_with_features)

                importance = model.set_feature_importance()
                self.assertIsInstance(importance, pd.DataFrame)

    def test_train_model_pipeline_core_happy_flow_with_legacy_data_prep(self):
        """Test happy flow of the train model pipeline with the legacy data prep class."""
        # Select 50 data points to speedup test
        train_input = self.train_input.iloc[::50, :]
        for model_type in list(ModelType) + [__name__ + ".DummyRegressor"]:
            with self.subTest(model_type=model_type):
                # Skip the arima model because it does not use legacy data prep
                if model_type == ModelType.ARIMA:
                    continue
                pj = self.pj
                pj.data_prep_class = DataPrepDataClass(
                    klass=LegacyDataPreparation,
                    arguments={},
                )
                pj["model"] = (
                    model_type.value if hasattr(model_type, "value") else model_type
                )
                model_specs = self.model_specs
                train_input = self.train_input

                # Use default parameters
                model_specs.hyper_params = {}
                model_specs.hyper_params["max_epochs"] = 1

                # For Linear model we need to choose an imputation strategy to handle missing value
                if model_type == ModelType.LINEAR:
                    model_specs.hyper_params["imputation_strategy"] = "mean"

                model, report, modelspecs, _ = train_model_pipeline_core(
                    pj=pj, model_specs=model_specs, input_data=train_input
                )

                # check if the model was fitted (raises NotFittedError when not fitted)
                self.assertIsNone(sklearn.utils.validation.check_is_fitted(model))

                # check if the model has a feature_names property
                self.assertIsNotNone(model.feature_names)

                # check if model is sklearn compatible
                self.assertIsInstance(model, sklearn.base.BaseEstimator)

                # check if report is a Report
                self.assertIsInstance(report, Report)

                # Validate and clean data
                validated_data = validation.drop_target_na(
                    validation.validate(
                        pj["id"],
                        train_input,
                        flatliner_threshold_minutes=360,
                        resolution_minutes=15,
                    )
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
                    operational_score_data,
                ) = split_data_train_validation_test(data_with_features)

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
        model, report, _, _ = train_model_pipeline_core(
            pj=pj, model_specs=model_specs, input_data=train_input
        )

        # check if the model was fitted (raises NotFittedError when not fitted)
        self.assertIsNone(sklearn.utils.validation.check_is_fitted(model))

        # check if the model has a feature_names property
        self.assertIsNotNone(model.feature_names)
        self.assertIn(dummy_feature, model.feature_names)

        # check if model is sklearn compatible
        self.assertIsInstance(model, sklearn.base.BaseEstimator)

        # check if report is a Report
        self.assertIsInstance(report, Report)

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

        # Assert the report was attempted to be written to the correct location
        assert report_mock.method_calls[0].args[0] == os.path.join(
            "./test/unit/trained_models", "307", "weight_plot.html"
        )
        # Assert the figure is in the correct location
        found_files = [
            os.path.basename(file_with_path)
            for file_with_path in glob.glob(
                os.path.join("./test/unit/trained_models/307/*.html")
            )
        ]
        excepted_fnames = [
            "Predictor0.25.html",
            "Predictor47.0.html",
            "weight_plot.html",
        ]
        assert set(found_files) == set(excepted_fnames)

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

        result = train_model_pipeline(
            pj=self.pj,
            input_data=self.train_input,
            check_old_model_age=True,
            mlflow_tracking_uri="./test/unit/trained_models/mlruns",
            artifact_folder="./test/unit/trained_models",
        )
        self.assertIsNone(result)
        self.assertFalse(pipeline_mock.called)

    @patch("openstef.model.serializer.MLflowSerializer.save_model")
    @patch("openstef.pipeline.train_model.train_model_pipeline_core")
    @patch("openstef.pipeline.train_model.MLflowSerializer")
    def test_train_model_pipeline_young_model_save_forecasts(
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

        pj = self.pj
        pj.save_train_forecasts = True

        with self.assertRaises(SkipSaveTrainingForecasts):
            train_model_pipeline(
                pj=pj,
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
    def test_train_model_OldModelHigherScoreError_save_forecast(
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

        pj = self.pj
        pj.save_train_forecasts = True

        with self.assertRaises(SkipSaveTrainingForecasts):
            train_model_pipeline(
                pj=pj,
                input_data=self.train_input,
                check_old_model_age=True,
                mlflow_tracking_uri="./test/unit/trained_models/mlruns",
                artifact_folder="./test/unit/trained_models",
            )
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
        (
            model,
            report,
            train_data,
            validation_data,
            test_data,
            operational_score_data,
        ) = train_pipeline_common(
            pj, modified_model_specs, self.train_input, horizons=[0.25, 47.0]
        )

        # check quantiles
        self.assertListEqual(list(model.estimators_.keys()), list(desired_quantiles))
        # check the score method works as expected
        combined = pd.concat([train_data, validation_data]).reset_index(drop=True)
        x_data, y_data = (
            combined.iloc[:, 1:-1],
            combined.iloc[:, 0],
        )
        model.score(x_data, y_data)

    def test_train_pipeline_common_with_missing_custom_horizon(self):
        with self.assertRaises(ValueError):
            (
                model,
                report,
                train_data,
                validation_data,
                test_data,
                operational_score_data,
            ) = train_pipeline_common(
                self.pj, self.model_specs, self.train_input, horizons="custom_horizon"
            )

    @patch("openstef.pipeline.train_model.MLflowSerializer")
    @patch("openstef.pipeline.train_model.train_model_pipeline_core")
    def test_train_model_pipeline_with_default_train_horizons(
        self, mock_train_model_pipeline_core, mock_serializer
    ):
        # Arrange
        mock_train_model_pipeline_core.return_value = "a", "b", "c", "d"

        # Act
        train_model_pipeline(
            pj=self.pj,
            input_data=self.train_input,
            check_old_model_age=False,
            mlflow_tracking_uri="./test/unit/trained_models/mlruns",
            artifact_folder=None,
        )

        # Assert
        self.pj.train_horizons_minutes == None
        assert (
            mock_train_model_pipeline_core.call_args.kwargs["horizons"]
            == DEFAULT_TRAIN_HORIZONS_HOURS
        )

    @patch("openstef.pipeline.train_model.MLflowSerializer")
    @patch("openstef.pipeline.train_model.train_model_pipeline_core")
    def test_train_model_pipeline_with_custom_train_horizons(
        self, mock_train_model_pipeline_core, mock_serializer
    ):
        # Arrange
        mock_train_model_pipeline_core.return_value = "a", "b", "c", "d"
        self.pj.train_horizons_minutes = [1440, 21600]
        train_horizons_hours = [24, 360]

        # Act
        train_model_pipeline(
            pj=self.pj,
            input_data=self.train_input,
            check_old_model_age=False,
            mlflow_tracking_uri="./test/unit/trained_models/mlruns",
            artifact_folder=None,
        )

        # Assert
        assert (
            mock_train_model_pipeline_core.call_args.kwargs["horizons"]
            == train_horizons_hours
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
