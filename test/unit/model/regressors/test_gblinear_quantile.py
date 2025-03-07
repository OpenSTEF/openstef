# SPDX-FileCopyrightText: 2017-2025 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import sklearn
from xgboost import Booster

from openstef.feature_engineering.apply_features import apply_features
from openstef.model.model_creator import ModelCreator
from openstef.model.regressors.gblinear_quantile import GBLinearQuantileOpenstfRegressor

train_input: pd.DataFrame = TestData.load("reference_sets/307-train-data.csv")


class MockBooster:
    feature_names = ["a", "b", "c"]

    def get_score(self, importance_type):
        return {"a": -0.1, "b": 0.5, "c": 0.6}


class TestGBLinearQuantile(BaseTestCase):
    def setUp(self) -> None:
        self.quantiles = [0.9, 0.5, 0.6, 0.1]

    def test_quantile_fit(self):
        """Test happy flow of the training of model"""
        # Arrange
        model = GBLinearQuantileOpenstfRegressor()

        # Act
        model.fit(train_input.iloc[:, 1:], train_input.iloc[:, 0])

        # Assert
        # check if the model was fitted (raises NotFittedError when not fitted)
        self.assertIsNone(sklearn.utils.validation.check_is_fitted(model))

        # check if model is sklearn compatible
        self.assertIsInstance(model, sklearn.base.BaseEstimator)

        model.predict(train_input.iloc[:, 1:])

        self.assertIsNotNone(model.feature_importances_)

    def test_imputer(self):
        # Arrange
        n_sample = train_input.shape[0]
        X = train_input.iloc[:, 1:].copy(deep=True)
        X["sparse"] = np.ones(n_sample)
        X.loc[X.index[-2], "sparse"] = np.nan
        X["sparse_2"] = np.ones(n_sample)
        X.loc[X.index[-1], "sparse_2"] = np.nan
        model1 = GBLinearQuantileOpenstfRegressor(imputation_strategy=None)
        model2 = GBLinearQuantileOpenstfRegressor(
            imputation_strategy="mean", no_fill_future_values_features=["sparse_2"]
        )

        # Act
        # Model should give error if nan values are present.
        with self.assertRaises(ValueError):
            model1.fit(X, train_input.iloc[:, 0])

        # Model should fill in the nans
        model2.fit(X, train_input.iloc[:, 0])

        # Assert
        self.assertIsNone(sklearn.utils.validation.check_is_fitted(model2))

        X_ = pd.DataFrame(model2.imputer_.transform(X), columns=X.columns)
        self.assertTrue((model2.predict(X_) == model2.predict(X)).all())

        # check if last row is removed because of trailing null values
        X_transformed, _ = model2.imputer_.fit_transform(X)
        self.assertEqual(X_transformed.shape[0], n_sample - 1)

    def test_value_error_raised(self):
        # Check if Value Error is raised when 0.5 is not in the requested quantiles list
        with self.assertRaises(ValueError):
            GBLinearQuantileOpenstfRegressor((0.2, 0.3, 0.6, 0.7))

    def test_predict_raises_valueerror_no_model_trained_for_quantile(self):
        # Arrange
        model = GBLinearQuantileOpenstfRegressor((0.2, 0.3, 0.5, 0.6, 0.7))

        # Act
        # Test if value error is raised when model is not available
        with self.assertRaises(ValueError):
            model.predict("test_data", quantile=0.8)

    def test_importance_names(self):
        # Arrange
        model = GBLinearQuantileOpenstfRegressor(tuple(self.quantiles))

        # Act
        importance_names = model._get_importance_names()

        # Assert
        self.assertIsInstance(importance_names, dict)

    def test_get_feature_importance_from_gblinear(self):
        # Arrange
        model = GBLinearQuantileOpenstfRegressor(quantiles=(0.2, 0.3, 0.5, 0.6, 0.7))
        model.imputer_ = MagicMock()
        model.imputer_.in_feature_names = ["a", "b", "c"]
        model.imputer_.non_null_feature_names = ["a", "b", "c"]

        model.is_fitted_ = True
        model.model_ = MockBooster()

        # Act
        feature_importance = model._get_feature_importances_from_booster(model.model_)
        abs_score = np.abs(list(model.model_.get_score("weight").values()))
        expected_feature_importance = abs_score / abs_score.sum()

        # Assert
        np.testing.assert_array_almost_equal(
            feature_importance, expected_feature_importance
        )

    def test_ignore_features(self):
        # Arrange
        model = GBLinearQuantileOpenstfRegressor(quantiles=(0.2, 0.3, 0.5, 0.6, 0.7))

        input_data_engineered = apply_features(train_input)

        # Add non-feature-engineerd columns that should be ignored
        input_data_engineered["E1B_AMI_I"] = 1
        input_data_engineered["E4A_I"] = 1

        for feat in model.TO_KEEP_FEATURES:
            self.assertIn(feat, input_data_engineered.columns)
        self.assertIn("T-7d", input_data_engineered.columns)
        self.assertIn("is_eerste_kerstdag", input_data_engineered.columns)
        self.assertIn("Month", input_data_engineered.columns)
        self.assertIn("load", input_data_engineered.columns)
        self.assertIn("E1B_AMI_I", input_data_engineered.columns)
        self.assertIn("E4A_I", input_data_engineered.columns)

        # Act
        input_data_filtered = model._remove_ignored_features(input_data_engineered)

        # Assert
        for feat in model.TO_KEEP_FEATURES:
            self.assertIn(feat, input_data_filtered.columns)
        self.assertNotIn("is_eerste_kerstdag", input_data_filtered.columns)
        self.assertNotIn("Month", input_data_filtered.columns)
        self.assertNotIn("E1B_AMI_I", input_data_filtered.columns)
        self.assertNotIn("E4A_I", input_data_filtered.columns)
        self.assertIn("load", input_data_filtered.columns)

    def test_create_model(self):
        # Arrange
        kwargs = {
            "weight_scale_percentile": 50,
            "weight_exponent": 2,
        }

        # Act
        model = ModelCreator.create_model(
            model_type="gblinear_quantile",
            quantiles=[0.5],
            **kwargs,
        )

        # Assert
        self.assertIsInstance(model, GBLinearQuantileOpenstfRegressor)
        self.assertEqual(model.weight_scale_percentile, 50)
        self.assertEqual(model.weight_exponent, 2)

    def test_feature_clipper(self):
        """Test the feature clipping functionality of LinearQuantileOpenstfRegressor"""
        # Create a sample dataset with a feature to be clipped
        X = pd.DataFrame(
            {
                "A": [1.0, 2.0, 3.0, 4.0, 5.0],
                "B": [10.0, 20.0, 30.0, 40.0, 50.0],
                "APX": [-10.0, 0.0, 50.0, 100.0, 200.0],  # Feature to be clipped
            }
        )
        y = pd.Series([1, 2, 3, 4, 5])

        # Initialize the model with clipping for 'APX' feature
        model = GBLinearQuantileOpenstfRegressor(clipped_features=["APX"])

        # Fit the model
        model.fit(X, y)

        # Create test data with values outside the training range
        X_test = pd.DataFrame(
            {
                "A": [2.5, 3.5],
                "B": [25.0, 35.0],
                "APX": [-20.0, 250.0],  # Values outside the training range
            }
        )

        # Check if the 'APX' feature was clipped during prediction
        clipped_X = model.feature_clipper_.transform(X_test)
        self.assertTrue(clipped_X["APX"].min() >= -10.0)
        self.assertTrue(clipped_X["APX"].max() <= 200.0)

        # Make predictions
        y_pred = model.predict(X_test)

        # Ensure the prediction was made using the clipped values
        self.assertEqual(len(y_pred), 2)
        self.assertIsNotNone(y_pred)
