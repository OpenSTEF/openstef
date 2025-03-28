# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import pickle
import unittest
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import sklearn
from sklearn.utils.estimator_checks import check_estimator

from openstef.feature_engineering.apply_features import apply_features
from openstef.model.model_creator import ModelCreator
from openstef.model.regressors.linear_quantile import LinearQuantileOpenstfRegressor

train_input = TestData.load("reference_sets/307-train-data.csv")


class MockModel:
    coef_ = np.array([1, 1, 3])


class TestLinearQuantile(BaseTestCase):
    def setUp(self) -> None:
        self.quantiles = [0.9, 0.5, 0.6, 0.1]

    @unittest.skip  # Use this during development, this test requires not allowing nan vallues which we explicitly do allow.
    def test_sklearn_compliant(self):
        # Use sklearn build in check, this will raise an exception if some check fails
        # During these tests the fit and predict methods are elaborately tested
        # More info: https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html
        check_estimator(LinearQuantileOpenstfRegressor(quantiles=tuple(self.quantiles)))

    def test_quantile_fit(self):
        """Test happy flow of the training of model"""
        # Arrange
        model = LinearQuantileOpenstfRegressor()

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
        model1 = LinearQuantileOpenstfRegressor(imputation_strategy=None)
        model2 = LinearQuantileOpenstfRegressor(
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
            LinearQuantileOpenstfRegressor((0.2, 0.3, 0.6, 0.7))

    def test_predict_raises_valueerror_no_model_trained_for_quantile(self):
        # Arrange
        model = LinearQuantileOpenstfRegressor((0.2, 0.3, 0.5, 0.6, 0.7))

        # Act
        # Test if value error is raised when model is not available
        with self.assertRaises(ValueError):
            model.predict("test_data", quantile=0.8)

    def test_importance_names(self):
        # Arrange
        model = LinearQuantileOpenstfRegressor(tuple(self.quantiles))

        # Act
        importance_names = model._get_importance_names()

        # Assert
        self.assertIsInstance(importance_names, dict)

    def test_get_feature_names_from_linear(self):
        # Arrange
        model = LinearQuantileOpenstfRegressor(quantiles=(0.2, 0.3, 0.5, 0.6, 0.7))
        model.imputer_ = MagicMock()
        model.imputer_.in_feature_names = ["a", "b", "c"]
        model.imputer_.non_null_feature_names = ["a", "b", "c"]

        model.is_fitted_ = True
        model.models_ = {0.5: MockModel()}

        # Act
        feature_importance = model._get_feature_importance_from_linear(quantile=0.5)

        # Assert
        self.assertTrue(
            (feature_importance == np.array([1, 1, 3], dtype=np.float32)).all()
        )

    def test_ignore_features(self):
        # Arrange
        model = LinearQuantileOpenstfRegressor(quantiles=(0.2, 0.3, 0.5, 0.6, 0.7))

        input_data_engineered = apply_features(train_input)

        # Add non-feature-engineerd columns that should be ignored
        input_data_engineered["E1B_AMI_I"] = 1
        input_data_engineered["E4A_I"] = 1

        self.assertIn("T-1d", input_data_engineered.columns)
        self.assertIn("is_eerste_kerstdag", input_data_engineered.columns)
        self.assertIn("IsWeekDay", input_data_engineered.columns)
        self.assertIn("load", input_data_engineered.columns)
        self.assertIn("E1B_AMI_I", input_data_engineered.columns)
        self.assertIn("E4A_I", input_data_engineered.columns)

        # Act
        input_data_filtered = model._remove_ignored_features(input_data_engineered)

        # Assert
        self.assertNotIn("T-1d", input_data_filtered.columns)
        self.assertNotIn("is_eerste_kerstdag", input_data_filtered.columns)
        self.assertNotIn("IsWeekDay", input_data_filtered.columns)
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
            model_type="linear_quantile",
            quantiles=[0.5],
            **kwargs,
        )

        # Assert
        self.assertIsInstance(model, LinearQuantileOpenstfRegressor)
        self.assertEqual(model.weight_scale_percentile, 50)
        self.assertEqual(model.weight_exponent, 2)

    def test_feature_clipper(self):
        """Test the feature clipping functionality of LinearQuantileOpenstfRegressor"""
        # Create a sample dataset with a feature to be clipped
        X = pd.DataFrame(
            {
                "A": [1.0, 2.0, 3.0, 4.0, 5.0],
                "B": [10.0, 20.0, 30.0, 40.0, 50.0],
                "day_ahead_electricity_price": [
                    -10.0,
                    0.0,
                    50.0,
                    100.0,
                    200.0,
                ],  # Feature to be clipped
            }
        )
        y = pd.Series([1, 2, 3, 4, 5])

        # Initialize the model with clipping for 'day_ahead_electricity_price' feature
        model = LinearQuantileOpenstfRegressor(
            clipped_features=["day_ahead_electricity_price"]
        )

        # Fit the model
        model.fit(X, y)

        # Create test data with values outside the training range
        X_test = pd.DataFrame(
            {
                "A": [2.5, 3.5],
                "B": [25.0, 35.0],
                "day_ahead_electricity_price": [
                    -20.0,
                    250.0,
                ],  # Values outside the training range
            }
        )

        # Check if the 'day_ahead_electricity_price' feature was clipped during prediction
        clipped_X = model.feature_clipper_.transform(X_test)
        self.assertTrue(clipped_X["day_ahead_electricity_price"].min() >= -10.0)
        self.assertTrue(clipped_X["day_ahead_electricity_price"].max() <= 200.0)

        # Make predictions
        y_pred = model.predict(X_test)

        # Ensure the prediction was made using the clipped values
        self.assertEqual(len(y_pred), 2)
        self.assertIsNotNone(y_pred)
