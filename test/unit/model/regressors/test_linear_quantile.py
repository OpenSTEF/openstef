# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import sklearn
from sklearn.utils.estimator_checks import check_estimator

from openstef.feature_engineering.apply_features import apply_features
from openstef.model.regressors.linear_quantile import LinearQuantileOpenstfRegressor
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData

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
        self.assertTrue(isinstance(model, sklearn.base.BaseEstimator))

        model.predict(train_input.iloc[:, 1:])

        self.assertIsNotNone(model.feature_importances_)

    def test_imputer(self):
        # Arrange
        n_sample = train_input.shape[0]
        X = train_input.iloc[:, 1:].copy(deep=True)
        sp = np.ones(n_sample)
        sp[-1] = np.nan
        X["Sparse"] = sp
        model1 = LinearQuantileOpenstfRegressor(imputation_strategy=None)
        model2 = LinearQuantileOpenstfRegressor(imputation_strategy="mean")

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
            (
                feature_importance == np.array([1, 1, 3], dtype=np.float32)
            ).all()
        )

    def test_ignore_features(self):
        # Arrange
        model = LinearQuantileOpenstfRegressor(quantiles=(0.2, 0.3, 0.5, 0.6, 0.7))

        input_data_engineered = apply_features(train_input)

        self.assertIn('T-1d', input_data_engineered.columns)
        self.assertIn('is_eerste_kerstdag', input_data_engineered.columns)
        self.assertIn('IsWeekDay', input_data_engineered.columns)
        self.assertIn('load', input_data_engineered.columns)

        # Act
        input_data_filtered = model._remove_ignored_features(input_data_engineered)

        # Assert
        self.assertNotIn('T-1d', input_data_filtered.columns)
        self.assertNotIn('is_eerste_kerstdag', input_data_filtered.columns)
        self.assertNotIn('IsWeekDay', input_data_filtered.columns)
        self.assertIn('load', input_data_filtered.columns)

