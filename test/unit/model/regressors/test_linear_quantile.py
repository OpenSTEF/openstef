# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import sklearn
from sklearn.utils.estimator_checks import check_estimator

from openstef.model.regressors.linear import LinearOpenstfRegressor
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
        model = LinearQuantileOpenstfRegressor()
        model.fit(train_input.iloc[:, 1:], train_input.iloc[:, 0])

        # check if the model was fitted (raises NotFittedError when not fitted)
        self.assertIsNone(sklearn.utils.validation.check_is_fitted(model))

        # check if model is sklearn compatible
        self.assertTrue(isinstance(model, sklearn.base.BaseEstimator))

        model.predict(train_input.iloc[:, 1:])

        self.assertIsNotNone(model.feature_importances_)

    def test_imputer(self):
        n_sample = train_input.shape[0]
        X = train_input.iloc[:, 1:].copy(deep=True)
        sp = np.ones(n_sample)
        sp[-1] = np.nan
        X["Sparse"] = sp
        model1 = LinearQuantileOpenstfRegressor(imputation_strategy=None)

        with self.assertRaises(ValueError):
            model1.fit(X, train_input.iloc[:, 0])

        model2 = LinearQuantileOpenstfRegressor(imputation_strategy="mean")
        model2.fit(X, train_input.iloc[:, 0])
        self.assertIsNone(sklearn.utils.validation.check_is_fitted(model2))

        X_ = pd.DataFrame(model2.imputer_.transform(X), columns=X.columns)
        self.assertTrue((model2.predict(X_) == model2.predict(X)).all())

    def test_value_error_raised(self):
        # Check if Value Error is raised when 0.5 is not in the requested quantiles list
        with self.assertRaises(ValueError):
            LinearQuantileOpenstfRegressor((0.2, 0.3, 0.6, 0.7))

    def test_predict_raises_valueerror_no_model_trained_for_quantile(self):
        # Test if value error is raised when model is not available
        with self.assertRaises(ValueError):
            model = LinearQuantileOpenstfRegressor((0.2, 0.3, 0.5, 0.6, 0.7))
            model.predict("test_data", quantile=0.8)

    def test_importance_names(self):
        model = LinearQuantileOpenstfRegressor(tuple(self.quantiles))
        self.assertIsInstance(model._get_importance_names(), dict)

    def test_get_feature_names_from_linear(self):
        # Check if feature importance is extracted corretly
        model = LinearQuantileOpenstfRegressor(quantiles=(0.2, 0.3, 0.5, 0.6, 0.7))
        model.imputer_ = MagicMock()
        model.imputer_.in_feature_names = ["a", "b", "c"]
        model.imputer_.non_null_feature_names = ["a", "b", "c"]

        model.is_fitted_ = True
        model.models_ = {0.5: MockModel()}

        self.assertTrue(
            (
                model._get_feature_importance_from_linear(quantile=0.5)
                == np.array([1, 1, 3], dtype=np.float32)
            ).all()
        )

