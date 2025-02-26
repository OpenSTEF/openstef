# SPDX-FileCopyrightText: 2017-2023 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData

import numpy as np
import pandas as pd
import sklearn
from sklearn.utils.estimator_checks import check_estimator

from openstef.model.regressors.linear import LinearOpenstfRegressor


class TestLinearOpenstfRegressor(BaseTestCase):
    def setUp(self):
        self.train_input = TestData.load("reference_sets/307-train-data.csv")

    def test_sklearn_compliant(self):
        # Use sklearn build in check, this will raise an exception if some check fails
        # During these tests the fit and predict methods are elaborately tested
        # More info: https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html
        check_estimator(LinearOpenstfRegressor())

    def test_fit(self):
        """Test happy flow of the training of model"""
        model = LinearOpenstfRegressor()
        model.fit(self.train_input.iloc[:, 1:], self.train_input.iloc[:, 0])

        # check if the model was fitted (raises NotFittedError when not fitted)
        self.assertIsNone(sklearn.utils.validation.check_is_fitted(model))

        # check if model is sklearn compatible
        self.assertIsInstance(model, sklearn.base.BaseEstimator)

    def test_non_null_columns_retrieval(self):
        n_sample = self.train_input.shape[0]
        columns = self.train_input.columns[1:]

        X = self.train_input.iloc[:, 1:].copy(deep=True)
        X["Empty"] = pd.Series(n_sample * [np.nan])

        model = LinearOpenstfRegressor()
        model.fit(X, self.train_input.iloc[:, 0])

        # check if non null columns are well retrieved
        self.assertTrue((columns == model.non_null_columns_).all())

        X_ = X.copy(deep=True)
        X_["Empty2"] = pd.Series(n_sample * [np.nan])

        # check the use of non null columns in predict
        self.assertTrue((model.predict(X_) == model.predict(X)).all())
        self.assertTrue(
            (model.predict(self.train_input.iloc[:, 1:]) == model.predict(X)).all()
        )

    def test_imputer(self):
        n_sample = self.train_input.shape[0]
        X = self.train_input.iloc[:, 1:].copy(deep=True)
        sp = np.ones(n_sample)
        sp[-1] = np.nan
        X["Sparse"] = sp
        model1 = LinearOpenstfRegressor(imputation_strategy=None)

        with self.assertRaises(ValueError):
            model1.fit(X, self.train_input.iloc[:, 0])

        model2 = LinearOpenstfRegressor(imputation_strategy="mean")
        model2.fit(X, self.train_input.iloc[:, 0])
        self.assertIsNone(sklearn.utils.validation.check_is_fitted(model2))

        X_ = pd.DataFrame(model2.imputer_.transform(X), columns=X.columns)
        self.assertTrue((model2.predict(X_) == model2.predict(X)).all())

    def test_get_feature_importance_from_linear(self):
        model = LinearOpenstfRegressor()
        model.fit(self.train_input.iloc[:, 1:], self.train_input.iloc[:, 0])
        features_in = list(self.train_input.columns[1:])

        feature_importance_linear = np.abs(model.regressor_.coef_)
        feature_importance_model = np.array(
            [
                x
                for name, x in zip(
                    features_in, model._get_feature_importance_from_linear()
                )
                if name in model.non_null_columns_
            ]
        )
        feature_importance_null = np.array(
            [
                x
                for name, x in zip(
                    features_in, model._get_feature_importance_from_linear()
                )
                if not (name in model.non_null_columns_)
            ]
        )

        # check the retrieval of feature importance
        self.assertTrue((feature_importance_linear == feature_importance_model).all())
        self.assertTrue((feature_importance_null == 0).all())
