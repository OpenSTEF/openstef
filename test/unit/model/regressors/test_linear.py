# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest

import numpy as np
import pandas as pd
import sklearn
from sklearn.utils.estimator_checks import check_estimator

from openstf.model.regressors.linear import LinearRTEOpenstfRegressor
from test.utils import BaseTestCase, TestData

train_input = TestData.load("reference_sets/307-train-data.csv")


class TestLinearRTEOpenstfRegressor(BaseTestCase):
    def test_sklearn_compliant(self):
        # Use sklearn build in check, this will raise an exception if some check fails
        # During these tests the fit and predict methods are elaborately tested
        # More info: https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html
        check_estimator(LinearRTEOpenstfRegressor())

    def test_fit(self):
        """Test happy flow of the training of model"""
        model = LinearRTEOpenstfRegressor()
        model.fit(train_input.iloc[:, 1:], train_input.iloc[:, 0])

        # check if the model was fitted (raises NotFittedError when not fitted)
        self.assertIsNone(sklearn.utils.validation.check_is_fitted(model))

        # check if model is sklearn compatible
        self.assertTrue(isinstance(model, sklearn.base.BaseEstimator))

    def test_non_null_columns_retrieval(self):
        n_sample = train_input.shape[0]
        columns = train_input.columns[1:]

        X = train_input.iloc[:, 1:].copy(deep=True)
        X["Empty"] = pd.Series(n_sample * [np.nan])

        model = LinearRTEOpenstfRegressor()
        model.fit(X, train_input.iloc[:, 0])

        # check if non null columns are well retrieved
        self.assertTrue((columns == model.estimator_.non_null_columns_).all())

        X_ = X.copy(deep=True)
        X_["Empty2"] = pd.Series(n_sample * [np.nan])

        # check the use of non null columns in predict
        self.assertTrue((model.predict(X_) == model.predict(X)).all())
        self.assertTrue(
            (model.predict(train_input.iloc[:, 1:]) == model.predict(X)).all()
        )

    def test_imputer(self):
        n_sample = train_input.shape[0]
        X = train_input.iloc[:, 1:].copy(deep=True)
        sp = np.ones(n_sample)
        sp[-1] = np.nan
        X["Sparse"] = sp
        model1 = LinearRTEOpenstfRegressor(imputation_strategy=None)

        with self.assertRaises(ValueError):
            model1.fit(X, train_input.iloc[:, 0])

        model2 = LinearRTEOpenstfRegressor(imputation_strategy="mean")
        model2.fit(X, train_input.iloc[:, 0])
        self.assertIsNone(sklearn.utils.validation.check_is_fitted(model2))

        X_ = pd.DataFrame(model2.estimator_.imputer_.transform(X), columns=X.columns)
        self.assertTrue((model2.predict(X_) == model2.predict(X)).all())
