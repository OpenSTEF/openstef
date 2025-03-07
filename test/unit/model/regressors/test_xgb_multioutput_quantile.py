# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest
from test.unit.model.regressors.test_xgb_quantile import MockBooster
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData

import numpy as np
import pandas as pd
import sklearn
from sklearn.utils.estimator_checks import check_estimator

from openstef.model.regressors.xgb_multioutput_quantile import (
    XGBMultiOutputQuantileOpenstfRegressor,
)

train_input = TestData.load("reference_sets/307-train-data.csv")


class TestXgbMultiOutputQuantile(BaseTestCase):
    def setUp(self) -> None:
        self.quantiles = [0.9, 0.5, 0.6, 0.1]

    @unittest.skip  # Use this during development, this test requires not allowing nan vallues which we explicitly do allow.
    def test_sklearn_compliant(self):
        # Use sklearn build in check, this will raise an exception if some check fails
        # During these tests the fit and predict methods are elaborately tested
        # More info: https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html
        check_estimator(XGBMultiOutputQuantileOpenstfRegressor(tuple(self.quantiles)))

    def test_quantile_loading(self):
        model = XGBMultiOutputQuantileOpenstfRegressor(tuple(self.quantiles))
        self.assertEqual(model.quantiles, tuple(self.quantiles))

    def test_quantile_fit(self):
        """Test happy flow of the training of model"""
        model = XGBMultiOutputQuantileOpenstfRegressor()
        model.fit(train_input.iloc[:, 1:], train_input.iloc[:, 0])

        # check if the model was fitted (raises NotFittedError when not fitted)
        self.assertIsNone(sklearn.utils.validation.check_is_fitted(model))

        # check if model is sklearn compatible
        self.assertTrue(isinstance(model, sklearn.base.BaseEstimator))

        # check if model has feature names
        self.assertIsNotNone(model.feature_names)

        # check if model has feature importance
        self.assertIsNotNone(model.feature_importances_)
        self.assertTrue((model.feature_importances_ > 0).any())

    def test_quantile_predict(self):
        """Test happy flow of the training of model"""
        model = XGBMultiOutputQuantileOpenstfRegressor()
        model.fit(train_input.iloc[:, 1:], train_input.iloc[:, 0])

        result = model.predict(train_input.iloc[150:155, 1:], quantile=0.5)

        self.assertEqual(result.shape, (5,))

    def test_value_error_raised(self):
        # Check if Value Error is raised when 0.5 is not in the requested quantiles list
        with self.assertRaises(ValueError):
            XGBMultiOutputQuantileOpenstfRegressor((0.2, 0.3, 0.6, 0.7))

    def test_predict_raises_valueerror_no_model_trained_for_quantile(self):
        # Test if value error is raised when model is not available
        with self.assertRaises(ValueError):
            model = XGBMultiOutputQuantileOpenstfRegressor((0.2, 0.3, 0.5, 0.6, 0.7))
            model.predict("test_data", quantile=0.8)

    def test_set_params(self):
        # Check hyperparameters are set correctly and do not cause errors

        model = XGBMultiOutputQuantileOpenstfRegressor(
            quantiles=(0.2, 0.3, 0.5, 0.6, 0.7),
            arctan_smoothing=0.42,
        )

        hyperparams = {
            "subsample": "0.9",
            "min_child_weight": "4",
            "max_depth": "4",
            "gamma": "0.37879654",
            "colsample_bytree": "0.78203051",
            "training_period_days": "90",
            "arctan_smoothing": "0.42",
        }
        valid_hyper_parameters = {
            key: value
            for key, value in hyperparams.items()
            if key in model.get_params().keys()
        }

        model.set_params(**valid_hyper_parameters)

        # Check if vallues are properly set
        self.assertEqual(model.max_depth, hyperparams["max_depth"])
        self.assertFalse(hasattr(model, "training_period_days"))

    def test_importance_names(self):
        model = XGBMultiOutputQuantileOpenstfRegressor(tuple(self.quantiles))
        self.assertIsInstance(model._get_importance_names(), dict)
