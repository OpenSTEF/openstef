# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData

import numpy as np
import pandas as pd
import sklearn
from sklearn.utils.estimator_checks import check_estimator

from openstef.model.regressors.xgb_quantile import XGBQuantileOpenstfRegressor


class MockModel:
    confidence_interval = pd.DataFrame()

    def predict(self, input, quantile):
        stdev_forecast = pd.DataFrame({"forecast": [5, 6, 7], "stdev": [0.5, 0.6, 0.7]})
        return stdev_forecast["stdev"].rename(quantile)


class MockScore:
    def get(self, a, b):
        book = {"a": 12, "b": 23, "c": 36}

        return book[a] + b


class MockBooster:
    feature_names = ["a", "b", "c"]

    def get_score(self, importance_type):
        if importance_type == "gain":
            return MockScore()
        else:
            return MockScore()


train_input = TestData.load("reference_sets/307-train-data.csv")


class TestXgbQuantile(BaseTestCase):
    def setUp(self) -> None:
        self.quantiles = [0.9, 0.5, 0.6, 0.1]

    @unittest.skip  # Use this during development, this test requires not allowing nan vallues which we explicitly do allow.
    def test_sklearn_compliant(self):
        # Use sklearn build in check, this will raise an exception if some check fails
        # During these tests the fit and predict methods are elaborately tested
        # More info: https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html
        check_estimator(XGBQuantileOpenstfRegressor(tuple(self.quantiles)))

    def test_quantile_loading(self):
        model = XGBQuantileOpenstfRegressor(tuple(self.quantiles))
        self.assertEqual(model.quantiles, tuple(self.quantiles))

    def test_quantile_fit(self):
        """Test happy flow of the training of model"""
        model = XGBQuantileOpenstfRegressor()
        model.fit(train_input.iloc[:, 1:], train_input.iloc[:, 0])

        # check if the model was fitted (raises NotFittedError when not fitted)
        self.assertIsNone(sklearn.utils.validation.check_is_fitted(model))

        # check if model is sklearn compatible
        self.assertIsInstance(model, sklearn.base.BaseEstimator)

    def test_value_error_raised(self):
        # Check if Value Error is raised when 0.5 is not in the requested quantiles list
        with self.assertRaises(ValueError):
            XGBQuantileOpenstfRegressor((0.2, 0.3, 0.6, 0.7))

    def test_predict_raises_valueerror_no_model_trained_for_quantile(self):
        # Test if value error is raised when model is not available
        with self.assertRaises(ValueError):
            model = XGBQuantileOpenstfRegressor((0.2, 0.3, 0.5, 0.6, 0.7))
            model.predict("test_data", quantile=0.8)

    def test_set_params(self):
        # Check hyperparameters are set correctly and do not cause errors

        model = XGBQuantileOpenstfRegressor((0.2, 0.3, 0.5, 0.6, 0.7))

        hyperparams = {
            "subsample": "0.9",
            "min_child_weight": "4",
            "max_depth": "4",
            "gamma": "0.37879654",
            "colsample_bytree": "0.78203051",
            "training_period_days": "90",
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

    def test_get_feature_names_from_booster(self):
        # Check if feature importance is extracted corretly
        model = XGBQuantileOpenstfRegressor((0.2, 0.3, 0.5, 0.6, 0.7))
        self.assertTrue(
            (
                model.get_feature_importances_from_booster(MockBooster())
                == np.array([0.16901408, 0.32394367, 0.5070422], dtype=np.float32)
            ).all()
        )

    def test_feature_names_property(self):
        model = XGBQuantileOpenstfRegressor((0.2, 0.3, 0.5, 0.6, 0.7))
        model._Booster = MockBooster()
        features_names = model.feature_names
        np.testing.assert_array_equal(
            features_names, MockBooster.feature_names, "Feature names are not equal"
        )

    def test_importance_names(self):
        model = XGBQuantileOpenstfRegressor(tuple(self.quantiles))
        self.assertIsInstance(model._get_importance_names(), dict)
