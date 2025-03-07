# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import sklearn
from sklearn.utils.estimator_checks import check_estimator

from openstef.feature_engineering.apply_features import apply_features
from openstef.model.regressors.flatliner import FlatlinerRegressor

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
        check_estimator(FlatlinerRegressor(quantiles=tuple(self.quantiles)))

    def test_quantile_fit(self):
        """Test happy flow of the training of model"""
        # Arrange
        model = FlatlinerRegressor()

        # Act
        model.fit(train_input.iloc[:, 1:], train_input.iloc[:, 0])

        # Assert
        # check if the model was fitted (raises NotFittedError when not fitted)
        self.assertIsNone(sklearn.utils.validation.check_is_fitted(model))

        # check if model is sklearn compatible
        self.assertIsInstance(model, sklearn.base.BaseEstimator)

        result: np.ndarray = model.predict(train_input.iloc[:, 1:])

        self.assertEqual(len(result), len(train_input.iloc[:, 1:]))
        self.assertTrue((result == 0).all())

    def test_get_feature_names_from_linear(self):
        # Arrange
        model = FlatlinerRegressor()
        model.feature_names_ = ["a", "b", "c"]

        # Act
        feature_importance = model._get_feature_importance_from_linear(quantile=0.5)

        # Assert
        self.assertTrue(
            (feature_importance == np.array([0, 0, 0], dtype=np.float32)).all()
        )
