# SPDX-FileCopyrightText: 2017-2023 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData

import numpy as np
import pandas as pd
import sklearn

from openstef.model.regressors.arima import ARIMAOpenstfRegressor


class TestARIMAOpenstfRegressor(BaseTestCase):
    def setUp(self):
        self.train_input = TestData.load("reference_sets/307-train-data.csv")

    def test_fit(self):
        """Test happy flow of the training of model"""
        model = ARIMAOpenstfRegressor()
        model.fit(self.train_input.iloc[:150, 1:], self.train_input.iloc[:150, 0])

        # check if the model was fitted (raises NotFittedError when not fitted)
        self.assertIsNone(sklearn.utils.validation.check_is_fitted(model))

        # check if model is sklearn compatible
        self.assertIsInstance(model, sklearn.base.BaseEstimator)

    def test_update_historic_data(self):
        """Test happy flow of the update of historic data"""
        model = ARIMAOpenstfRegressor()
        model.fit(self.train_input.iloc[:150, 1:], self.train_input.iloc[:150, 0])
        model.update_historic_data(
            self.train_input.iloc[150:155, 1:], self.train_input.iloc[150:155, 0]
        )

        # check if the model was fitted (raises NotFittedError when not fitted)
        self.assertIsNone(sklearn.utils.validation.check_is_fitted(model))

        # check if model is sklearn compatible
        self.assertIsInstance(model, sklearn.base.BaseEstimator)

    def test_predict_wrong_historic(self):
        """Test the prediction with the wrong historic (missing data)"""
        model = ARIMAOpenstfRegressor()
        model.fit(self.train_input.iloc[:150, 1:], self.train_input.iloc[:150, 0])

        # check the prediction with wrong historic (missing data before starting forcast date)
        with self.assertRaises(ValueError):
            model.predict(self.train_input.iloc[155:160, 1:])

    def test_predict(self):
        """Test the prediction"""
        model = ARIMAOpenstfRegressor()
        model.fit(self.train_input.iloc[:150, 1:], self.train_input.iloc[:150, 0])

        forecast = model.results_.forecast(
            steps=5, exog=self.train_input.iloc[150:155, 1:]
        )
        pred = model.predict(self.train_input.iloc[150:155, 1:])

        # check the prediction with start and end inferred from the x dataframe
        self.assertTrue((forecast == pred).all())

    def test_predict_quantile(self):
        """Test the quantile prediction"""
        model = ARIMAOpenstfRegressor()
        model.fit(self.train_input.iloc[:150, 1:], self.train_input.iloc[:150, 0])

        forecast_ci = model.results_.get_forecast(
            steps=5, exog=self.train_input.iloc[150:155, 1:]
        ).conf_int(0.5)
        pred_q5 = model.predict(self.train_input.iloc[150:155, 1:], quantile=0.05)
        pred_q95 = model.predict(self.train_input.iloc[150:155, 1:], quantile=0.95)

        # check the quantile predictions
        self.assertTrue((forecast_ci["lower load"] == pred_q5).all())
        self.assertTrue((forecast_ci["upper load"] == pred_q95).all())

    @unittest.skip  # Skip because not working in the CI/CD, not sure why ...
    def test_set_feature_importance_from_arima(self):
        """Test the set of feature importance"""
        model = ARIMAOpenstfRegressor()
        model.fit(self.train_input.iloc[:150, 1:], self.train_input.iloc[:150, 0])

        params = model.results_.params
        pvalues = model.results_.pvalues
        importances = model.set_feature_importance()

        # check the retrieval of feature importance
        self.assertTrue(np.allclose(params, importances["weight"]))
        self.assertTrue(np.allclose(pvalues, importances["gain"]))

    def test_score_backtest(self):
        model = ARIMAOpenstfRegressor(backtest_max_horizon=180)
        model.fit(self.train_input.iloc[:150, 1:], self.train_input.iloc[:150, 0])

        score_r2 = model.score(
            self.train_input.iloc[:150, 1:], self.train_input.iloc[:150, 0]
        )
        self.assertLessEqual(score_r2, 1.0)
        self.assertGreaterEqual(score_r2, 0.5)
