# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from unittest import TestCase

from sklearn.utils.estimator_checks import check_estimator

from openstf.model.xgb_quantile import XGBQuantileRegressor

import pandas as pd

from openstf.model.confidence_interval_applicator import ConfidenceIntervalApplicator


class MockModel:
    confidence_interval = pd.DataFrame()

    def predict(self, input, quantile):
        stdev_forecast = pd.DataFrame({"forecast": [5, 6, 7], "stdev": [0.5, 0.6, 0.7]})
        return stdev_forecast["stdev"].rename(quantile)


class TestXgbQuantile(TestCase):
    def setUp(self) -> None:
        self.quantiles = [0.9, 0.5, 0.6, 0.1]

    def test_sklearn_compliant(self):
        # Use sklearn build in check, this will raise an exception if some check fails
        # During these tests the fit and predict methods are elaborately tested
        # More info: https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html
        check_estimator(XGBQuantileRegressor(tuple(self.quantiles)))

    def test_quantile_loading(self):
        model = XGBQuantileRegressor(tuple(self.quantiles))
        self.assertEqual(model.quantiles, tuple(self.quantiles))

    def test_value_error_raised(self):
        # Check if Value Error is raised when 0.5 is not in the requested quantiles list
        with self.assertRaises(ValueError):
            XGBQuantileRegressor((0.2, 0.3, 0.6, 0.7))

    def test_predict_raises_valueerror_no_model_trained_for_quantile(self):
        # Test if value error is raised when model is not available
        with self.assertRaises(ValueError):
            model = XGBQuantileRegressor((0.2, 0.3, 0.5, 0.6, 0.7))
            model.predict("test_data", quantile=0.8)
