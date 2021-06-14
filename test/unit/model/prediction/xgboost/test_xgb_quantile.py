# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from unittest import TestCase
import pytest

from sklearn.utils.estimator_checks import check_estimator

from openstf.model.xgb_quantile import XgbQuantile


class TestXgbQuantile(TestCase):
    def setUp(self) -> None:
        self.quantiles = [0.9, 0.5, 0.6, 0.1]

    def test_sklearn_compliant(self):
        # Use sklearn build in check, this will raise an exception if some check fails
        # During these tests the fit and predict methods are elaborately tested
        # More info: https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html
        check_estimator(XgbQuantile(tuple(self.quantiles)))

    def test_quantile_loading(self):
        model = XgbQuantile(tuple(self.quantiles))
        self.assertEqual(model.quantiles, tuple(self.quantiles))

    def test_value_error_raised(self):
        # Check if Value Error is raised when 0.5 is not in the requested quantiles list
        with pytest.raises(ValueError):
            XgbQuantile(tuple(0.2, 0.3, 0.6, 0.7))

    def test_value_error_raised(self):
        # Check if Value Error is raised when 0.5 is not in the requested quantiles list
        with pytest.raises(ValueError):
            XgbQuantile((0.2, 0.3, 0.6, 0.7))

    def test_value_error_raised_no_model_trained_for_quantile(self):
        # Test if value error is raised when model is not available
        with pytest.raises(ValueError):
            model = XgbQuantile((0.2, 0.3, 0.5, 0.6, 0.7))
            model.predict("test_data", quantile=0.8)
