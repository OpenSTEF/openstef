# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from unittest import TestCase

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
