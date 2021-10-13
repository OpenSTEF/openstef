# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from test.utils import BaseTestCase
import numpy as np
from openstf.model.regressors.lgbm import LGBMOpenstfRegressor


class MockBooster:
    feature_names = ["a", "b", "c"]


class TestLGBM(BaseTestCase):
    def test_feature_names_property(self):
        model = LGBMOpenstfRegressor()
        model._Booster = MockBooster()
        feature_names = model.feature_names
        np.testing.assert_array_equal(
            feature_names, MockBooster.feature_names, "Feature names are not equal"
        )
