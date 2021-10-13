from test.utils import BaseTestCase
import numpy as np
from openstf.model.regressors.xgb import XGBOpenstfRegressor


class MockBooster:
    feature_names = ["a", "b", "c"]


class TestXGB(BaseTestCase):
    def test_feature_names_property(self):
        model = XGBOpenstfRegressor()
        model._Booster = MockBooster()
        feature_names = model.feature_names
        np.testing.assert_array_equal(
            feature_names, MockBooster.feature_names, "Feature names are not equal"
        )
