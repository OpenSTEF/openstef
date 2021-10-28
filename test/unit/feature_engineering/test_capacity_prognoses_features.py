# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest

import openstf.feature_engineering.capacity_prognoses_features as cf
from test.utils import TestData, BaseTestCase

data = TestData.load("input_data_train.pickle")
load_profile_names = TestData.load("input_data_train_load_profile_names.json")


class TestCapacityPrognosesFeatures(BaseTestCase):
    def setUp(self):
        super().setUp()
        self.data = TestData.load("input_data_train.pickle")

    def test_happy_apply_resample(self):
        result = cf.apply_resample(self.data, {"load": ["mean"]}, "1D")
        self.assertEqual(str(result.index.freq), "<Day>")
        self.assertEqual(result.columns, ["load_mean"])

    def test_happy_apply_calender_features(self):
        result = cf.apply_calender_features(self.data)
        self.assertEqual(len(result), len(self.data))

    def test_happy_apply_lag_features(self, lag_features=["load"], lag_times=[1]):
        result, lags = cf.apply_lag_features(self.data, lag_features, lag_times)
        self.assertIn("load-1", result.columns)

    def test_happy_apply_capacity_features(self):
        d, classes = cf.apply_capacity_features(
            self.data,
            "load_mean",
            [1, 2],
            apply_class_labels=True,
            outlier_removal=True,
            load_profile_names=load_profile_names,
        )
        return d, classes


# Run all tests
if __name__ == "__main__":
    unittest.main()
