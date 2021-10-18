# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from unittest import TestCase

from openstf.feature_engineering.feature_applicator import (
    TrainFeatureApplicator,
    OperationalPredictFeatureApplicator,
)

from test.utils.data import TestData


class TestFeatureApplicator(TestCase):
    def test_train_feature_applicator_correct_order(self):
        # Test for expected column order of the output
        input_data = TestData.load("input_data.pickle")
        data_with_features = TrainFeatureApplicator(horizons=[0.25, 24.0]).add_features(
            input_data[["load"]]
        )
        self.assertEqual(data_with_features.columns.to_list()[0], "load")
        self.assertEqual(data_with_features.columns.to_list()[-1], "horizon")

    def test_operational_feature_applicator_correct_order(self):
        # Test for expected column order of the output
        # Also check "horizons" is not in the output
        input_data = TestData.load("input_data.pickle")
        data_with_features = OperationalPredictFeatureApplicator(
            horizons=[0.25]
        ).add_features(input_data[["load"]])
        self.assertEqual(data_with_features.columns.to_list()[0], "load")
        self.assertTrue("horizon" not in data_with_features.columns.to_list())

    def test_train_feature_applicator_correct_order_historic_load(self):
        # Test for expected column order of the output and test for expected historic_load column
        input_data = TestData.load("input_data.pickle")
        input_data["T-x"] = input_data["load"]
        data_with_features = TrainFeatureApplicator(
            horizons=[0.25, 24.0], feature_names=["T-x"]
        ).add_features(input_data)
        self.assertTrue("historic_load" in data_with_features.columns.to_list())
        self.assertEqual(data_with_features.columns.to_list()[0], "load")
        self.assertEqual(data_with_features.columns.to_list()[-1], "horizon")

    def test_operational_feature_applicator_correct_order_historic_load(self):
        # Test for expected column order of the output and test for expected historic_load column
        # Also check "horizons" is not in the output
        input_data = TestData.load("input_data.pickle")
        input_data["T-x"] = input_data["load"]
        data_with_features = OperationalPredictFeatureApplicator(
            horizons=[0.25], feature_names=["T-x"]
        ).add_features(input_data)
        self.assertTrue("historic_load" in data_with_features.columns.to_list())
        self.assertEqual(data_with_features.columns.to_list()[0], "load")
        self.assertTrue("horizon" not in data_with_features.columns.to_list())
