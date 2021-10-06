# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from unittest import TestCase

from openstf.feature_engineering.feature_applicator import (
    TrainFeatureApplicator,
    OperationalPredictFeatureApplicator,
)
from test.utils import TestData


class TestFeatureApplicator(TestCase):
    def test_train_feature_applicator_correct_order(self):
        # Test for expected column order of the output
        input_data = TestData.load("input_data.pickle")
        data_with_features = TrainFeatureApplicator(horizons=[0.25, 24.0]).add_features(
            input_data[["load"]]
        )
        self.assertEqual(data_with_features.columns.to_list()[0], "load")
        self.assertEqual(data_with_features.columns.to_list()[-1], "horizon")

    def test_operational_feature_applicator_correct_order_horizon_not_in_output(self):
        # Test for expected column order of the output
        # Also check "horizons" is not in the output
        input_data = TestData.load("input_data.pickle")
        requested_features = [
            "IsSunday",
            "IsWeekDay",
            "IsWeekendDay",
            "Month",
            "Quarter",
        ]

        data_with_features = OperationalPredictFeatureApplicator(
            horizons=[0.25], feature_names=requested_features
        ).add_features(input_data[["load"]])
        self.assertEqual(data_with_features.columns.to_list()[0], "load")
        self.assertTrue("horizon" not in data_with_features.columns.to_list())

    def test_operational_feature_applicator_correct_order_horizon_in_output(self):
        # Test for expected column order of the output
        # Also check "horizons" is not in the output
        input_data = TestData.load("input_data.pickle")
        requested_features = [
            "IsSunday",
            "IsWeekDay",
            "IsWeekendDay",
            "Month",
            "Quarter",
            "horizon",
        ]

        data_with_features = OperationalPredictFeatureApplicator(
            horizons=[0.25], feature_names=requested_features
        ).add_features(input_data[["load"]])
        self.assertEqual(data_with_features.columns.to_list()[0], "load")
        self.assertTrue("horizon" in data_with_features.columns.to_list())
        self.assertFalse(data_with_features["horizon"].isna().all())
