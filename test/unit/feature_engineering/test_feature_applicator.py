# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from test.unit.utils.data import TestData
from unittest import TestCase

import numpy as np

from openstef.feature_engineering.feature_applicator import (
    OperationalPredictFeatureApplicator,
    TrainFeatureApplicator,
)


class TestFeatureApplicator(TestCase):
    def setUp(self) -> None:
        self.input_data = TestData.load("input_data.csv")

    def test_train_feature_applicator_correct_order(self):
        # Test for expected column order of the output
        data_with_features = TrainFeatureApplicator(horizons=[0.25, 24.0]).add_features(
            self.input_data[["load"]]
        )
        self.assertEqual(data_with_features.columns.to_list()[0], "load")
        self.assertEqual(data_with_features.columns.to_list()[-1], "horizon")

    def test_train_feature_applicator_filter_features(self):
        # Test for expected column order of the output
        # Also check "horizons" is not in the output
        features = self.input_data.columns.to_list()[:15]
        data_with_features = TrainFeatureApplicator(
            horizons=[0.25, 24.0], feature_names=features
        ).add_features(self.input_data)

        self.assertIn("horizon", data_with_features.columns.to_list())
        self.assertListEqual(
            list(np.sort(features + ["horizon"])),
            list(np.sort(data_with_features.columns.to_list())),
        )

    def test_train_feature_applicator_custom_horizon(self):
        input_data = self.input_data.copy(deep=True)
        input_data["custom_horizon"] = 0
        data_with_features = TrainFeatureApplicator(
            horizons="custom_horizon"
        ).add_features(input_data)
        self.assertEqual(data_with_features.columns.to_list()[0], "load")
        self.assertEqual(data_with_features.columns.to_list()[-1], "horizon")
        self.assertTrue(
            (data_with_features["horizon"] == input_data["custom_horizon"]).all()
        )

    def test_operational_feature_applicator_correct_order(self):
        # Test for expected column order of the output
        # Also check "horizons" is not in the output
        data_with_features = OperationalPredictFeatureApplicator(
            horizons=[0.25]
        ).add_features(self.input_data[["load"]])
        self.assertEqual(data_with_features.columns.to_list()[0], "load")
        self.assertNotIn("horizon", data_with_features.columns.to_list())

    def test_operational_feature_applicator_one_horizon(self):
        # Test for expected column order of the output
        # Also check "horizons" is not in the output
        with self.assertRaises(ValueError):
            OperationalPredictFeatureApplicator(horizons=[0.25, 1.0]).add_features(
                self.input_data[["load"]]
            )
        with self.assertRaises(ValueError):
            OperationalPredictFeatureApplicator(horizons=[]).add_features(
                self.input_data[["load"]]
            )

    def test_operational_feature_applicator_filter_features(self):
        # Test for expected column order of the output
        # Also check "horizons" is not in the output
        features = self.input_data.columns.to_list()
        data_with_features = OperationalPredictFeatureApplicator(
            horizons=[0.25], feature_names=features
        ).add_features(self.input_data[["load"]])

        self.assertListEqual(
            list(np.sort(features)), list(np.sort(data_with_features.columns.to_list()))
        )
