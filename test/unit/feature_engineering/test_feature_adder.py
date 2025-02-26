# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData

import numpy as np

from openstef.feature_engineering.feature_adder import (
    FeatureAdder,
    FeatureDispatcher,
    adders_from_modules,
)


class DummyFeature(FeatureAdder):
    @property
    def name(self):
        return "dummy"

    @property
    def _regex(self):
        return r"dummy_(?P<value>[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+))"

    def required_features(self, feature_names):
        return []

    def apply_features(self, df, parsed_features):
        new_features = {}
        for feature_name, params in parsed_features:
            new_features[feature_name] = float(params["value"]) * np.ones(df.shape[0])
        return df.assign(**new_features)


class DummyIntFeature(FeatureAdder):
    @property
    def name(self):
        return "dummy int"

    @property
    def _regex(self):
        return r"dummy_(?P<value>[+-]?([0-9]*))$"

    def required_features(self, feature_names):
        return []

    def apply_features(self, df, parsed_features):
        new_features = {}
        for feature_name, params in parsed_features:
            new_features[feature_name] = int(params["value"]) * np.ones(df.shape[0])
        return df.assign(**new_features)


class TestFeatureAdder(BaseTestCase):
    def setUp(self) -> None:
        self.input_data = TestData.load("input_data.csv")

    def test_repr_fearureAdder(self):
        dummy = DummyFeature()
        self.assertTrue(dummy.__repr__(), "DummyFeature(<dummy>)")

    def test_ambiguous_features(self):
        feature_names = ["dummy_0", "dummy_-1", "dummy_0.5", "dummy_42"]
        feat_disp = FeatureDispatcher([DummyFeature(), DummyIntFeature()])
        with self.assertRaises(RuntimeError):
            feat_disp.dispatch_features(feature_names)

    def test_duplicate_featureAdder(self):
        with self.assertRaises(RuntimeError):
            FeatureDispatcher([DummyFeature(), DummyFeature()])

    def test_dispatch_features(self):
        feature_names = ["dummy_0", "dummy_-1", "dummy_0.5", "dummy_42"]
        feat_disp = FeatureDispatcher([DummyFeature()])
        df_out = feat_disp.apply_features(self.input_data, feature_names)
        # Test if the features have been correctly added
        self.assertEqual(
            set(feature_names + list(self.input_data.columns)), set(df_out.columns)
        )
        self.assertTrue((df_out["dummy_0"] == 0).all())
        self.assertTrue((df_out["dummy_-1"] == -1).all())
        self.assertTrue((df_out["dummy_0.5"] == 0.5).all())
        self.assertTrue((df_out["dummy_42"] == 42).all())

    def test_load_modules(self):
        adders = adders_from_modules(
            ["test.unit.feature_engineering.test_feature_adder"]
        )
        adders_type = [type(adder) for adder in adders]
        self.assertEqual(len(adders), 2)
        self.assertIn(DummyFeature, adders_type)
        self.assertIn(DummyIntFeature, adders_type)
