# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from unittest import TestCase

from xgboost import XGBRegressor

from openstf.model.xgb_quantile import XGBQuantileRegressor

from openstf.model.model_creator import ModelCreator


class TestModelCreator(TestCase):
    def setUp(self) -> None:
        self.pj = {"model": "xgb", "quantiles": [0.5, 0.2, 0.5]}

    def test_create_model_happy_flow(self):
        # Test happy flow
        model = ModelCreator.create_model(self.pj)

        self.assertIsInstance(model, XGBRegressor)

    def test_create_model_quantile_model(self):
        # Test if quantile model is properly returned
        self.pj["model"] = "xgb_quantile"
        # Create relevant model
        model = ModelCreator.create_model(self.pj)

        self.assertIsInstance(model, XGBQuantileRegressor)
        self.assertEqual(model.quantiles, tuple(self.pj["quantiles"]))

    def test_create_model_unknown_model(self):
        # Test if ValueError is raised when model type is unknown
        self.pj["model"] = "Unknown"

        with self.assertRaises(ValueError):
            ModelCreator.create_model(self.pj)
