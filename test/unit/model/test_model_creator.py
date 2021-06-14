# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from unittest import TestCase
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from openstf.model.xgb_quantile import XgbQuantile

from openstf.model.model_creator import ModelCreator


class TestModelCreator(TestCase):
    def setUp(self) -> None:
        self.pj = {"model": "xgb", "quantiles": [0.5, 0.2, 0.5]}

    def test_happy_flow(self):
        model = ModelCreator.create_model(self.pj)

        self.assertIsInstance(model, XGBRegressor)

    def test_quantile_model(self):

        self.pj["model"] = "xgb_quantile"
        # Create relevant model
        model = ModelCreator.create_model(self.pj)

        self.assertIsInstance(model, XgbQuantile)
        self.assertEqual(model.quantiles, tuple(self.pj["quantiles"]))
