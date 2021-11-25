# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from unittest import TestCase

from test.utils.data import TestData

from openstf.enums import MLModelType
from openstf.model.model_creator import ModelCreator
from openstf.model.regressors.regressor_interface import OpenstfRegressorInterface
from openstf.model.regressors.xgb_quantile import XGBQuantileOpenstfRegressor


class TestModelCreator(TestCase):
    def test_create_model_happy_flow(self):
        # Test happy flow (both str and enum model_type arguments)
        pj = TestData.get_prediction_job(pid=307)
        valid_types = [t.value for t in MLModelType] + [t for t in MLModelType]
        for model_type in valid_types:
            pj["model"] = model_type
            model = ModelCreator.create_model(pj)
            self.assertIsInstance(model, OpenstfRegressorInterface)

    def test_create_model_quantile_model(self):
        # Test if quantile model is properly returned
        pj = TestData.get_prediction_job(pid=307)
        pj["model"] = "xgb_quantile"
        # Create relevant model
        model = ModelCreator.create_model(pj)

        self.assertIsInstance(model, OpenstfRegressorInterface)
        self.assertIsInstance(model, XGBQuantileOpenstfRegressor)
        self.assertEqual(model.quantiles, pj["quantiles"])

    def test_create_model_unknown_model(self):
        # Test if NotImplementedError is raised when model type is unknown
        pj = TestData.get_prediction_job(pid=307)
        pj["model"] = "Unknown"
        with self.assertRaises(NotImplementedError):
            ModelCreator.create_model(pj)
