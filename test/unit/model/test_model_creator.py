# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from unittest import TestCase

from openstf.enums import MLModelType
from openstf.model.model_creator import ModelCreator
from openstf.model.regressors.regressor_interface import OpenstfRegressorInterface
from openstf.model.regressors.xgb_quantile import XGBQuantileOpenstfRegressor


class TestModelCreator(TestCase):
    def test_create_model_happy_flow(self):
        # Test happy flow (both str and enum model_type arguments)
        valid_types = [t.value for t in MLModelType] + [t for t in MLModelType]
        for model_type in valid_types:
            model = ModelCreator.create_model(model_type)
            self.assertIsInstance(model, OpenstfRegressorInterface)

    def test_create_model_quantile_model(self):
        # Test if quantile model is properly returned
        model_type = MLModelType.XGB_QUANTILE
        quantiles = tuple([0.5, 0.2, 0.5])
        # Create relevant model
        model = ModelCreator.create_model(model_type, quantiles=quantiles)

        self.assertIsInstance(model, OpenstfRegressorInterface)
        self.assertIsInstance(model, XGBQuantileOpenstfRegressor)
        self.assertEqual(model.quantiles, quantiles)

    def test_create_model_unknown_model(self):
        # Test if NotImplementedError is raised when model type is unknown
        model_type = "Unknown"
        with self.assertRaises(NotImplementedError):
            ModelCreator.create_model(model_type)
