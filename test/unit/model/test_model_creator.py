# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import sys
from unittest import TestCase

from openstef.enums import ModelType
from openstef.model.model_creator import ModelCreator
from openstef.model.regressors.regressor import OpenstefRegressor
from openstef.model.regressors.xgb_quantile import XGBQuantileOpenstefRegressor


class TestModelCreator(TestCase):
    def test_create_model_happy_flow(self):
        # Test happy flow (both str and enum model_type arguments)
        valid_types = [t.value for t in ModelType] + [t for t in ModelType]
        for model_type in valid_types:
            model = ModelCreator.create_model(model_type)
            self.assertIsInstance(model, OpenstefRegressor)
            self.assertTrue(hasattr(model, "can_predict_quantiles"))
            if model_type in [
                "xgb_quantile",
                ModelType("xgb_quantile"),
                "arima",
                ModelType("arima"),
                "linear_quantile",
                ModelType("linear_quantile"),
                "gblinear_quantile",
                ModelType("gblinear_quantile"),
                "xgb_multioutput_quantile",
                ModelType("xgb_multioutput_quantile"),
                "flatliner",
                ModelType("flatliner"),
            ]:
                self.assertTrue(model.can_predict_quantiles)
            else:
                self.assertFalse(model.can_predict_quantiles)
            # Assert model has .score method - used in training to compare to old model
            assert callable(getattr(model, "score", None))

    def test_create_model_quantile_model(self):
        # Test if quantile model is properly returned
        model_type = ModelType.XGB_QUANTILE
        quantiles = tuple([0.5, 0.2, 0.5])
        # Create relevant model
        model = ModelCreator.create_model(model_type, quantiles=quantiles)

        self.assertIsInstance(model, OpenstefRegressor)
        self.assertIsInstance(model, XGBQuantileOpenstefRegressor)
        self.assertEqual(model.quantiles, quantiles)

    def test_create_model_unknown_model(self):
        # Test if NotImplementedError is raised when model type is unknown
        model_type = "Unknown"
        with self.assertRaises(NotImplementedError):
            ModelCreator.create_model(model_type)
