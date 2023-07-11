# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import sys
from unittest import TestCase

from openstef.enums import MLModelType
from openstef.model.model_creator import ModelCreator
from openstef.model.regressors.regressor import OpenstfRegressor
from openstef.model.regressors.xgb_quantile import XGBQuantileOpenstfRegressor


class TestModelCreator(TestCase):
    def test_create_model_happy_flow(self):
        # Test happy flow (both str and enum model_type arguments)
        valid_types = [t.value for t in MLModelType] + [t for t in MLModelType]
        for model_type in valid_types:
            # skip the optional proloaf model
            if model_type in [MLModelType.ProLoaf, "proloaf"]:
                continue
            model = ModelCreator.create_model(model_type)
            self.assertIsInstance(model, OpenstfRegressor)
            self.assertTrue(hasattr(model, "can_predict_quantiles"))
            if model_type in [
                "xgb_quantile",
                MLModelType("xgb_quantile"),
                "arima",
                MLModelType("arima"),
            ]:
                self.assertTrue(model.can_predict_quantiles)
            else:
                self.assertFalse(model.can_predict_quantiles)
            # Assert model has .score method - used in training to compare to old model
            assert callable(getattr(model, "score", None))

    def test_create_model_quantile_model(self):
        # Test if quantile model is properly returned
        model_type = MLModelType.XGB_QUANTILE
        quantiles = tuple([0.5, 0.2, 0.5])
        # Create relevant model
        model = ModelCreator.create_model(model_type, quantiles=quantiles)

        self.assertIsInstance(model, OpenstfRegressor)
        self.assertIsInstance(model, XGBQuantileOpenstfRegressor)
        self.assertEqual(model.quantiles, quantiles)

    def test_create_model_unknown_model(self):
        # Test if NotImplementedError is raised when model type is unknown
        model_type = "Unknown"
        with self.assertRaises(NotImplementedError):
            ModelCreator.create_model(model_type)


class TestWithoutProloaf(TestCase):
    """Explicitly test if trying to create a proloaf model when proloaf was not installed yields an ImportError"""

    def setUp(self):
        self._temp_proloaf = None
        if sys.modules.get("proloaf"):
            self._temp_proloaf = sys.modules["proloaf"]
        sys.modules["proloaf"] = None

    def tearDown(self):
        if self._temp_proloaf:
            sys.modules["proloaf"] = self._temp_proloaf
        else:
            del sys.modules["proloaf"]

    def tests_create_model_uninstalled_proloaf(self):
        with self.assertRaises(ImportError):
            ModelCreator.create_model("proloaf")
