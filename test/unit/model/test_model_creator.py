# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from unittest import TestCase

from xgboost import XGBRegressor

from openstf.model.regressors.xgb_quantile import XGBQuantileStfRegressor

from openstf.model.model_creator import ModelCreator
from openstf.enums import MLModelType


class TestModelCreator(TestCase):
    def test_create_model_happy_flow(self):
        # Test happy flow
        model_type = "xgb"
        model = ModelCreator.create_model(model_type)

        self.assertIsInstance(model, XGBRegressor)

    def test_create_model_quantile_model(self):
        # Test if quantile model is properly returned
        model_type = "xgb_quantile"
        quantiles = tuple([0.5, 0.2, 0.5])
        # Create relevant model
        model = ModelCreator.create_model(model_type, quantiles=quantiles)

        self.assertIsInstance(model, XGBQuantileStfRegressor)
        self.assertEqual(model.quantiles, quantiles)

    def test_create_model_unknown_model(self):
        # Test if ValueError is raised when model type is unknown
        model_type = "Unknown"

        with self.assertRaises(ValueError):
            ModelCreator.create_model(model_type)


class TestMLModelInterfaces(TestCase):
    """Test if all ml models defined in openstf.model.ml_model
    adhere to the expected interface"""

    def _check_expected_methods(self, cls):
        assert hasattr(cls, "predict")
        assert hasattr(cls, "fit")
        assert hasattr(cls, "score")
        assert hasattr(cls, "set_params")
        # assert hasattr(cls, 'valid_model_kwargs')  # TODO: determine if this really is required

    def test_ml_model_interfaces(self):
        # Loop over all specified model types
        for model_type in MLModelType:
            model = ModelCreator.create_model(model_type)
            try:
                self._check_expected_methods(model)
            except AssertionError as e:
                raise AssertionError(
                    f"Required method not found"
                    f" for model_type: {model}. Details: {e}"
                )
