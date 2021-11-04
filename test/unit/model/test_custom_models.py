# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from typing import List, Type
import numpy as np
import unittest

from openstf.model.regressors.custom_regressor import (
    CustomOpenstfRegressor,
    create_custom_model,
    create_custom_objective,
)
from openstf.model.objective import RegressorObjective
from openstf.model.objective_creator import ObjectiveCreator
from openstf.model.model_creator import ModelCreator
from openstf.enums import MLModelType


class DummyObjective(RegressorObjective):
    ...


class DummyRegressor(CustomOpenstfRegressor):
    @staticmethod
    def valid_kwargs() -> List[str]:
        return []

    @property
    def objective(self) -> Type[DummyObjective]:
        return DummyObjective

    @property
    def feature_names(self):
        return self._feature_names

    def fit(self, X, y, **fit_params):
        self._feature_names = list(X.columns)
        return self

    def predict(self, X, **kwargs):
        return np.zeros(len(X))


class InvalidRegressor:
    pass


class TestCustomModels(unittest.TestCase):
    def test_create_custom_model(self):
        model_path = __name__ + ".DummyRegressor"
        model = create_custom_model(model_path)
        assert isinstance(model, DummyRegressor)

        model = ModelCreator().create_model(
            MLModelType.CUSTOM, custom_model_path=model_path
        )
        assert isinstance(model, DummyRegressor)

        with self.assertRaises(AttributeError):
            model_path = __name__ + ".UnknownRegressor"
            create_custom_model(model_path)

        with self.assertRaises(ValueError):
            model_path = __name__ + ".InvalidRegressor"
            create_custom_model(model_path)

    def test_create_custom_objective(self):
        model = DummyRegressor()
        obj = create_custom_objective(model, None)

        assert isinstance(obj, DummyObjective)

        obj = ObjectiveCreator().create_objective(MLModelType.CUSTOM)(model, None)

        assert isinstance(obj, DummyObjective)
