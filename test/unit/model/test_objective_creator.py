# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest

from openstf.model.objective_creator import ObjectiveCreator
from test.utils.base import BaseTestCase


class TestObjectiveCreator(BaseTestCase):
    def test_create_objective_happy(self):
        """Test if all defined objected can be created."""

        for model_type, objective_class in ObjectiveCreator.OBJECTIVES.items():
            objective = ObjectiveCreator.create_objective(model_type)
            self.assertEqual(objective, objective_class)

    def test_create_objective_not_implemented_model_type(self):
        """Test if a nice error is returned if a not-implemented model is requested"""
        model_type = "AnUnimplementedModelType"
        with self.assertRaises(NotImplementedError):
            ObjectiveCreator.create_objective(model_type)


if __name__ == "__main__":
    unittest.main()
