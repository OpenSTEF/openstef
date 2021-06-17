import unittest

from test.utils.base import BaseTestCase

from openstf.enums import MLModelType
from openstf.model.objective_creator import ObjectiveCreator
from openstf.model.objective import XGBRegressorObjective


class TestObjectiveCreator(BaseTestCase):

    def test_create_objective(self):
        Objective = ObjectiveCreator.create_objective(MLModelType.XGB)
        self.assertEqual(Objective, XGBRegressorObjective)


if __name__ == "__main__":
    unittest.main()